"""Train a LoRA adapter on the base model. Run on a free GPU (see docs/KAGGLE.md).

    python -m rail_extract.train_lora --overfit --out runs/overfit
    python -m rail_extract.train_lora --config C1 --seed 1 --out runs/C1-s1

A checkpoint is written every --ckpt-every steps and at the end; running the same command again
resumes from it (with --precision fp16 the loss scale restarts, so only bf16 resumes bit-exactly).
A resumed bf16 run is identical to an uninterrupted one: same batches, same random
state. The overfit test comes first (PREREGISTRATION.md section 4); it trains on 8 passages and
must reproduce them with field micro-F1 of at least 0.90 and every output schema-valid.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from rail_extract import scoring as S
from rail_extract import training as T
from rail_extract.data import load_split
from rail_extract.predict import BASE_MODEL, BASE_REVISION

CKPT = "training_state.pt"


def load_for_training(
    cfg: T.TrainConfig, model: str, revision: str | None, seed: int, precision: str = "bf16"
):
    """Tokenizer and PEFT-wrapped model. LoRA weights are initialised from `seed`.

    `precision` is the base-model dtype on a GPU: "bf16" (default) or "fp16". The T4 has no
    native bf16, so fp16 can be much faster there. LoRA weights stay in float32."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    dtype = gpu_dtype if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    net = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=dtype).to(device)
    net.config.use_cache = False
    net.gradient_checkpointing_enable()
    net.enable_input_require_grads()
    torch.manual_seed(seed)
    peft_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.targets),
        bias="none",
    )
    return tok, get_peft_model(net, peft_cfg), device


def encode(rows: list[dict], tok) -> tuple[list[dict], list[str]]:
    def render(messages: list[dict]) -> str:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def tokenize(text: str) -> list[int]:
        return tok(text, add_special_tokens=False).input_ids

    return T.encode_rows(rows, render, tokenize, tok.eos_token_id)


def _save_state(out: Path, net, opt, step: int, elapsed: float, losses: list[float]) -> None:
    import torch

    state = {
        "step": step,
        "elapsed": elapsed,
        "losses": losses,
        "weights": {k: v.cpu() for k, v in net.named_parameters() if v.requires_grad},
        "optimizer": opt.state_dict(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "rng_python": list(random.getstate()[1]),
    }
    tmp = out / (CKPT + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, out / CKPT)  # a crash during the save never damages the last checkpoint


def _load_state(out: Path, net, opt) -> tuple[int, float, list[float]]:
    import torch

    path = out / CKPT
    if not path.exists():
        return 0, 0.0, []
    state = torch.load(path, map_location="cpu", weights_only=False)
    missing = net.load_state_dict(state["weights"], strict=False)
    if missing.unexpected_keys:
        raise RuntimeError(f"checkpoint does not fit the model: {missing.unexpected_keys[:3]}")
    opt.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng_torch"])
    if torch.cuda.is_available() and state["rng_cuda"]:
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    rng = random.getstate()
    random.setstate((rng[0], tuple(state["rng_python"]), rng[2]))
    return state["step"], state["elapsed"], state["losses"]


def accumulate(
    net, examples, idx, micro_batch, pad, device, use_amp=False, amp_dtype=None, scaler=None
) -> float:
    """Backward pass for one batch, run in micro-batches (memory: the logits are 100k wide).

    Each micro-batch loss is weighted by its share of the batch's target tokens, so the summed
    gradient equals the gradient of the whole batch's mean token loss. Returns that loss."""
    import torch

    n_tokens = sum(sum(t != T.IGNORE for t in examples[i]["labels"]) for i in idx)
    batch_loss = 0.0
    for start in range(0, len(idx), micro_batch):
        chunk = idx[start : start + micro_batch]
        enc = {
            k: torch.tensor(v, device=device)
            for k, v in T.collate([examples[i] for i in chunk], pad).items()
        }
        share = sum(sum(t != T.IGNORE for t in examples[i]["labels"]) for i in chunk) / n_tokens
        with torch.autocast(device_type=device, dtype=amp_dtype or torch.bfloat16, enabled=use_amp):
            loss = net(**enc).loss
        scaled = loss * share
        (scaler.scale(scaled) if scaler else scaled).backward()
        batch_loss += float(loss.detach()) * share
    return batch_loss


def train_loop(
    cfg: T.TrainConfig,
    net,
    tok,
    examples: list[dict],
    out: Path,
    seed: int,
    device: str,
    ckpt_every: int = 50,
    stop_after: int | None = None,
    precision: str = "bf16",
) -> dict:
    """Train; `stop_after` ends early after that step, as a crash would (used by the tests)."""
    import torch

    out.mkdir(parents=True, exist_ok=True)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = T.steps_per_epoch(len(examples), cfg.batch_size) * cfg.epochs
    random.seed(seed)
    start, elapsed, losses = _load_state(out, net, opt)
    if start:
        print(f"resuming at step {start} of {total}", flush=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    net.train()
    t0 = time.monotonic()
    use_amp = device == "cuda"
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    # fp16 needs loss scaling so small gradients do not underflow; bf16 does not
    scaler = torch.amp.GradScaler("cuda") if use_amp and precision == "fp16" else None
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for step in range(start, total):
        for group in opt.param_groups:
            group["lr"] = cfg.lr * T.lr_factor(step, total, cfg.warmup)
        idx = T.batch_indices(step, len(examples), cfg.batch_size, seed)
        batch_loss = accumulate(
            net, examples, idx, cfg.micro_batch, pad, device, use_amp, amp_dtype, scaler
        )
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(batch_loss)
        done = step + 1
        if done % 10 == 0 or done == total:
            mean = sum(losses[-10:]) / len(losses[-10:])
            print(f"step {done}/{total} loss {mean:.4f}", flush=True)
        if done % ckpt_every == 0 or done == total or done == stop_after:
            _save_state(out, net, opt, done, elapsed + time.monotonic() - t0, losses)
        if done == stop_after:
            break
    ran = len(losses) - start
    return {
        "steps": total,
        "precision": precision,
        "train_seconds": round(elapsed + time.monotonic() - t0, 1),
        "seconds_per_step": round((time.monotonic() - t0) / max(1, ran), 2),
        "final_loss": sum(losses[-10:]) / max(1, len(losses[-10:])),
        "losses": [round(x, 5) for x in losses],
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if use_amp else None,
        "gpu": torch.cuda.get_device_name(0) if use_amp else None,
    }


def save_adapter(net, out: Path) -> int:
    """Write the adapter and return its size in bytes."""
    path = out / "adapter"
    net.save_pretrained(path)
    return sum(f.stat().st_size for f in path.iterdir())


def run_overfit(net, tok, rows: list[dict], out: Path, max_new_tokens: int = 2048) -> dict:
    """Generate on the training passages and score against their labels."""
    from rail_extract.backends import HFBackend
    from rail_extract.predict import predict

    net.gradient_checkpointing_disable()
    net.config.use_cache = True
    backend = HFBackend.from_loaded(
        net, tok, "overfit", batch_size=4, max_new_tokens=max_new_tokens
    )
    records = predict("overfit", rows, backend)
    (out / "overfit_predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    scores = [
        S.score_passage(r, r["gold"], p, r["stratum"]) for r, p in zip(rows, records, strict=True)
    ]
    summary = S.summarise(scores).to_dict()
    return {"f1": summary["f1"], "schema_valid": summary["schema_valid"], "summary": summary}


def eval_on_val(net, tok, cfg_name: str, seed: int, root: Path | None = None) -> dict:
    """Generate on the validation split, unconstrained (L1) and schema-constrained (L2), and score.

    Writes predictions/<system>/val.jsonl and results/<system>.val.json like predict.py and
    evaluate.py do, so the later analysis reads them the same way. Validation only."""
    import datetime

    from rail_extract import labels as L
    from rail_extract.backends import HFBackend
    from rail_extract.evaluate import evaluate, format_summary
    from rail_extract.predict import predict, save

    root = root or L.ROOT
    net.gradient_checkpointing_disable()
    net.config.use_cache = True
    rows = load_split("val")
    report = {}
    for tag, constrained in (("L1", False), ("L2", True)):
        system = f"{tag}-{cfg_name}-s{seed}"
        backend = HFBackend.from_loaded(net, tok, system, constrained=constrained, batch_size=4)
        t0 = time.monotonic()
        records = predict(system, rows, backend, "v1", None)
        seconds = round(time.monotonic() - t0, 1)
        meta = {
            "system": system,
            "split": "val",
            "n": len(records),
            "backend": system,
            "prompt": "v1",
            "k": 0,
            "constrained": constrained,
            "thinking": False,
            "precision": "fp16",
            "generation_seconds": seconds,
            "date": datetime.date.today().isoformat(),
        }
        save(system, "val", records, meta, root)
        result = evaluate(system, "val", root=root)
        out = root / "results" / f"{system}.val.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(format_summary(result), f"\n  generation {seconds} s", flush=True)
        report[system] = result["summary"]["f1"]
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=list(T.CONFIGS))
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--revision", default=BASE_REVISION)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--val", action="store_true", help="after training, score on validation")
    ap.add_argument("--stop-after", type=int, help="stop after this many steps (speed tests)")
    a = ap.parse_args()
    if a.overfit == bool(a.config):
        raise SystemExit("give exactly one of --overfit and --config")

    cfg = T.OVERFIT if a.overfit else T.CONFIGS[a.config]
    rows = load_split("train")
    if a.overfit:
        rows = T.overfit_rows(rows)
    tok, net, device = load_for_training(cfg, a.model, a.revision, a.seed, a.precision)
    examples, dropped = encode(rows, tok)
    trainable, total = net.get_nb_trainable_parameters()
    print(f"{len(examples)} examples, {len(dropped)} dropped, device {device}", flush=True)
    print(f"trainable parameters {trainable:,} of {total:,}", flush=True)
    log = train_loop(
        cfg, net, tok, examples, a.out, a.seed, device, a.ckpt_every, a.stop_after, a.precision
    )
    log |= {
        "config": cfg.to_dict(),
        "seed": a.seed,
        "model": a.model,
        "revision": a.revision,
        "examples": len(examples),
        "dropped": dropped,
        "trainable_parameters": trainable,
        "adapter_bytes": save_adapter(net, a.out),
    }
    if a.overfit:
        result = run_overfit(net, tok, rows, a.out)
        log["overfit"] = result
        passed = T.overfit_passed(result)
        log["overfit_passed"] = passed
        print(
            f"OVERFIT {'PASSED' if passed else 'FAILED'}: "
            f"field F1 {result['f1']:.3f}, schema-valid {result['schema_valid']:.0%}",
            flush=True,
        )
    if a.val and not a.overfit:
        log["val_f1"] = eval_on_val(net, tok, cfg.name, a.seed)
    (a.out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(
        f"done in {log['train_seconds']} s; adapter {log['adapter_bytes'] / 1e6:.1f} MB", flush=True
    )


if __name__ == "__main__":
    main()
