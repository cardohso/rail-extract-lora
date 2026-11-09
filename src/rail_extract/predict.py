"""Run one system over one split and save raw predictions (PREREGISTRATION.md section 5).

    uv run python -m rail_extract.predict --system R0 --split val
    uv run python -m rail_extract.predict --system B2 --split val --backend hf --k 5
    uv run python -m rail_extract.predict --system B3 --split val --backend hf --k 5 --constrained
    uv run python -m rail_extract.predict --system A1 --split val --backend gemini --model <id>
    uv run python -m rail_extract.predict --system R0 --split test --final

Writes predictions/<system>/<split>.jsonl (one record per passage, see scoring.py) and a
.meta.json beside it with the configuration. Scoring is a separate step (evaluate.py).
Few-shot examples come from the train split only.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from rail_extract import labels as L
from rail_extract import prompts, regex_baseline
from rail_extract import scoring as S
from rail_extract.backends import Backend, GeminiBackend, GroqBackend, HFBackend
from rail_extract.data import load_split

BASE_MODEL = "ibm-granite/granite-4.2-3b"
BASE_REVISION = "4e9c275e232f10211ec88644d106a1f6a1b60e06"


def build_backend(a: argparse.Namespace) -> Backend:
    if a.backend == "hf":
        return HFBackend(
            a.model or BASE_MODEL,
            a.revision or BASE_REVISION,
            constrained=a.constrained,
            thinking=a.thinking,
            batch_size=a.batch_size,
        )
    if a.backend == "gemini":
        return GeminiBackend(a.model or "gemini-3.5-flash-lite")
    if a.backend == "groq":
        return GroqBackend(a.model or "openai/gpt-oss-120b")
    raise SystemExit(f"backend '{a.backend}' needs --backend hf|gemini|groq")


def predict(
    system: str,
    rows: list[dict],
    backend: Backend | None,
    variant: str = "v1",
    examples: list[dict] | None = None,
) -> list[dict]:
    """Prediction records for `rows`. R0 needs no backend."""
    if system == "R0":
        raws = [regex_baseline.run(r) for r in rows]
    else:
        assert backend is not None
        raws = backend.generate([prompts.build_messages(r, variant, examples) for r in rows])
    return [S.make_prediction(r["id"], system, raw) for r, raw in zip(rows, raws, strict=True)]


def save(system: str, split: str, records: list[dict], meta: dict, root: Path = L.ROOT) -> Path:
    out = root / "predictions" / system
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{split}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (out / f"{split}.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", required=True, help="R0, B1, B2, B3, L1, L2, A1 (or a label)")
    ap.add_argument("--split", required=True, choices=L.SPLITS)
    ap.add_argument("--final", action="store_true", help="required for test and heldout")
    ap.add_argument("--backend", choices=["hf", "gemini", "groq"])
    ap.add_argument("--model")
    ap.add_argument("--revision")
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--k", type=int, default=0, help="few-shot examples from train")
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, help="first N passages only (smoke runs)")
    a = ap.parse_args()

    rows = load_split(a.split, a.final)[: a.limit]
    examples = None
    if a.k:
        examples = prompts.select_examples(load_split("train"))[: a.k]
    backend = None if a.system == "R0" else build_backend(a)
    records = predict(a.system, rows, backend, a.prompt, examples)
    meta = {
        "system": a.system,
        "split": a.split,
        "n": len(records),
        "backend": backend.name if backend else "regex",
        "prompt": a.prompt,
        "prompt_hash": prompts.prompt_hash(a.prompt, examples),
        "k": a.k,
        "example_ids": [e["id"] for e in examples or []],
        "constrained": a.constrained,
        "thinking": a.thinking,
        "date": datetime.date.today().isoformat(),
    }
    path = save(a.system, a.split, records, meta)
    print(f"wrote {len(records)} predictions to {path.relative_to(L.ROOT)}")


if __name__ == "__main__":
    main()
