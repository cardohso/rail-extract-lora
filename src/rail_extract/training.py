"""Training configuration and data preparation for the LoRA systems. No torch here, so CI tests it.

The eight configurations are fixed before any training run (PREREGISTRATION.md section 4: at most
eight, seed 1, scored on validation). C1 is the default; C2 to C7 change one thing; C8 combines
the larger settings. Training is on the train split only, with the zero-shot instruction `v1` and
thinking off, so training and inference see the same prompt.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass

from rail_extract import prompts

IGNORE = -100
MAX_LEN = 4096  # tokens; a longer example is dropped and counted, never truncated
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
ATTN_MLP = (*ATTN, "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class TrainConfig:
    name: str
    r: int = 16
    lr: float = 2e-4
    epochs: int = 4
    targets: tuple[str, ...] = ATTN
    batch_size: int = 4
    dropout: float = 0.05
    weight_decay: float = 0.0
    warmup: float = 0.1  # share of the steps with a linear ramp-up
    max_grad_norm: float = 1.0

    @property
    def alpha(self) -> int:
        return 2 * self.r

    def to_dict(self) -> dict:
        return asdict(self) | {"alpha": self.alpha}


CONFIGS = {
    c.name: c
    for c in (
        TrainConfig("C1"),
        TrainConfig("C2", lr=1e-4),
        TrainConfig("C3", lr=5e-4),
        TrainConfig("C4", r=8),
        TrainConfig("C5", r=32),
        TrainConfig("C6", epochs=8),
        TrainConfig("C7", targets=ATTN_MLP),
        TrainConfig("C8", r=32, lr=1e-4, epochs=8, targets=ATTN_MLP),
    )
}
OVERFIT = TrainConfig("overfit", lr=1e-3, epochs=60, dropout=0.0, warmup=0.05)
OVERFIT_N = 8
OVERFIT_MIN_F1 = 0.90  # fixed before the first run; below this the test fails


def overfit_rows(train_rows: list[dict]) -> list[dict]:
    """A fixed set of short train passages with empty, conditional and multi-requirement cases."""
    return prompts.select_examples(train_rows)[:OVERFIT_N]


def overfit_passed(summary: dict) -> bool:
    return summary["f1"] >= OVERFIT_MIN_F1 and summary["schema_valid"] == 1.0


def encode_example(
    prompt_ids: list[int], answer_ids: list[int], eos_id: int, max_len: int = MAX_LEN
) -> dict | None:
    """Input and labels for one example; the prompt is masked from the loss. None if too long."""
    tail = [*answer_ids, eos_id]
    if len(prompt_ids) + len(tail) > max_len:
        return None
    return {"input_ids": prompt_ids + tail, "labels": [IGNORE] * len(prompt_ids) + tail}


def encode_rows(
    rows: list[dict],
    render: Callable[[list[dict]], str],
    tokenize: Callable[[str], list[int]],
    eos_id: int,
    max_len: int = MAX_LEN,
) -> tuple[list[dict], list[str]]:
    """Encoded examples and the ids of those dropped for length.

    `render` turns chat messages into the generation-prompt string the model sees at inference."""
    out, dropped = [], []
    for r in rows:
        prompt = render(prompts.build_messages(r, "v1", None))
        ex = encode_example(tokenize(prompt), tokenize(prompts.answer(r["gold"])), eos_id, max_len)
        if ex is None:
            dropped.append(r["id"])
        else:
            out.append(ex)
    return out, dropped


def collate(batch: list[dict], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-padded lists; the caller makes tensors."""
    n = max(len(x["input_ids"]) for x in batch)
    return {
        "input_ids": [x["input_ids"] + [pad_id] * (n - len(x["input_ids"])) for x in batch],
        "labels": [x["labels"] + [IGNORE] * (n - len(x["labels"])) for x in batch],
        "attention_mask": [
            [1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"])) for x in batch
        ],
    }


def steps_per_epoch(n: int, batch_size: int) -> int:
    return math.ceil(n / batch_size)


def batch_indices(step: int, n: int, batch_size: int, seed: int) -> list[int]:
    """The example indices of a training step. A pure function of the step, so a resumed run sees
    exactly the batches an uninterrupted run would have seen."""
    per_epoch = steps_per_epoch(n, batch_size)
    epoch, pos = divmod(step, per_epoch)
    order = list(range(n))
    random.Random(f"{seed}:{epoch}").shuffle(order)
    return order[pos * batch_size : (pos + 1) * batch_size]


def lr_factor(step: int, total: int, warmup: float) -> float:
    """Linear ramp-up, then cosine decay towards zero."""
    ramp = max(1, int(total * warmup))
    if step < ramp:
        return (step + 1) / ramp
    progress = (step - ramp) / max(1, total - ramp)
    return 0.5 * (1 + math.cos(math.pi * progress))
