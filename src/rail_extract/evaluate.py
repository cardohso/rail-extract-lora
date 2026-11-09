"""Score saved predictions. No model is run here, so every reported number can be reproduced from
the committed prediction files.

    uv run python -m rail_extract.evaluate --system R0 --split val
    uv run python -m rail_extract.evaluate --system R0 --split test --final

Writes results/<system>.<split>.json with the summary and the per-passage counts (the counts feed
the paired bootstrap in the analysis step).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rail_extract import labels as L
from rail_extract import scoring as S
from rail_extract.data import load_split


def read_predictions(system: str, split: str, root: Path = L.ROOT) -> dict[str, dict]:
    path = root / "predictions" / system / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"no predictions at {path}")
    return {r["id"]: r for r in L.read_jsonl(path)}


def evaluate(system: str, split: str, final: bool = False, root: Path = L.ROOT) -> dict:
    rows = load_split(split, final, root)
    preds = read_predictions(system, split, root)
    missing = [r["id"] for r in rows if r["id"] not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} passages have no prediction, e.g. {missing[0]}")
    scores = [S.score_passage(r, r["gold"], preds[r["id"]], r["stratum"]) for r in rows]
    summary = S.summarise(scores)
    return {
        "system": system,
        "split": split,
        "summary": summary.to_dict(),
        "passages": [s.to_dict() for s in scores],
    }


def format_summary(result: dict) -> str:
    s = result["summary"]
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    lines = [
        f"{result['system']} on {result['split']} (n={s['n']}, {s['empty_passages']} empty)",
        f"  field micro-F1 {s['f1']:.3f}  (P {s['precision']:.3f}, R {s['recall']:.3f}); "
        f"after verifier {s['f1_verified']:.3f}",
        "  per field F1: " + ", ".join(f"{k} {v['f1']:.2f}" for k, v in s["per_field"].items()),
        f"  exact match {pct(s['exact_match'])}  false extraction {pct(s['false_extraction'])}",
        f"  fidelity error: passages {pct(s['fidelity_error_passages'])}, "
        f"values {pct(s['fidelity_error_values'])}",
        f"  valid JSON {pct(s['json_valid'])}  schema-valid {pct(s['schema_valid'])}  "
        f"verbatim quotes {pct(s['verbatim_quote'])}  "
        f"coarse comparator {pct(s['coarse_comparator'])}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", required=True)
    ap.add_argument("--split", required=True, choices=L.SPLITS)
    ap.add_argument("--final", action="store_true", help="required for test and heldout")
    a = ap.parse_args()
    result = evaluate(a.system, a.split, a.final)
    out = L.ROOT / "results" / f"{a.system}.{a.split}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(format_summary(result))
    print(f"wrote {out.relative_to(L.ROOT)}")


if __name__ == "__main__":
    main()
