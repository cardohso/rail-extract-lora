"""Table of the LoRA configurations scored on validation (PREREGISTRATION.md section 5).

    uv run python -m rail_extract.tune_table

Reads results/L*-C*-s*.val.json and results/training/C*-s*.json. Nothing is run here."""

from __future__ import annotations

import json
import re

from rail_extract import labels as L


def rows(root=L.ROOT) -> list[dict]:
    out = []
    for path in sorted((root / "results").glob("L[12]-C*-s*.val.json")):
        result = json.loads(path.read_text())
        s = result["summary"]
        tag, cfg, seed = re.fullmatch(r"(L[12])-(C\d+)-s(\d+)", result["system"]).groups()
        log = root / "results" / "training" / f"{cfg}-s{seed}.json"
        train = json.loads(log.read_text()) if log.exists() else {}
        out.append(
            {
                "system": result["system"],
                "f1": s["f1"],
                "precision": s["precision"],
                "recall": s["recall"],
                "false_extraction": s["false_extraction"],
                "schema_valid": s["schema_valid"],
                "train_minutes": round(train.get("train_seconds", 0) / 60, 1),
                "adapter_mb": round(train.get("adapter_bytes", 0) / 1e6),
            }
        )
    return out


def main() -> None:
    print("system      F1    P     R     false-ext  schema-valid  train min  adapter MB")
    for r in sorted(rows(), key=lambda r: -r["f1"]):
        print(
            f"{r['system']:<11} {r['f1']:.3f} {r['precision']:.3f} {r['recall']:.3f} "
            f"{r['false_extraction']:.3f}      {r['schema_valid']:.2f}          "
            f"{r['train_minutes']:<9}  {r['adapter_mb']}"
        )


if __name__ == "__main__":
    main()
