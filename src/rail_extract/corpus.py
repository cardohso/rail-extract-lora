"""Build data/passages.jsonl: the nine TSI/Directive texts, parsed and chunked.

    uv run python -m rail_extract.corpus            # download missing raw files, then build
    uv run python -m rail_extract.corpus --offline  # use data/raw only

Raw downloads go to data/raw (gitignored). Passages are committed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from rail_extract.chunking import chunk_section
from rail_extract.ingest import USER_AGENT, Document, download, parse_xhtml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "passages.jsonl"

DOCUMENTS = [
    Document(
        "iop-directive",
        "Interoperability Directive",
        "02016L0797-20260802",
        "Directive (EU) 2016/797",
    ),
    Document("tsi-ccs", "CCS TSI", "02023R1695-20260505", "Implementing Regulation (EU) 2023/1695"),
    Document("tsi-prm", "PRM TSI", "02014R1300-20230928", "Regulation (EU) No 1300/2014"),
    Document("tsi-ene", "ENE TSI", "02014R1301-20230928", "Regulation (EU) No 1301/2014"),
    Document("tsi-srt", "SRT TSI", "02014R1303-20240129", "Regulation (EU) No 1303/2014"),
    Document("tsi-noi", "NOI TSI", "02014R1304-20230928", "Regulation (EU) No 1304/2014"),
    Document("tsi-ope", "OPE TSI", "02019R0773-20250331", "Implementing Regulation (EU) 2019/773"),
    Document("tsi-inf", "INF TSI", "02014R1299-20230928", "Regulation (EU) No 1299/2014"),
    Document("tsi-loc-pas", "LOC&PAS TSI", "02014R1302-20250427", "Regulation (EU) No 1302/2014"),
]


def build(offline: bool = False) -> list[dict]:
    passages = []
    with httpx.Client(follow_redirects=True, timeout=180, headers={"User-Agent": USER_AGENT}) as c:
        for doc in DOCUMENTS:
            raw = RAW / f"{doc.celex}.xhtml"
            if not offline:
                raw = download(doc, RAW, c)
            sections = parse_xhtml(raw.read_text(encoding="utf-8"), doc.id, doc.celex)
            for s in sections:
                passages += [p.to_dict() for p in chunk_section(s.to_record(), doc.short)]
    return passages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    passages = build(args.offline)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for p in passages:
        counts[p["doc_id"]] = counts.get(p["doc_id"], 0) + 1
    print(len(passages), "passages", counts)


if __name__ == "__main__":
    main()
