"""Split parsed sections into passages small enough to embed, keeping citation keys."""

from __future__ import annotations

from dataclasses import asdict, dataclass

MAX_WORDS = 250  # ~330 tokens, below the 512-token limit of the embedding model
OVERLAP = 40


@dataclass(frozen=True)
class Passage:
    id: str  # "<doc_id>:<section>#<n>", the citation key used everywhere downstream
    doc_id: str
    doc_short: str
    section: str
    heading: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc_short}, {self.section}"

    @property
    def embed_text(self) -> str:
        return f"{self.doc_short}, {self.heading}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Passage:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def _windows(words: list[str], max_words: int, overlap: int) -> list[str]:
    step = max_words - overlap
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words) - overlap, step)]


def chunk_section(
    record: dict, doc_short: str, max_words: int = MAX_WORDS, overlap: int = OVERLAP
) -> list[Passage]:
    pieces: list[str] = []
    for para in record["text"].split("\n"):
        words = para.split()
        if len(words) > max_words:
            pieces += _windows(words, max_words, overlap)
        else:
            pieces.append(para)

    # Greedily pack whole paragraphs into passages of at most max_words.
    chunks: list[list[str]] = []
    size = 0
    for piece in pieces:
        n = len(piece.split())
        if chunks and size + n <= max_words:
            chunks[-1].append(piece)
            size += n
        else:
            chunks.append([piece])
            size = n

    base = f"{record['doc_id']}:{record['section']}"
    return [
        Passage(
            id=f"{base}#{i}",
            doc_id=record["doc_id"],
            doc_short=doc_short,
            section=record["section"],
            heading=record["heading"],
            text="\n".join(chunk),
        )
        for i, chunk in enumerate(chunks, start=1)
    ]
