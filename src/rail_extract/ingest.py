"""Download EUR-Lex consolidated texts from Cellar and parse them into citable sections.

Copied from rail-regulation-rag (same author, MIT) so this repo rebuilds its corpus on its own.
"""

from __future__ import annotations

import json
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

CELLAR_URL = "http://publications.europa.eu/resource/celex/{celex}"
USER_AGENT = "rail-extract-lora/0.1 (+https://github.com/cardohso/rail-extract-lora)"

# Paragraph classes that carry no regulatory content.
SKIP_CLASSES = re.compile(
    r"^(arrow|modref|footnote|title-toc|toc-.*|hd-toc-.*|title-doc-.*|title-fam-member.*"
    r"|title-division-.*)$"
)
# Inline consolidation markers such as "►M1", "►C2", "▼B" and the closing "◄".
MARKERS = re.compile(r"[►▼]\s*[A-Z]\d*|◄")
FOOTNOTE_REF = re.compile(r"\s*\(\s*\d+\s*\)\s*$")
SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*(?:\.[a-z])?)\.?\s")
DELETED_PLACEHOLDER = re.compile(r"^[—–\-\s]*$")  # "►M1 ————— ◄" marks a deleted provision
BLOCK_TAGS = {"p", "div", "table", "tr", "td", "th", "tbody", "thead"}


@dataclass(frozen=True)
class Document:
    id: str
    short: str
    celex: str
    title: str


@dataclass
class Section:
    doc_id: str
    celex: str
    section: str  # stable citation key, e.g. "Article 5" or "Annex §4.2.1"
    heading: str
    paragraphs: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)

    def to_record(self) -> dict:
        record = asdict(self)
        del record["paragraphs"]
        return record | {"text": self.text}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", MARKERS.sub("", text)).strip()


def _classes(tag: Tag) -> list[str]:
    return tag.get("class") or []


def _block_text(node: Tag) -> str:
    """Text of a block element: inline content is concatenated, child blocks are space-separated."""
    parts, inline = [], ""
    for child in node.children:
        if not isinstance(child, Tag):
            inline += str(child)
        elif child.name == "table":
            parts += [inline, _table_text(child)]
            inline = ""
        elif child.name in BLOCK_TAGS:
            parts += [inline, _block_text(child)]
            inline = ""
        elif child.name != "img":
            inline += child.get_text("")
    return _clean(" ".join([*parts, inline]))


def _table_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [_block_text(td) for td in tr.find_all(["td", "th"], recursive=False)]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


class _Parser:
    def __init__(self, doc_id: str, celex: str):
        self.doc_id, self.celex = doc_id, celex
        self.sections: list[Section] = []
        self.in_body = False
        self.container: str | None = None  # "Annex I", "Appendix A", ...
        self.current: Section | None = None

    def start(self, section: str, heading: str) -> None:
        self.current = Section(self.doc_id, self.celex, section, heading, [])
        self.sections.append(self.current)

    def add(self, text: str) -> None:
        if self.in_body and self.current is not None and not DELETED_PLACEHOLDER.match(text):
            self.current.paragraphs.append(text)

    def paragraph(self, p: Tag) -> None:
        cls = " ".join(_classes(p))
        text = _clean(p.get_text(""))
        if cls == "title-article-norm":
            self.in_body, self.container = True, None
            self.start(text, text)
        elif not self.in_body or SKIP_CLASSES.match(cls):
            return
        elif cls in ("stitle-article-norm", "title-annex-2") and self.current is not None:
            self.current.heading = f"{self.current.heading} — {text}"
        elif cls == "title-annex-1":
            label = FOOTNOTE_REF.sub("", text)
            first, _, rest = label.partition(" ")
            self.container = f"{first.capitalize()} {rest}".strip()
            self.start(self.container, self.container)
        elif cls.startswith("title-gr-seq-level-"):
            m = SECTION_NUMBER.match(text)
            prefix = f"{self.container} §" if self.container else "§"
            self.start(f"{prefix}{m.group(1)}" if m else f"{prefix}{text}", text)
        else:
            self.add(text)

    def walk(self, node: Tag) -> None:
        for child in node.children:
            if not isinstance(child, Tag):
                continue
            if child.name == "p":
                self.paragraph(child)
            elif child.name == "table":
                self.add(_table_text(child))
            elif child.name == "div" and {"norm", "grid-container"} & set(_classes(child)):
                # Numbered paragraphs and lettered lists; they never contain headings.
                self.add(_block_text(child))
            elif child.name != "img":
                self.walk(child)


def parse_xhtml(xhtml: str, doc_id: str, celex: str) -> list[Section]:
    # The lenient HTML parser copes with markup quirks better than a strict XML parser.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(xhtml, "lxml")
    parser = _Parser(doc_id, celex)
    parser.walk(soup.body or soup)
    sections = [s for s in parser.sections if s.paragraphs]
    # Citation keys must be unique within a document; repeated headings get a suffix.
    seen: dict[str, int] = {}
    for s in sections:
        seen[s.section] = seen.get(s.section, 0) + 1
        if seen[s.section] > 1:
            s.section = f"{s.section} ({seen[s.section]})"
    return sections


def download(doc: Document, raw_dir: Path, client: httpx.Client, retries: int = 3) -> Path:
    target = raw_dir / f"{doc.celex}.xhtml"
    if target.exists():
        return target
    headers = {"Accept": "application/xhtml+xml", "Accept-Language": "eng"}
    for attempt in range(retries):
        try:
            resp = client.get(CELLAR_URL.format(celex=doc.celex), headers=headers)
            resp.raise_for_status()
            break
        except httpx.HTTPError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt * 5)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target.write_bytes(resp.content)
    time.sleep(1)  # be polite to Cellar
    return target


def write_jsonl(sections: list[Section], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in sections:
            f.write(json.dumps(s.to_record(), ensure_ascii=False) + "\n")
