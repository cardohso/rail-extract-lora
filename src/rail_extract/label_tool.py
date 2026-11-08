"""Local labelling page. Standard library only, bound to 127.0.0.1.

    uv run python -m rail_extract.label_tool --split val
    uv run python -m rail_extract.label_tool --split test     # your own session, never an LLM's

Shows one passage at a time, with a draft if one exists, checks your label against the rules and
logs the time spent. Labels are written to data/labels/<split>.jsonl on every save.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pydantic import ValidationError

from rail_extract import labels as L
from rail_extract.schema import Extraction

PAGE = Path(__file__).with_name("label_tool.html")


class State:
    def __init__(self, split: str, root: Path = L.ROOT) -> None:
        if split not in L.SPLITS:
            raise ValueError(f"split must be one of {L.SPLITS}")
        self.split, self.root = split, root
        self.passages = {r["id"]: r for r in L.read_jsonl(root / "data/passages.jsonl")}
        self.manifest = L.read_jsonl(root / "data/splits/manifest.jsonl")
        self.drafts = L.read_drafts(split, root)

    def labels(self) -> dict[str, L.Label]:
        return L.read_labels(self.split, self.root)

    def next(self, only: str | None = None) -> dict:
        labels = self.labels()
        pid = only or L.next_pending(self.manifest, self.split, labels)
        prog = L.progress(self.manifest, self.split, labels)
        if pid is None:
            return {"id": None, "progress": prog}
        p = self.passages[pid]
        existing = labels.get(pid)
        draft = self.drafts.get(pid)
        return {
            "id": pid,
            "heading": p["heading"],
            "section": p["section"],
            "text": p["text"],
            "existing": existing.model_dump() if existing else None,
            "draft": draft,
            "progress": prog,
        }

    def check(self, body: dict) -> dict:
        try:
            ext = Extraction.model_validate({"requirements": body["requirements"]})
        except ValidationError as exc:
            msgs = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
            return {"valid": False, "errors": msgs, "warnings": []}
        warnings = L.check_requirements(self.passages[body["id"]], ext.requirements)
        return {"valid": True, "errors": [], "warnings": warnings}

    def save(self, body: dict) -> dict:
        if body["id"] not in self.passages:
            raise KeyError("unknown passage")
        seconds = max(0.0, min(float(body.get("seconds", 0)), L.MAX_SECONDS_PER_PASSAGE))
        label = L.Label(
            id=body["id"],
            split=self.split,
            requirements=[] if body.get("skipped") else body["requirements"],
            unsure=bool(body.get("unsure")),
            skipped=bool(body.get("skipped")),
            skip_reason=body.get("skip_reason"),
            seconds=seconds,
        )
        labels = self.labels()
        if body["id"] in labels:  # a re-edit adds its time to the earlier time
            label.seconds += labels[body["id"]].seconds
        labels[label.id] = label
        L.write_labels(self.split, labels, self.root)
        return {"ok": True, "progress": L.progress(self.manifest, self.split, labels)}


def make_handler(state: State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, code: int, obj: object) -> None:
            self._send(code, json.dumps(obj).encode())

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path.startswith("/api/next"):
                only = self.path.split("id=", 1)[1] if "id=" in self.path else None
                self._json(200, state.next(only))
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if self.path == "/api/check":
                    self._json(200, state.check(body))
                elif self.path == "/api/save":
                    self._json(200, state.save(body))
                else:
                    self._json(404, {"error": "not found"})
            except (ValidationError, KeyError, ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, *args: object) -> None:  # keep the terminal quiet
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=L.SPLITS)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    state = State(args.split)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    print(f"labelling '{args.split}' at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
