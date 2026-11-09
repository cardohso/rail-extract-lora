"""Text generators behind one interface: `generate(list of chat messages) -> list of raw strings`.

- MockBackend: tests and CI (no network, no GPU).
- HFBackend: local Hugging Face model, greedy, thinking off, optional XGrammar schema constraint.
  Needs the `train` dependency group and a GPU; imports are lazy so CI never loads torch.
- GeminiBackend / GroqBackend: free-tier APIs with schema-enforced output, temperature 0.
  Keys come from the environment (GEMINI_API_KEY, GROQ_API_KEY) and are never printed or stored.
  Responses are cached on disk so a re-run costs no requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from rail_extract.schema import flat_json_schema

Messages = list[dict]
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "api"


class Backend(Protocol):
    name: str

    def generate(self, batch: list[Messages]) -> list[str]: ...


class MockBackend:
    """Returns whatever `fn(messages)` returns. Used by tests."""

    name = "mock"

    def __init__(self, fn: Callable[[Messages], str]) -> None:
        self.fn = fn

    def generate(self, batch: list[Messages]) -> list[str]:
        return [self.fn(m) for m in batch]


class HFBackend:
    """Local model. Greedy decoding; `thinking` is passed to the chat template (off by default)."""

    def __init__(
        self,
        model: str,
        revision: str | None,
        constrained: bool = False,
        thinking: bool = False,
        max_new_tokens: int = 2048,
        batch_size: int = 8,
        device: str | None = None,
        adapter: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(model, revision=revision)
        net = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.bfloat16)
        if adapter:
            from peft import PeftModel

            net = PeftModel.from_pretrained(net, adapter)
        name = f"hf:{model}@{(revision or 'local')[:8]}" + (f"+{adapter}" if adapter else "")
        self._setup(net.to(device), tok, name, constrained, thinking, max_new_tokens, batch_size)

    @classmethod
    def from_loaded(cls, net, tok, name: str = "hf:loaded", **options) -> HFBackend:
        """Wrap a model already in memory (the overfit test generates right after training)."""
        self = cls.__new__(cls)
        defaults = {
            "constrained": False,
            "thinking": False,
            "max_new_tokens": 2048,
            "batch_size": 8,
        }
        self._setup(net, tok, name, **(defaults | options))
        return self

    def _setup(self, net, tok, name, constrained, thinking, max_new_tokens, batch_size) -> None:
        import torch

        self.torch, self.model, self.tok, self.name = torch, net.eval(), tok, name
        self.device = next(net.parameters()).device
        self.constrained, self.thinking = constrained, thinking
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.compiled = None
        if constrained:
            import xgrammar as xg

            vocab = net.config.vocab_size
            info = xg.TokenizerInfo.from_huggingface(tok, vocab_size=vocab)
            self.compiled = xg.GrammarCompiler(info).compile_json_schema(
                json.dumps(flat_json_schema())
            )

    def generate(self, batch: list[Messages]) -> list[str]:
        prompts = [
            self.tok.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking
            )
            for m in batch
        ]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths
        out: dict[int, str] = {}
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self.tok(
                [prompts[i] for i in idx],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,  # the chat template already wrote them
            ).to(self.device)
            procs = None
            if self.compiled is not None:
                from xgrammar.contrib.hf import LogitsProcessor

                procs = [LogitsProcessor(self.compiled)]
            with self.torch.no_grad():
                new = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    logits_processor=procs,
                )[:, enc.input_ids.shape[1] :]
            for i, text in zip(
                idx, self.tok.batch_decode(new, skip_special_tokens=True), strict=True
            ):
                out[i] = text
        return [out[i] for i in range(len(batch))]


class _HttpBackend:
    name = "http"
    url = ""
    key_env = ""
    min_interval = 4.0  # seconds between requests; stays under typical free-tier per-minute caps
    retries = 6

    def __init__(self, model: str, client=None, sleep=time.sleep) -> None:
        import httpx

        self.model = model
        self.name = f"{type(self).__name__.removesuffix('Backend').lower()}:{model}"
        self.client = client or httpx.Client(timeout=120)
        self.sleep = sleep
        self._last = 0.0

    def _key(self) -> str:
        key = os.environ.get(self.key_env)
        if not key:
            raise SystemExit(f"set {self.key_env} in your environment (it is never stored)")
        return key

    def _request(self, messages: Messages) -> tuple[dict, dict]:
        raise NotImplementedError

    def _text(self, data: dict) -> str:
        raise NotImplementedError

    def _one(self, messages: Messages) -> str:
        headers, body = self._request(messages)
        cache = CACHE_DIR / (
            hashlib.sha256(json.dumps([self.url, body]).encode()).hexdigest() + ".txt"
        )
        if cache.exists():
            return cache.read_text(encoding="utf-8")
        for attempt in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            r = self.client.post(self.url, headers=headers, json=body)
            if r.status_code in (429, 500, 502, 503, 504):
                self.sleep(min(60.0, 5.0 * 2**attempt))
                continue
            r.raise_for_status()
            text = self._text(r.json())
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
            return text
        raise RuntimeError(f"{self.name}: still rate limited after {self.retries} attempts")

    def generate(self, batch: list[Messages]) -> list[str]:
        return [self._one(m) for m in batch]


class GeminiBackend(_HttpBackend):
    key_env = "GEMINI_API_KEY"

    @property
    def url(self) -> str:  # type: ignore[override]
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )

    def _request(self, messages: Messages) -> tuple[dict, dict]:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
            if m["role"] != "system"
        ]
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseJsonSchema": flat_json_schema(),
            },
        }
        return {"x-goog-api-key": self._key()}, body

    def _text(self, data: dict) -> str:
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            return ""  # blocked or empty reply: scored as invalid output


class GroqBackend(_HttpBackend):
    key_env = "GROQ_API_KEY"
    url = "https://api.groq.com/openai/v1/chat/completions"

    def _request(self, messages: Messages) -> tuple[dict, dict]:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "strict": True, "schema": flat_json_schema()},
            },
        }
        return {"Authorization": f"Bearer {self._key()}"}, body

    def _text(self, data: dict) -> str:
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            return ""
