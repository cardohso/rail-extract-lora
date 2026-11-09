# rail-extract

[![ci](https://github.com/cardohso/rail-extract-lora/actions/workflows/ci.yml/badge.svg)](https://github.com/cardohso/rail-extract-lora/actions/workflows/ci.yml)

Is a LoRA fine-tuned small model better than prompting at pulling measurable requirements
(parameter, comparator, value, unit, conditions) out of EU railway TSI passages, and is it
worth the effort? One domain, one reviewer, small data.

**Status: in progress. No results yet.** The plan is frozen in
[docs/PREREGISTRATION.md](docs/PREREGISTRATION.md). A 30-second summary and reproduction
steps will be added with the report.

## Development

```bash
uv sync --frozen
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

Tests run offline with models mocked; no GPU is needed. GPU runs use the `train` dependency
group on a free Kaggle GPU ([docs/KAGGLE.md](docs/KAGGLE.md)). The training loop's own tests need
torch: `uv run --frozen --group train pytest tests/test_train_loop.py`.

## Run a system and score it

```bash
uv run python -m rail_extract.predict --system R0 --split val      # regex baseline, no model
uv run python -m rail_extract.evaluate --system R0 --split val     # reads saved predictions only
```

Model systems (`--backend hf` on a GPU, `--backend gemini|groq` with a key in the environment)
use the same two commands. `test` and `heldout` are refused without `--final`.
Predictions are committed under `predictions/`, so every reported number can be re-scored with
`evaluate` alone.

## Data and licences

Code: MIT. Source texts: see [data/SOURCES.md](data/SOURCES.md). The base model is
`ibm-granite/granite-4.2-3b` (Apache 2.0). Weights are never committed.
