"""Output schema from docs/PREREGISTRATION.md section 2. One definition feeds prompts,
constrained decoding and validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Comparator = Literal[">=", "<=", ">", "<", "==", "between"]


class Requirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameter: str = Field(min_length=1)
    comparator: Comparator
    value: float
    value_max: float | None
    unit: str | None
    conditions: list[str]
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def _value_max_only_for_between(self) -> Requirement:
        if (self.comparator == "between") != (self.value_max is not None):
            raise ValueError("value_max is set if and only if comparator is 'between'")
        return self


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirements: list[Requirement]


def json_schema() -> dict:
    """JSON Schema handed to constrained decoders and API structured output."""
    return Extraction.model_json_schema()


def flat_json_schema() -> dict:
    """`json_schema()` with `$ref`s inlined and titles dropped, for decoders and APIs that do not
    resolve references. Same constraints."""
    schema = json_schema()
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
            return {k: walk(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def parse_output(text: str) -> Extraction | None:
    """Schema-valid output, or None. The caller counts None as an empty prediction."""
    try:
        return Extraction.model_validate_json(text)
    except ValidationError:
        return None
