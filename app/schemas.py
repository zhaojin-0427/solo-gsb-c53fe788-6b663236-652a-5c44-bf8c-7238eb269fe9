"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------

class VariableSpec(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    states: list[str] = Field(min_length=2, max_length=16)


class CPTRow(BaseModel):
    """One CPT row: parent assignment -> distribution over the variable."""

    when: dict[str, str]
    probs: dict[str, float]


class CPTSpec(BaseModel):
    parents: list[str] = Field(default_factory=list)
    rows: list[CPTRow] = Field(min_length=1, max_length=50000)


class ModelSpec(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    variables: list[VariableSpec] = Field(min_length=5, max_length=20)
    edges: list[tuple[str, str]] = Field(default_factory=list, max_length=380)
    cpts: dict[str, CPTSpec]


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

class InterventionSpec(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    do: dict[str, str] = Field(min_length=1)
    observe: Optional[list[str]] = None  # defaults to the posterior query set


class QuerySpec(BaseModel):
    # Variables whose posterior marginals are wanted. Empty means
    # "all non-evidence variables".
    posterior: list[str] = Field(default_factory=list, max_length=20)
    mpe: bool = False  # most probable explanation over non-evidence variables
    interventions: list[InterventionSpec] = Field(default_factory=list, max_length=8)


class DiagnosisRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=64)
    idempotency_key: str = Field(min_length=1, max_length=128)
    evidence: dict[str, str] = Field(default_factory=dict)
    queries: QuerySpec = Field(default_factory=QuerySpec)
