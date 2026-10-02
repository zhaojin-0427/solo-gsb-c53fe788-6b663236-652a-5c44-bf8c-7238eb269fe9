"""Shared helpers: canonical JSON, hashing, deterministic float rounding."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Serialize to a deterministic JSON byte representation.

    Sorted keys and tight separators make the output stable across
    processes, which is what the idempotent-replay guarantee is built on.
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def round12(value: float) -> float:
    """Round to 12 decimal places and normalize negative zero.

    Keeps serialized results deterministic and avoids ``-0.0`` leaking
    into the canonical JSON body.
    """
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0 else rounded
