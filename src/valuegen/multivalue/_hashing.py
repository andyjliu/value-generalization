"""Canonical JSON, content hashes and hash orderings.

The salts below are this package's own, so its orderings are independent of
any other hash ordering in the repo.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

ROW_ORDER_SALT = "mv-row-v1"
QUOTA_ORDER_SALT = "mv-quota-v1"
MIX_ORDER_SALT = "mv-mix-v1"
SAMPLE_SALT = "mv-sample-v1"


def canonical_json(obj) -> str:
    """UTF-8 JSON with sorted keys, compact separators, non-ASCII kept."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(obj, n: int = 12) -> str:
    return sha256_json(obj)[:n]


def order_key(salt: str, *parts) -> str:
    """The ordering hash ``SHA256(canonical_json([salt, *parts]))``."""
    return sha256_json([salt, *parts])


def row_order_key(seed: int, value: str, row_identity: str) -> str:
    return order_key(ROW_ORDER_SALT, seed, value, row_identity)


def quota_order_key(seed: int, arm_id: str, value: str) -> str:
    return order_key(QUOTA_ORDER_SALT, seed, arm_id, value)


def mix_order_key(seed: int, arm_id: str, row_identity: str) -> str:
    return order_key(MIX_ORDER_SALT, seed, arm_id, row_identity)


def balanced_quota(values: Sequence[str], n_total: int, seed: int, arm_id: str) -> dict[str, int]:
    """``floor(n/k)`` rows per value plus one for ``n mod k`` values, the
    extra going to the values that sort first under the quota hash."""
    values = list(values)
    if len(set(values)) != len(values):
        raise ValueError("duplicate values in quota request")
    if not values:
        raise ValueError("no values to allocate to")
    base, rem = divmod(n_total, len(values))
    order = sorted(values, key=lambda v: (quota_order_key(seed, arm_id, v), v))
    extra = set(order[:rem])
    quota = {v: base + (1 if v in extra else 0) for v in values}
    assert sum(quota.values()) == n_total
    return quota
