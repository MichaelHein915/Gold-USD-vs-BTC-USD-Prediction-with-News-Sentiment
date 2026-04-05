"""
Pick which hybrid (B_*) model to use for backtesting and trading runtime.

Keeps selection identical between step5_backtesting and btc_trading_runtime.
"""
from __future__ import annotations

from typing import Any


def pick_best_hybrid(results: dict[str, Any]) -> dict[str, Any]:
    """Select the best hybrid model by F1, then accuracy."""
    hybrid_keys = [k for k in results if k.startswith("B_")]
    if not hybrid_keys:
        raise ValueError("No hybrid (B_*) models found in results.")
    best_key = max(
        hybrid_keys,
        key=lambda k: (results[k]["f1"], results[k]["accuracy"]),
    )
    return results[best_key]
