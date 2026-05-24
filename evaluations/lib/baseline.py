"""Baseline score storage and regression checks for evaluation runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

LIB_DIR = Path(__file__).resolve().parent
BASELINE_PATH = LIB_DIR / "baseline-scores.json"
DEFAULT_REGRESSION_THRESHOLD = 0.10


def regression_threshold() -> float:
    raw = os.getenv("EVAL_REGRESSION_THRESHOLD", str(DEFAULT_REGRESSION_THRESHOLD))
    return float(raw)


def should_update_baseline() -> bool:
    return os.getenv("EVAL_UPDATE_BASELINE", "").strip().lower() in ("1", "true", "yes")


def load_baseline(path: Path | None = None) -> dict[str, Any]:
    target = path or BASELINE_PATH
    if not target.is_file():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def save_baseline(scores: dict[str, Any], path: Path | None = None) -> None:
    target = path or BASELINE_PATH
    target.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_regression(
    current: dict[str, float],
    baseline: dict[str, Any],
    *,
    threshold: float | None = None,
) -> list[str]:
    """Return human-readable failures when a metric drops more than threshold vs baseline."""
    limit = threshold if threshold is not None else regression_threshold()
    baseline_metrics = baseline.get("metrics", baseline)
    failures: list[str] = []
    for name, base_val in baseline_metrics.items():
        if not isinstance(base_val, (int, float)):
            continue
        cur_val = current.get(name)
        if cur_val is None or not isinstance(cur_val, (int, float)):
            continue
        if base_val <= 0:
            continue
        drop = (base_val - cur_val) / base_val
        if drop > limit:
            failures.append(
                f"{name}: {cur_val:.3f} vs baseline {base_val:.3f} "
                f"(drop {drop * 100:.1f}% > {limit * 100:.0f}%)"
            )
    return failures
