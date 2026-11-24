"""
adaptive_selection.py
=====================
** Main research contribution: Adaptive Feature Selection (AFS). **

Problem
-------
Feature-selection papers typically report performance at one arbitrary ``k``
(``k = 10``, ``k = 15``, ...). That choice is neither justified nor dataset
adaptive: it may discard accuracy, or keep features that buy nothing.

Proposed rule
-------------
For a ranked feature list and a threshold ``epsilon``, the adaptive selector
returns the **smallest** ``k`` such that

        F1(k)  >=  F1(full)  -  epsilon

evaluated on the *validation* split, with the full-feature model as the
baseline. ``epsilon`` is a stated, configurable performance tolerance
(0.01 / 0.02 / 0.05 by default), so the user chooses the speed-accuracy
trade-off explicitly instead of assuming that fewer features are always better.

Two safeguards against validation noise
---------------------------------------
1. **Patience.** The criterion must hold for ``patience`` *consecutive* subset
   sizes, not a single lucky ``k``. A lone spurious spike in F1 at k=3 does not
   justify a 3-feature model.
2. **Explicit fallback.** If no ``k`` satisfies the criterion, the full set is
   retained and ``fallback_used=True`` is recorded. The selector never silently
   returns a worse model, and never assumes fewer features must win.

The test split is not referenced anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------
@dataclass
class AdaptiveSelectionResult:
    """Outcome of one adaptive selection run."""

    method: str
    model: str
    epsilon: float
    f1_full: float
    n_features_full: int
    selected_k: int
    selected_features: List[str]
    f1_at_selected: float
    meets_criterion: bool
    fallback_used: bool
    raw_smallest_k: Optional[int]          # without patience (transparency)
    patience: int
    k_grid: List[int] = field(default_factory=list)

    @property
    def reduction_pct(self) -> float:
        if self.n_features_full == 0:
            return float("nan")
        return 100.0 * (1.0 - self.selected_k / self.n_features_full)

    @property
    def f1_drop(self) -> float:
        return self.f1_full - self.f1_at_selected

    def as_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "model": self.model,
            "epsilon": self.epsilon,
            "n_features_full": self.n_features_full,
            "selected_k": self.selected_k,
            "reduction_pct": round(self.reduction_pct, 2),
            "f1_full": round(self.f1_full, 6),
            "f1_selected": round(self.f1_at_selected, 6),
            "f1_drop": round(self.f1_drop, 6),
            "meets_criterion": self.meets_criterion,
            "fallback_used": self.fallback_used,
            "raw_smallest_k": self.raw_smallest_k,
            "patience": self.patience,
            "selected_features": ";".join(self.selected_features),
        }


# --------------------------------------------------------------------------------------
# Core algorithm
# --------------------------------------------------------------------------------------
def _consecutive_run_start(qualifies: Sequence[bool], patience: int) -> Optional[int]:
    """Index of the first element starting a run of ``patience`` True values."""
    run = 0
    for index, value in enumerate(qualifies):
        run = run + 1 if value else 0
        if run >= patience:
            return index - patience + 1
    return None


def adaptive_select(
    curve: pd.DataFrame,
    ranking: Sequence[str],
    f1_full: float,
    epsilon: float,
    method: str = "unknown",
    model: str = "unknown",
    patience: int = 3,
    n_features_full: Optional[int] = None,
) -> AdaptiveSelectionResult:
    """Apply the tolerance rule to a validation curve.

    Parameters
    ----------
    curve : DataFrame
        Columns ``n_features`` and ``f1`` -- out-of-sample (validation) F1 for
        each candidate subset size, for one (method, model) pair.
    ranking : sequence of str
        Full feature ranking; the top-k prefix defines each subset.
    f1_full : float
        Validation F1 of the full-feature baseline (same model).
    epsilon : float
        Acceptable F1 drop.
    patience : int
        Number of consecutive qualifying subset sizes required.

    Returns
    -------
    AdaptiveSelectionResult
    """
    n_features_full = int(n_features_full if n_features_full is not None else len(ranking))
    curve = curve.sort_values("n_features").reset_index(drop=True)

    if curve.empty:
        raise ValueError("Empty validation curve; cannot perform adaptive selection")

    threshold = f1_full - epsilon
    qualities = (curve["f1"] >= threshold).to_numpy()
    k_values = curve["n_features"].astype(int).tolist()

    # Smallest qualifying k with no smoothing requirement (reported for transparency).
    raw_smallest = None
    raw_qualifying = [i for i, ok in enumerate(qualities) if ok]
    if raw_qualifying:
        raw_smallest = int(k_values[raw_qualifying[0]])

    patience = max(1, int(patience))
    start = _consecutive_run_start(qualities, patience)

    if start is None:
        # No subset satisfies the tolerance -> keep everything. Never assume that
        # fewer features must be better.
        return AdaptiveSelectionResult(
            method=method,
            model=model,
            epsilon=epsilon,
            f1_full=float(f1_full),
            n_features_full=n_features_full,
            selected_k=n_features_full,
            selected_features=list(ranking)[:n_features_full],
            f1_at_selected=float(f1_full),
            meets_criterion=False,
            fallback_used=True,
            raw_smallest_k=raw_smallest,
            patience=patience,
            k_grid=k_values,
        )

    selected_k = int(k_values[start])
    f1_at_k = float(curve.loc[start, "f1"])

    return AdaptiveSelectionResult(
        method=method,
        model=model,
        epsilon=epsilon,
        f1_full=float(f1_full),
        n_features_full=n_features_full,
        selected_k=selected_k,
        selected_features=list(ranking)[:selected_k],
        f1_at_selected=f1_at_k,
        meets_criterion=True,
        fallback_used=False,
        raw_smallest_k=raw_smallest,
        patience=patience,
        k_grid=k_values,
    )


# --------------------------------------------------------------------------------------
# Curve construction
# --------------------------------------------------------------------------------------
def build_k_grid(n_features: int, step: int = 2) -> List[int]:
    """Coarse subset-size grid, always including 1 and the full set."""
    if n_features <= 0:
        return []
    if step <= 1:
        return list(range(1, n_features + 1))
    grid = list(range(step, n_features + 1, step))
    grid = [1, 2] + grid
    if n_features not in grid:
        grid.append(n_features)
    return sorted(set(int(k) for k in grid if 1 <= k <= n_features))


def refine_k_grid(selected_k: int, n_features: int, step: int = 2) -> List[int]:
    """Dense window around a coarse solution, used to pin down the smallest k.

    Guarantees the *true* smallest qualifying subset is not missed merely
    because of grid granularity.
    """
    low = max(1, selected_k - step)
    high = min(n_features, selected_k + step)
    return list(range(low, high + 1))


# --------------------------------------------------------------------------------------
# Aggregate reporting
# --------------------------------------------------------------------------------------
def summarize_across_epsilons(results: Sequence[AdaptiveSelectionResult]) -> pd.DataFrame:
    """One row per (method, model, epsilon) -- the paper's main results table."""
    return pd.DataFrame([r.as_dict() for r in results])
