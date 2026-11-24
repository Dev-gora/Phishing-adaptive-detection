"""
evaluation.py
=============
Metric computation, timing instrumentation, feature-selection stability and
result persistence.

Every number written to ``results/metrics`` is produced by this module from an
actual model fit; nothing here is estimated, imputed or hand-entered.
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from models import positive_class_proba


# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------
class Timer:
    """Context manager returning wall-clock seconds in ``.elapsed``."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self.elapsed = time.perf_counter() - self._start


# --------------------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------------------
def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Standard binary metrics for the phishing (positive) class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_score)))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def evaluate_estimator(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    fit: bool = False,
    X_fit: Optional[pd.DataFrame] = None,
    y_fit: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Optionally fit, then time prediction and compute metrics on (X, y)."""
    result: Dict[str, float] = {}

    if fit:
        if X_fit is None or y_fit is None:
            raise ValueError("X_fit/y_fit are required when fit=True")
        with Timer() as timer:
            estimator.fit(X_fit, y_fit)
        result["train_time_s"] = timer.elapsed

    with Timer() as timer:
        y_pred = estimator.predict(X)
        y_score = positive_class_proba(estimator, X)
    result["inference_time_s"] = timer.elapsed
    result["inference_time_per_sample_ms"] = 1000.0 * timer.elapsed / max(len(X), 1)

    result.update(compute_metrics(y, y_pred, y_score))
    result["n_eval_samples"] = int(len(X))
    return result


def roc_points(estimator, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, float]:
    y_score = positive_class_proba(estimator, X)
    fpr, tpr, _ = roc_curve(y, y_score)
    auc = float(roc_auc_score(y, y_score))
    return fpr, tpr, auc


# --------------------------------------------------------------------------------------
# Feature-selection stability
# --------------------------------------------------------------------------------------
def jaccard(set_a: Iterable[str], set_b: Iterable[str]) -> float:
    """Jaccard similarity |A n B| / |A u B|; 1.0 for two empty sets."""
    a, b = set(set_a), set(set_b)
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def stability_metrics(rankings: List[List[str]], k: int) -> Dict[str, float]:
    """Mean pairwise Jaccard similarity of the top-k sets across repeated runs.

    ``rankings`` is a list of ordered feature lists, one per run (bootstrap
    resample or CV fold). Values near 1.0 mean the selector returns essentially
    the same features regardless of the particular training sample.
    """
    top_k_sets = [list(ranking)[:k] for ranking in rankings]
    pairs = list(combinations(range(len(top_k_sets)), 2))
    if not pairs:
        return {"k": k, "mean_jaccard": float("nan"), "std_jaccard": float("nan"), "n_pairs": 0}

    scores = [jaccard(top_k_sets[i], top_k_sets[j]) for i, j in pairs]
    return {
        "k": k,
        "mean_jaccard": float(np.mean(scores)),
        "std_jaccard": float(np.std(scores)),
        "min_jaccard": float(np.min(scores)),
        "n_pairs": len(scores),
    }


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def save_json(payload, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def format_metrics_table(df: pd.DataFrame, columns: Optional[List[str]] = None) -> str:
    """Compact console rendering used in the stage reports."""
    if columns is None:
        columns = [
            c for c in ("model", "method", "n_features", "accuracy", "precision",
                        "recall", "f1", "roc_auc")
            if c in df.columns
        ]
    if df.empty:
        return "(empty)"
    view = df[columns].copy()
    for column in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if column in view.columns:
            view[column] = view[column].map(lambda v: f"{v:.4f}" if pd.notna(v) else "  n/a ")
    return view.to_string(index=False)
