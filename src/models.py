"""
models.py
=========
Classifier definitions, leakage-safe preprocessing pipelines and
hyper-parameter tuning.

Experimental-condition rules
----------------------------
* Hyper-parameters are tuned **once**, on the training split only, with
  cross-validation. The resulting configuration is then reused for *every*
  feature subset so that subset comparison isolates the effect of the feature set
  (the alternative -- re-tuning for each of ~90 subsets -- would mix two effects
  and cost ~90x more compute). This is stated as a limitation in the README.
* Scaling is part of each model's own pipeline and is therefore fitted on the
  training fold only. Tree models do not need scaling; KNN does.
* Nothing in this module ever sees the test split.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ExperimentConfig, set_global_seed

# --------------------------------------------------------------------------------------
# Availability flags
# --------------------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # pragma: no cover
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover
    HAS_LIGHTGBM = False


MODELS_REQUIRING_SCALING = {"KNN"}


# --------------------------------------------------------------------------------------
# Default hyper-parameters
# --------------------------------------------------------------------------------------
def default_params(name: str, cfg: ExperimentConfig) -> Dict[str, Any]:
    if name == "RandomForest":
        return {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": cfg.seed,
            "n_jobs": cfg.n_jobs,
        }
    if name == "XGBoost":
        return {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "random_state": cfg.seed,
            "n_jobs": cfg.n_jobs,
        }
    if name == "LightGBM":
        return {
            "n_estimators": 300,
            "num_leaves": 31,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": cfg.seed,
            "n_jobs": cfg.n_jobs,
            "verbose": -1,
        }
    if name == "KNN":
        return {
            "n_neighbors": 5,
            "weights": "distance",
            "metric": "minkowski",
            "n_jobs": cfg.n_jobs,
        }
    raise ValueError(f"Unknown model: {name}")


def build_estimator(name: str, cfg: ExperimentConfig, params: Optional[Dict[str, Any]] = None):
    """Return an unfitted estimator (or a scaling pipeline for KNN)."""
    set_global_seed(cfg.seed)
    merged = default_params(name, cfg)
    if params:
        merged.update(params)

    if name == "RandomForest":
        return RandomForestClassifier(**merged)
    if name == "XGBoost":
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed; install it or set include_lightgbm=False")
        return XGBClassifier(**merged)
    if name == "LightGBM":
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is not installed")
        return LGBMClassifier(**merged)
    if name == "KNN":
        # Scaling is mandatory for a distance-based learner; it lives inside the
        # pipeline so it can only ever be fitted on training data.
        return Pipeline(
            [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(**merged))]
        )
    raise ValueError(f"Unknown model: {name}")


def supports_tree_explanations(name: str) -> bool:
    """TreeExplainer / Gini / permutation-on-tree are only valid for tree models."""
    return name in {"RandomForest", "XGBoost", "LightGBM"}


# --------------------------------------------------------------------------------------
# Hyper-parameter tuning (train split only)
# --------------------------------------------------------------------------------------
def tuning_grid(name: str) -> Dict[str, List[Any]]:
    if name == "RandomForest":
        return {"n_estimators": [200, 400], "max_depth": [None, 20]}
    if name == "XGBoost":
        return {"n_estimators": [200, 400], "max_depth": [4, 6]}
    if name == "LightGBM":
        return {"n_estimators": [200, 400], "num_leaves": [31, 63]}
    if name == "KNN":
        return {"knn__n_neighbors": [3, 5, 9, 15]}
    raise ValueError(f"Unknown model: {name}")


def tune_hyperparameters(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: ExperimentConfig,
) -> Tuple[Dict[str, Any], float]:
    """Grid-search on the training split with stratified CV (F1-optimised).

    Returns the best *estimator-level* parameters and the CV F1. For KNN the
    returned keys are stripped of the ``knn__`` pipeline prefix.
    """
    if not cfg.tune_hyperparameters:
        return {}, float("nan")

    estimator = build_estimator(name, cfg)
    cv = StratifiedKFold(n_splits=cfg.tuning_cv_folds, shuffle=True, random_state=cfg.seed)
    search = GridSearchCV(
        estimator,
        tuning_grid(name),
        scoring="f1",
        cv=cv,
        n_jobs=cfg.n_jobs,
        refit=False,
    )
    search.fit(X_train, y_train)

    best = dict(search.best_params_)
    if name == "KNN":
        best = {k.replace("knn__", ""): v for k, v in best.items()}
    return best, float(search.best_score_)


def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: ExperimentConfig,
) -> Dict[str, Dict[str, Any]]:
    """Tune every requested model on the training split and record the outcome."""
    tuned: Dict[str, Dict[str, Any]] = {}
    for name in cfg.model_names:
        if name == "XGBoost" and not HAS_XGBOOST:
            print("  [skip] XGBoost not available")
            continue
        if name == "LightGBM" and not HAS_LIGHTGBM:
            print("  [skip] LightGBM not available")
            continue
        params, cv_score = tune_hyperparameters(name, X_train, y_train, cfg)
        tuned[name] = {"params": params, "cv_f1": cv_score}
        print(f"  {name:14s} best params={params} cv_f1={cv_score:.4f}")
    return tuned


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def positive_class_proba(estimator, X: pd.DataFrame) -> np.ndarray:
    """Probability of the phishing class, robust to estimator flavour."""
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError("Estimator exposes neither predict_proba nor decision_function")


def describe_model(name: str, cfg: ExperimentConfig) -> Dict[str, Any]:
    """Short human-readable description used in result tables."""
    descriptions = {
        "RandomForest": "Random Forest (bagged decision trees, Gini impurity)",
        "XGBoost": "XGBoost (gradient-boosted trees, hist method)",
        "LightGBM": "LightGBM (leaf-wise gradient boosting)",
        "KNN": "k-Nearest Neighbours (StandardScaler + Euclidean distance)",
    }
    return {
        "model": name,
        "description": descriptions.get(name, name),
        "requires_scaling": name in MODELS_REQUIRING_SCALING,
        "seed": cfg.seed,
    }
