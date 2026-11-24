"""
config.py
=========
Central, serialisable experiment configuration for the study:

    "Lightweight and Explainable Phishing Detection Using Adaptive Feature Selection"

Every knob that can change an experimental result lives here so that a run is
fully reproducible: one ``ExperimentConfig`` + one seed + one dataset checksum
determines every number reported in the paper.

Design notes
------------
* All randomness is driven by ``cfg.seed`` (numpy, random, sklearn, xgboost/lgbm).
* Paths are resolved relative to the project root, so the pipeline runs on a
  Windows laptop and on Google Colab without edits.
* ``to_dict()`` is written next to every results table, so an artefact can never
  be detached from the configuration that produced it.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
RANKINGS_DIR: Path = RESULTS_DIR / "feature_rankings"
FIG_DIR: Path = RESULTS_DIR / "visualizations"
LOG_DIR: Path = RESULTS_DIR / "logs"

RAW_DATASET: Path = DATA_DIR / "dataset_phishing_raw.csv"

for _directory in (DATA_DIR, METRICS_DIR, RANKINGS_DIR, FIG_DIR, LOG_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Complete experimental configuration.

    Defaults are chosen for a standard Windows laptop / free Colab CPU runtime
    (target: full pipeline under ~40 min). Nothing here is tuned on the test set.
    """

    # ---- Reproducibility -------------------------------------------------------
    seed: int = 42

    # ---- Splitting -------------------------------------------------------------
    # Stratified 60 / 20 / 20 -> train (fit), validation (selection), test (report).
    test_size: float = 0.20
    val_size: float = 0.20          # fraction of the *full* dataset
    stratify: bool = True

    # ---- Preprocessing ---------------------------------------------------------
    drop_constant_features: bool = True
    drop_duplicate_urls: bool = True
    correlation_threshold: float = 0.95   # |r| above this is *reported*, not dropped

    # ---- Feature selection -----------------------------------------------------
    # Number of bootstrap-style resamples used for stability analysis.
    stability_repeats: int = 5
    stability_fraction: float = 0.80      # subsample fraction per repeat
    stability_k: int = 10                 # headline k for the stability metric

    # LIME
    lime_num_samples: int = 500           # instances explained to build global score
    lime_num_features: int = 87           # features per local surrogate

    # SHAP
    shap_max_samples: int = 1000          # background sample for TreeExplainer

    # Permutation importance
    perm_n_repeats: int = 5
    perm_cv_folds: int = 3

    # Mutual information
    mi_n_neighbors: int = 3

    # RFE: step == 1 gives a complete ranking; larger steps are much faster.
    rfe_step: int = 1

    # ---- Adaptive feature selection (main contribution) ------------------------
    # Performance tolerances tested for F1(k) >= F1(full) - epsilon
    epsilons: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])
    # Coarse grid of subset sizes. A dense 1..n grid is used for refinement.
    k_step: int = 2
    # Number of *consecutive* subset sizes that must satisfy the tolerance before a
    # size is accepted. Guards against selecting a small k because of validation noise.
    patience: int = 3
    # Final refinement around the coarse solution, with step 1.
    refine: bool = True

    # ---- Models ----------------------------------------------------------------
    model_names: List[str] = field(default_factory=lambda: ["RandomForest", "XGBoost", "KNN"])
    include_lightgbm: bool = True
    n_jobs: int = -1
    tuning_cv_folds: int = 3
    tune_hyperparameters: bool = True

    # ---- Output ----------------------------------------------------------------
    fast_mode: bool = False
    save_pdf: bool = True
    figure_dpi: int = 300
    verbose: bool = True

    # ------------------------------------------------------------------ helpers
    def __post_init__(self) -> None:
        if self.include_lightgbm and "LightGBM" not in self.model_names:
            self.model_names.append("LightGBM")
        if self.fast_mode:
            self._apply_fast_mode()

    def _apply_fast_mode(self) -> None:
        """Drastic reductions used only for smoke-testing the pipeline end to end.

        Results produced in fast mode are NOT reportable and are written with a
        ``_fast`` suffix by ``main.py``.
        """
        self.stability_repeats = 2
        self.lime_num_samples = 60
        self.shap_max_samples = 200
        self.perm_n_repeats = 2
        self.perm_cv_folds = 2
        self.k_step = 10
        self.patience = 2
        self.tuning_cv_folds = 2
        self.tune_hyperparameters = False
        self.include_lightgbm = False
        self.model_names = [m for m in self.model_names if m != "LightGBM"]

    @property
    def suffix(self) -> str:
        """File-name suffix: empty for reportable runs, ``_fast`` for smoke tests."""
        return "_fast" if self.fast_mode else ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | None = None) -> Path:
        path = path or (RESULTS_DIR / f"experiment_config{self.suffix}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path


# --------------------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Seed every library that participates in the experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy is a hard dependency
        pass


def describe_environment() -> Dict[str, str]:
    """Record library versions so results can be reproduced exactly."""
    import platform
    import sys

    info: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for module_name in (
        "numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm",
        "shap", "lime", "matplotlib", "seaborn",
    ):
        try:
            module = __import__(module_name)
            info[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            info[module_name] = "NOT INSTALLED"
    return info
