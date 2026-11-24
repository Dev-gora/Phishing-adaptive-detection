"""Analysis engine for the prototype API.

Design rules enforced here:

* Models are trained once and cached on disk. Analysing a URL never retrains.
* Dataset-level metrics are always computed on the held-out test split.
* Feature selection runs on the training split only, so no test information
  leaks into the subset that a model is fitted on.
* Only URL-derived features are served for live analysis, because the other
  dataset columns require fetching the page or a third-party service.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
CACHE_DIR = BACKEND_DIR / ".cache"
# Curated artefacts shipped with the deployment (project root, not next to the code).
BUNDLED_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"


def _writable_dir() -> Path:
    """Pick a writable cache directory.

    Serverless runtimes mount the deployment read-only, so on Vercel (or any
    Lambda-style host) artefacts are only ever written to the temp directory.
    """
    override = os.environ.get("PD_CACHE_DIR")
    if override:
        target = Path(override)
    elif os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        target = Path(tempfile.gettempdir()) / "pd-cache"
    else:
        target = CACHE_DIR
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        target = Path(tempfile.gettempdir()) / "pd-cache"
        target.mkdir(parents=True, exist_ok=True)
    return target


WRITE_DIR = _writable_dir()
MODEL_DIR = WRITE_DIR / "models"
# Where a fitted model may already exist, in priority order.
MODEL_SEARCH_DIRS = (MODEL_DIR, CACHE_DIR / "models", BUNDLED_DIR / "models")

# ``src`` is a package that uses relative imports, so the project root must be on
# the path and the modules must be imported as ``src.*``.
for path in (str(PROJECT_ROOT), str(SRC_DIR), str(BACKEND_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.config import ExperimentConfig  # noqa: E402
from src.evaluation import evaluate_estimator  # noqa: E402
from src.explainability import explain_instance_lime  # noqa: E402
from src.feature_selection import SelectionContext, run_all_selectors  # noqa: E402
from src.models import HAS_LIGHTGBM, HAS_XGBOOST, build_estimator  # noqa: E402
from src.preprocessing import clean_dataset, load_raw_dataset, make_splits, validate_dataset  # noqa: E402

import url_features  # noqa: E402

MODEL_ORDER = ["RandomForest", "XGBoost", "KNN", "LightGBM"]
MODEL_LABELS = {
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "KNN": "K-Nearest Neighbours",
    "LightGBM": "LightGBM",
}
MODEL_NOTES = {
    "RandomForest": "Bagged decision trees; supports exact tree SHAP and Gini importance.",
    "XGBoost": "Gradient-boosted trees; supports exact tree SHAP.",
    "KNN": "Distance-based learner on standardised features; no tree structure, so tree SHAP is unavailable.",
    "LightGBM": "Leaf-wise gradient boosting; supports exact tree SHAP.",
}

SELECTOR_METHODS = {
    "SHAP": "Mean absolute SHAP value from the shared Random Forest, i.e. the average magnitude of each feature's contribution across instances.",
    "LIME": "Aggregate absolute LIME weights over sampled training instances, i.e. how often and how strongly a feature drives local predictions.",
    "Permutation": "Drop in out-of-sample performance when a feature's values are shuffled, averaged over repeats.",
    "MutualInfo": "Univariate mutual information between each feature and the label.",
    "RFE": "Recursive feature elimination: repeatedly drop the least important features of a fitted Random Forest.",
    "RFImportance": "Impurity-based (Gini) importance of the shared Random Forest.",
}

METHOD_LABELS = {
    "Full": "Full Feature Set",
    "SHAP": "SHAP",
    "LIME": "LIME",
    "Permutation": "Permutation Importance",
    "MutualInfo": "Mutual Information",
    "RFE": "Recursive Feature Elimination",
    "RFImportance": "Random Forest Importance",
    "SLA-FS": "SLA-FS (SHAP + LIME)",
    "Adaptive": "Adaptive Feature Selection",
}

METHOD_ORDER = ["Full", "SHAP", "LIME", "Permutation", "MutualInfo", "RFE", "RFImportance", "SLA-FS", "Adaptive"]

DEFAULT_K = 12


def _load_tuned_params() -> Dict[str, dict]:
    """Tuned hyper-parameters, preferring the copy shipped with the deployment.

    ``results/`` is not part of the deployment bundle, so the artefacts directory
    carries the same file; without it a refitted model would silently fall back
    to the library defaults and no longer match the paper's configuration.
    """
    for path in (BUNDLED_DIR / "tuning.json", RESULTS_DIR / "tuning.json"):
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        params: Dict[str, dict] = {}
        for model, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            best = payload.get("best_params", payload.get("params", payload))
            if isinstance(best, dict):
                clean = {}
                for key, value in best.items():
                    clean[key.replace("knn__", "")] = value
                params[model] = clean
        if params:
            return params
    return {}


class Engine:
    """Owns the dataset, the trained models and the cached feature rankings."""

    def __init__(self) -> None:
        for directory in (WRITE_DIR, MODEL_DIR):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

        self.cfg = ExperimentConfig()
        self.cfg.include_lightgbm = HAS_LIGHTGBM
        self.cfg.n_jobs = -1

        # Bounded settings so the prototype stays interactive while keeping the
        # same selector implementations used in the research pipeline.
        self.cfg.shap_max_samples = 300
        self.cfg.lime_num_samples = 300
        self.cfg.perm_n_repeats = 2
        self.cfg.perm_cv_folds = 2
        self.cfg.rfe_step = 2

        started = time.perf_counter()
        self.raw = load_raw_dataset()
        self.audit = validate_dataset(self.raw)
        self.bundle = clean_dataset(self.raw, self.cfg)
        self.splits = make_splits(self.bundle, self.cfg)

        self.fidelity = url_features.fidelity_report(self.raw, sample_n=400, seed=self.cfg.seed)
        self.url_feature_names: List[str] = [
            f for f in url_features.feature_order() if f in self.fidelity
            and f in self.bundle.X.columns
            and self._is_reliable(self.fidelity[f])
        ]
        self.url_feature_names.sort(key=lambda f: list(self.bundle.X.columns).index(f))

        self.X_train_url = self.splits.X_train[self.url_feature_names]
        self.X_val_url = self.splits.X_val[self.url_feature_names]
        self.X_test_url = self.splits.X_test[self.url_feature_names]

        self.tuned = _load_tuned_params()
        self._models: Dict[str, object] = {}
        self._model_lock = threading.Lock()
        self._eval_cache: Dict[str, dict] = {}
        self._knn_cache: Dict[str, int] = {}

        self.rankings: Dict[str, dict] = {}
        self.rankings_ready = threading.Event()
        self.rankings_error: Optional[str] = None
        self.models_ready = threading.Event()
        self.ready_s = round(time.perf_counter() - started, 2)

        self._load_cached_rankings()
        if not self.rankings:
            # Only pay for the selectors when no cached ranking could be read.
            threading.Thread(target=self._build_rankings, daemon=True).start()
        threading.Thread(target=self._warm_models, daemon=True).start()

    def _warm_models(self) -> None:
        """Fit the full-feature models once so live analysis is instant."""
        try:
            for name in self.available_models():
                try:
                    self.url_model(name)
                except Exception:
                    continue
        finally:
            self.models_ready.set()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _is_reliable(stats: dict) -> bool:
        if stats["discrete"]:
            return stats["exact_match"] >= 0.95
        return stats["correlation"] >= 0.95 and stats["mae"] <= 2.5

    def _params_for(self, model: str) -> dict:
        params = dict(self.tuned.get(model, {}))
        if model == "KNN":
            # Joblib process startup dominates a one-row prediction, so KNN
            # predicts single-threaded. Fitting is cheap on this dataset.
            params["n_jobs"] = 1
        return params

    def available_models(self) -> List[str]:
        models = ["RandomForest", "KNN"]
        if HAS_XGBOOST:
            models.insert(1, "XGBoost")
        if HAS_LIGHTGBM:
            models.append("LightGBM")
        return models

    # ------------------------------------------------------------- model fitting
    def model(self, name: str, features: Sequence[str], cache_key: Optional[str] = None):
        """Fit (or load) a model on the given feature subset of the training split."""
        key = cache_key or f"{name}__{hashlib.sha1('|'.join(features).encode()).hexdigest()[:12]}"
        with self._model_lock:
            if key in self._models:
                return self._models[key]
        for directory in MODEL_SEARCH_DIRS:
            path = directory / f"{key}.joblib"
            if not path.exists():
                continue
            try:
                import joblib

                model = joblib.load(path)
            except Exception:
                # A pickle written by a different Python/scikit-learn build cannot
                # always be loaded; fall through and refit instead of failing.
                continue
            with self._model_lock:
                self._models[key] = model
            return model

        estimator = build_estimator(name, self.cfg, self._params_for(name))
        columns = list(features)
        estimator.fit(self.splits.X_train[columns], self.splits.y_train)
        try:
            import joblib

            joblib.dump(estimator, path)
        except Exception:
            pass
        with self._model_lock:
            self._models[key] = estimator
        return estimator

    def url_model(self, name: str):
        return self.model(name, self.url_feature_names, cache_key=f"urlfull_{name}")

    # ---------------------------------------------------------------- evaluation
    def evaluate(self, model_name: str, method: str, k: Optional[int] = None,
                 epsilon: float = 0.02) -> dict:
        """Dataset-level metrics for one (model, feature-selection) configuration."""
        cache_key = f"{model_name}|{method}|{k}|{epsilon}"
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]

        if method == "Adaptive":
            features, chosen_k = self.adaptive_subset(model_name, epsilon)
            k_used = chosen_k
        elif method == "Full":
            features = list(self.url_feature_names)
            k_used = len(features)
        else:
            ranking = self._ranking(method)
            k_used = int(k or DEFAULT_K)
            features = ranking["features"][: max(1, min(k_used, len(self.url_feature_names)))]
            k_used = len(features)

        model = self.model(model_name, features,
                           cache_key=f"{model_name}__{method}__{k_used}")
        metrics = evaluate_estimator(model, self.splits.X_test[features], self.splits.y_test)

        payload = {
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_features": int(len(features)),
            "features": list(features),
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "roc_auc": round(metrics["roc_auc"], 4) if not np.isnan(metrics["roc_auc"]) else None,
            "train_time_s": round(metrics.get("train_time_s", float("nan")), 3)
            if "train_time_s" in metrics else None,
            "predict_time_ms": round(metrics["inference_time_per_sample_ms"], 4),
            "predict_time_total_s": round(metrics["inference_time_s"], 4),
            "n_eval_samples": int(metrics["n_eval_samples"]),
            "confusion": {"tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"], "tp": metrics["tp"]},
        }
        self._eval_cache[cache_key] = payload
        return payload

    # ------------------------------------------------------- feature selection
    def _build_rankings(self) -> None:
        try:
            t0 = time.perf_counter()
            ctx = SelectionContext(
                X_train=self.X_train_url,
                y_train=self.splits.y_train,
                cfg=self.cfg,
                X_val=self.X_val_url,
                y_val=self.splits.y_val,
            )
            produced = run_all_selectors(ctx, methods=list(SELECTOR_METHODS), progress=False)
            payload = {}
            for name, ranking in produced.items():
                if ranking.n_features == 0:
                    continue
                payload[name] = {
                    "method": name,
                    "features": list(ranking.features),
                    "scores": {f: float(ranking.scores.get(f, 0.0)) for f in ranking.features},
                    "top3": list(ranking.features[:3]),
                    "elapsed_s": round(float(ranking.elapsed_s), 3),
                }

            shap_list = payload.get("SHAP", {}).get("features", [])
            lime_list = payload.get("LIME", {}).get("features", [])
            if shap_list and lime_list:
                fused_scores = {}
                for f in self.url_feature_names:
                    r_shap = shap_list.index(f) + 1 if f in shap_list else len(shap_list) + 1
                    r_lime = lime_list.index(f) + 1 if f in lime_list else len(lime_list) + 1
                    fused_scores[f] = -(r_shap + r_lime) / 2.0
                fused = sorted(fused_scores, key=lambda f: (-fused_scores[f], f))
                payload["SLA-FS"] = {
                    "method": "SLA-FS",
                    "features": fused,
                    "scores": fused_scores,
                    "top3": fused[:3],
                    "elapsed_s": 0.0,
                    "note": "Rank-average of the SHAP and LIME orderings (SLA-FS fusion rule).",
                }

            self.rankings = payload
            self.rankings_s = round(time.perf_counter() - t0, 2)
            self._save_rankings(payload)
        except Exception as exc:  # keep the API alive; UI reports the failure
            self.rankings_error = f"{type(exc).__name__}: {exc}"
        finally:
            self.rankings_ready.set()

    def _rankings_candidates(self) -> List[Path]:
        digest = hashlib.sha1("|".join(self.url_feature_names).encode()).hexdigest()[:12]
        return [
            BUNDLED_DIR / "rankings.json",
            CACHE_DIR / f"rankings_{digest}.json",
            WRITE_DIR / f"rankings_{digest}.json",
        ]

    def _load_cached_rankings(self) -> None:
        for path in self._rankings_candidates():
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("features") == self.url_feature_names and payload.get("methods"):
                self.rankings = payload["methods"]
                self.rankings_s = payload.get("elapsed_s", 0.0)
                self.rankings_ready.set()
                return

    def _save_rankings(self, methods: dict) -> None:
        payload = {
            "features": self.url_feature_names,
            "methods": methods,
            "elapsed_s": getattr(self, "rankings_s", 0.0),
        }
        digest = hashlib.sha1("|".join(self.url_feature_names).encode()).hexdigest()[:12]
        try:
            (WRITE_DIR / f"rankings_{digest}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def _ranking(self, method: str) -> dict:
        if not self.rankings_ready.is_set():
            self.rankings_ready.wait(timeout=180)
        if method in self.rankings:
            return self.rankings[method]
        raise RuntimeError(
            f"Feature-selection method '{method}' is not available."
            + (f" {self.rankings_error}" if self.rankings_error else "")
        )

    def ranking_table(self) -> dict:
        if not self.rankings_ready.is_set():
            self.rankings_ready.wait(timeout=180)
        return {
            "features": self.url_feature_names,
            "methods": {
                name: {
                    "label": METHOD_LABELS.get(name, name),
                    "top": payload["features"][:12],
                    "elapsed_s": payload.get("elapsed_s", 0.0),
                    "note": payload.get("note", SELECTOR_METHODS.get(name, "")),
                }
                for name, payload in self.rankings.items()
            },
            "error": self.rankings_error,
        }

    # -------------------------------------------------------- adaptive selection
    def adaptive_subset(self, model_name: str, epsilon: float) -> tuple[List[str], int]:
        """Smallest subset that stays within ``epsilon`` F1 of the full model.

        Evaluated on the validation split, exactly as the research pipeline does,
        with a patience rule so a noisy single point cannot win.
        """
        key = f"{model_name}|{epsilon}"
        if key in self._knn_cache:
            k = self._knn_cache[key]
            return self.url_feature_names[:k], k

        ranking = self._ranking("SHAP")
        ordered = [f for f in ranking["features"] if f in self.url_feature_names]
        n = len(ordered)
        grid = sorted({5, 8, 10, 12, 16, 20, 25, n})
        grid = [g for g in grid if g <= n]

        full_model = self.model(model_name, ordered, cache_key=f"adaptive_{model_name}_full")
        full_f1 = evaluate_estimator(full_model, self.splits.X_val[ordered], self.splits.y_val)["f1"]

        # Evaluate the size grid from smallest to largest and stop as soon as the
        # tolerance holds for two consecutive sizes (patience), which is both the
        # research rule and a large saving over scoring the whole grid.
        keys = sorted(grid)
        chosen = n
        streak = 0
        for index, k in enumerate(keys):
            columns = ordered[:k]
            model = self.model(model_name, columns, cache_key=f"adaptive_{model_name}_{k}")
            score = evaluate_estimator(model, self.splits.X_val[columns], self.splits.y_val)["f1"]
            if score >= full_f1 - float(epsilon):
                streak += 1
                if streak >= 2:
                    chosen = keys[max(0, index - 1)]
                    break
            else:
                streak = 0

        self._knn_cache[key] = chosen
        return ordered[:chosen], chosen

    # ------------------------------------------------------------- live analysis
    def extract(self, url: str) -> Dict[str, float]:
        values = url_features.extract(url)
        return {name: float(values.get(name, 0.0)) for name in self.url_feature_names}

    def _aligned_frame(self, values: Dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame([[values[f] for f in self.url_feature_names]], columns=self.url_feature_names)

    def predict_url(self, url: str, models: Optional[Sequence[str]] = None) -> dict:
        started = time.perf_counter()
        methods = [m for m in (models or self.available_models()) if m in self.available_models()]
        if not methods:
            methods = ["RandomForest"]

        values = self.extract(url)
        frame = self._aligned_frame(values)

        predictions = []
        for name in methods:
            model = self.url_model(name)
            tick = time.perf_counter()
            probability = float(model.predict_proba(frame)[0][1])
            took = (time.perf_counter() - tick) * 1000.0
            predictions.append({
                "model": name,
                "model_label": MODEL_LABELS[name],
                "probability": round(probability, 4),
                "prediction": "Suspicious" if probability >= 0.5 else "Legitimate",
                "time_ms": round(took, 3),
            })

        headline = max(predictions, key=lambda p: p["probability"])
        elapsed = (time.perf_counter() - started) * 1000.0

        shap_table: List[dict] = []
        shap_note = None
        tree_model = next((m for m in methods if m != "KNN"), None)
        if tree_model is None:
            shap_note = (
                "Tree SHAP needs a tree ensemble. K-Nearest Neighbours has no tree "
                "structure, so no SHAP values are produced for it; the LIME "
                "explanation below is still available."
            )
        else:
            try:
                shap_table = self.shap_instance(tree_model, frame, values)
            except Exception as exc:
                shap_note = f"SHAP could not be computed ({type(exc).__name__}: {exc})."

        lime_table: List[dict] = []
        lime_note = None
        try:
            lime_table = self.lime_instance(headline["model"], frame)
        except Exception as exc:
            lime_note = f"LIME could not be computed ({type(exc).__name__}: {exc})."

        importance_rows = shap_table or lime_table
        return {
            "url": url,
            "features": [self._describe_feature(f, values[f], importance_rows) for f in self.url_feature_names],
            "importance_source": "shap" if shap_table else ("lime" if lime_table else None),
            "predictions": predictions,
            "headline": {
                "model": headline["model"],
                "model_label": headline["model_label"],
                "probability": headline["probability"],
                "prediction": headline["prediction"],
                "time_ms": round(elapsed, 2),
            },
            "shap": {"model": tree_model, "rows": shap_table, "note": shap_note},
            "lime": {"model": headline["model"], "rows": lime_table, "note": lime_note},
            "notes": [
                "Only URL-derived features are shown. Content, search-index, page-rank and "
                "domain-age features of the full benchmark require fetching the page or a "
                "third-party service, so they are not claimed here.",
                "This is a single-URL prediction, not a benchmark score: accuracy is only "
                "meaningful on a labelled dataset, reported in Model Playground.",
            ],
        }

    def _describe_feature(self, name: str, value: float, shap_table: List[dict]) -> dict:
        lookup = {row["feature"]: row for row in shap_table}
        row = lookup.get(name)
        importance = abs(row["value"]) if row else None
        direction = None
        if row:
            direction = "towards suspicious" if row["value"] > 0 else "towards legitimate"
        return {
            "feature": name,
            "value": round(float(value), 4),
            "importance": round(float(importance), 4) if importance is not None else None,
            "direction": direction,
            "description": url_features.DESCRIPTIONS.get(name, ""),
            "interpretation": _interpret(name, value),
        }

    # ------------------------------------------------------------ explanations
    def shap_instance(self, model_name: str, frame: pd.DataFrame, values: Dict[str, float],
                      reference: Optional[pd.DataFrame] = None) -> List[dict]:
        import shap

        model = self.url_model(model_name)
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(frame)
        arr = np.asarray(raw)
        if isinstance(raw, list):
            arr = np.asarray(raw[1])
        elif arr.ndim == 3:
            arr = arr[:, :, 1]
        arr = arr.reshape(-1)
        rows = [
            {"feature": name, "value": float(arr[i]), "feature_value": float(values[name])}
            for i, name in enumerate(self.url_feature_names)
        ]
        rows.sort(key=lambda r: -abs(r["value"]))
        for row in rows:
            row["direction"] = "towards suspicious" if row["value"] > 0 else "towards legitimate"
        return rows

    def lime_instance(self, model_name: str, frame: pd.DataFrame) -> List[dict]:
        model = self.url_model(model_name)
        instance = frame.iloc[0]
        _, table = explain_instance_lime(
            model,
            self.X_train_url,
            self.url_feature_names,
            instance,
            self.cfg,
            num_features=min(12, len(self.url_feature_names)),
        )
        return [
            {
                "feature": row["feature"],
                "value": round(float(row["weight"]), 4),
                "direction": row["direction"],
            }
            for _, row in table.iterrows()
        ]

    def global_shap(self, model_name: str, top: int = 20) -> dict:
        import shap

        key = f"globalshap_{model_name}"
        if key in self._eval_cache:
            cached = self._eval_cache[key]
            return {**cached, "rows": cached["rows"][:top]}

        if model_name == "KNN":
            raise RuntimeError(
                "Tree SHAP is not defined for K-Nearest Neighbours. Choose Random Forest, "
                "XGBoost or LightGBM for global SHAP importance."
            )

        model = self.url_model(model_name)
        sample = self.X_test_url.sample(
            n=min(400, len(self.X_test_url)), random_state=self.cfg.seed
        )
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(sample)
        arr = np.asarray(raw[1]) if isinstance(raw, list) else np.asarray(raw)
        if arr.ndim == 3:
            arr = arr[:, :, 1]
        means = np.abs(arr).mean(axis=0)
        rows = sorted(
            (
                {"feature": name, "importance": round(float(means[i]), 5), "rank": 0}
                for i, name in enumerate(self.url_feature_names)
            ),
            key=lambda r: -r["importance"],
        )
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        payload = {
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "n_samples": int(len(sample)),
            "rows": rows,
        }
        self._eval_cache[key] = payload
        return {**payload, "rows": rows[:top]}

    # ------------------------------------------------------------------ dataset
    def dataset_info(self) -> dict:
        counts = self.raw["status"].value_counts().to_dict()
        cleaned_counts = {
            "phishing": int((self.bundle.y == 1).sum()),
            "legitimate": int((self.bundle.y == 0).sum()),
        }
        feature_columns = [c for c in self.raw.columns if c not in ("url", "status")]
        missing = int(self.raw.isna().sum().sum())
        split_summary = self.splits.summary().to_dict(orient="records")
        removed = [c for c in feature_columns if c not in self.bundle.X.columns]
        return {
            "name": "Phishing Dataset (Hannousse & Yahiouche)",
            "source": "Mendeley Data, doi:10.17632/c2gw7fy2j4.3",
            "url_column": "url",
            "raw_shape": [int(self.raw.shape[0]), int(self.raw.shape[1])],
            "n_samples": int(len(self.raw)),
            "n_features_raw": len(feature_columns),
            "n_features_modelled": int(self.bundle.X.shape[1]),
            "n_features_served": len(self.url_feature_names),
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "class_counts_clean": cleaned_counts,
            "missing_values": missing,
            "duplicate_feature_rows": self.audit.get("duplicate_feature_rows", 0),
            "constant_features_removed": removed,
            "splits": split_summary,
            "expected_from_paper": {"n_samples": 11430, "n_features": 87},
            "matches_paper": bool(len(self.raw) == 11430 and len(feature_columns) == 87),
            "cleaning_report": self.bundle.report,
            "served_features": self.url_feature_names,
            "fidelity": {
                name: self.fidelity[name] for name in self.fidelity
                if name in url_features.DESCRIPTIONS
            },
        }

    def models_info(self) -> dict:
        return {
            "models": [
                {
                    "name": name,
                    "label": MODEL_LABELS[name],
                    "available": True,
                    "default": name != "LightGBM",
                    "params": self._params_for(name),
                    "note": MODEL_NOTES[name],
                    "tree_shap": name != "KNN",
                }
                for name in self.available_models()
            ],
            "methods": [
                {
                    "name": name,
                    "label": METHOD_LABELS[name],
                    "description": (
                        "All URL-derived features, no selection. The baseline every other row is compared against."
                        if name == "Full" else
                        "Picks the smallest subset whose validation F1 stays within epsilon of the full model, then refits."
                        if name == "Adaptive" else
                        "Rank-average fusion of the SHAP and LIME orderings." if name == "SLA-FS" else
                        SELECTOR_METHODS.get(name, "")
                    ),
                    "available": name in ("Full", "Adaptive", "SLA-FS") or name in self.rankings,
                }
                for name in METHOD_ORDER
            ],
            "defaults": {
                "k": DEFAULT_K,
                "epsilon": 0.02,
                "cv_folds": self.cfg.tuning_cv_folds,
                "seed": self.cfg.seed,
                "max_k": len(self.url_feature_names),
                "models": ["RandomForest", "XGBoost", "KNN"],
                "methods": ["Full", "SHAP", "LIME"],
            },
            "dataset": {
                "n_eval_samples": int(len(self.splits.y_test)),
                "eval_split": "held-out test split (20% of the cleaned dataset)",
                "n_features_available": len(self.url_feature_names),
            },
        }

    def compare(self, models: Sequence[str], methods: Sequence[str], k: int,
                epsilon: float) -> dict:
        started = time.perf_counter()
        results, skipped = [], []
        for method in methods:
            for model_name in models:
                try:
                    results.append(self.evaluate(model_name, method, k=k, epsilon=epsilon))
                except Exception as exc:
                    skipped.append({
                        "model": MODEL_LABELS.get(model_name, model_name),
                        "method": METHOD_LABELS.get(method, method),
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
        return {
            "results": results,
            "skipped": skipped,
            "elapsed_s": round(time.perf_counter() - started, 2),
            "eval_split": "held-out test split",
            "n_eval_samples": int(len(self.splits.y_test)),
            "k": int(k),
            "epsilon": float(epsilon),
        }


def _interpret(name: str, value: float) -> str:
    """A factual sentence about one feature value (no model speculation)."""
    if name in ("https_token",):
        return "No HTTPS in the URL" if value >= 1 else "HTTPS appears in the URL"
    if name == "ip":
        return "Host is a raw IP address" if value >= 1 else "Host is a domain name"
    if name == "punycode":
        return "Punycode host (possible lookalike domain)" if value >= 1 else "No punycode in host"
    if name == "port":
        return "Explicit port in the URL" if value >= 1 else "Default port implied"
    if name == "length_url":
        if value > 120:
            return "Unusually long URL"
        if value > 75:
            return "Above-average URL length"
        return "Typical URL length"
    if name == "length_hostname":
        if value > 30:
            return "Long hostname"
        if value < 10:
            return "Short hostname"
        return "Typical hostname length"
    if name == "nb_dots":
        return "Many dot separators" if value >= 4 else "Few dot separators"
    if name == "nb_hyphens":
        return "Hyphenated host or path" if value >= 2 else "No meaningful hyphenation"
    if name == "nb_slash":
        return "Deep path structure" if value >= 6 else "Shallow path structure"
    if name == "nb_at":
        return "Contains '@' (can mask the real host)" if value >= 1 else "No '@' in URL"
    if name == "ratio_digits_url":
        return "Digit-heavy URL" if value > 0.15 else "Mostly alphabetic URL"
    if name == "nb_www":
        return "Repeated 'www' token" if value >= 2 else ("Standard 'www' prefix" if value == 1 else "No 'www' token")
    if name == "suspecious_tld":
        return "TLD frequently abused for phishing" if value >= 1 else "TLD not on the abuse list"
    if name == "tld_in_subdomain":
        return "TLD token inside a subdomain" if value >= 1 else "No TLD token in subdomains"
    if name == "http_in_path":
        return "Embedded 'http' in the path" if value >= 1 else "No 'http' in the path"
    if name == "nb_dslash":
        return "Double slash inside the path" if value >= 1 else "No double slash in the path"
    if name == "longest_words_raw":
        return "Very long unbroken token" if value > 30 else "Token lengths look ordinary"
    if name == "length_words_raw":
        return "Many URL tokens" if value > 12 else "Few URL tokens"
    if name == "nb_percent":
        return "Percent-encoded characters present" if value >= 1 else "No percent-encoding"
    if name == "nb_tilde":
        return "Tilde present" if value >= 1 else "No tilde"
    if name == "nb_star":
        return "Asterisk present" if value >= 1 else "No asterisk"
    if name == "nb_space":
        return "Whitespace or %20 present" if value >= 1 else "No whitespace"
    return f"Raw value {value:g}"


engine: Optional[Engine] = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    global engine
    if engine is None:
        with _engine_lock:
            if engine is None:
                engine = Engine()
    return engine
