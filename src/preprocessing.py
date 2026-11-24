"""
preprocessing.py
================
Dataset acquisition verification, validation, cleaning and leakage-safe splitting.

The dataset is the one used by the reference paper (Shafin, 2025,
DOI 10.1016/j.dsm.2024.08.004): Hannousse & Yahiouche (2021),
"Web page phishing detection", Mendeley Data V3, DOI 10.17632/c2gw7fy2j4.3,
CC BY 4.0. 11,430 URLs x 87 features.

Leakage-control rules implemented here
--------------------------------------
1. ``url`` is an identifier, not a feature -> dropped before modelling.
2. Exact duplicate feature rows are removed *before* splitting.
3. A URL whose two records carry conflicting labels is removed entirely
   (ambiguous label; keeping it risks identical-instance leakage).
4. Zero-variance features are removed (no information, undefined correlations).
5. All subsequent scaling/imputation is fitted inside a scikit-learn Pipeline,
   i.e. on training folds only (see ``models.build_pipeline``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ExperimentConfig, METRICS_DIR, RAW_DATASET

LABEL_COLUMN = "status"
ID_COLUMN = "url"
POSITIVE_LABEL = "phishing"
NEGATIVE_LABEL = "legitimate"


# --------------------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------------------
@dataclass
class DatasetBundle:
    """Cleaned dataset plus the audit trail of how it was produced."""

    X: pd.DataFrame
    y: pd.Series
    report: Dict[str, object] = field(default_factory=dict)

    @property
    def feature_names(self) -> List[str]:
        return list(self.X.columns)

    def __repr__(self) -> str:  # pragma: no cover
        return f"DatasetBundle(n={len(self.X)}, p={self.X.shape[1]})"


@dataclass
class DataSplits:
    """Stratified train / validation / test split (features + labels)."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, y in (
            ("train", self.y_train),
            ("validation", self.y_val),
            ("test", self.y_test),
        ):
            n = len(y)
            positives = int((y == 1).sum())
            rows.append(
                {
                    "split": name,
                    "n_samples": n,
                    "n_phishing": positives,
                    "n_legitimate": n - positives,
                    "phishing_rate": round(positives / n, 4) if n else np.nan,
                }
            )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Loading & validation
# --------------------------------------------------------------------------------------
def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV exactly as published (URL-indexed, 87 features + label)."""
    path = Path(path or RAW_DATASET)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python main.py --stage download` "
            "or fetch it from https://data.mendeley.com/datasets/c2gw7fy2j4/3"
        )
    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame, expected_n: int = 11430, expected_p: int = 87) -> Dict[str, object]:
    """Audit the raw file against the dimensions reported in the reference paper.

    Returns a JSON-serialisable report. Nothing is modified here.
    """
    report: Dict[str, object] = {
        "shape": list(df.shape),
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "expected_from_paper": {"n_samples": expected_n, "n_features": expected_p},
    }

    report["matches_paper_dimensions"] = bool(
        df.shape[0] == expected_n and df.shape[1] == expected_p + 2  # 87 features + url + status
    )

    report["columns_present"] = {
        "id_column": ID_COLUMN in df.columns,
        "label_column": LABEL_COLUMN in df.columns,
    }
    report["missing_values"] = int(df.isna().sum().sum())
    report["missing_values_per_column"] = {
        k: int(v) for k, v in df.isna().sum().items() if v > 0
    }
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["duplicate_feature_rows"] = int(
        df.drop(columns=[c for c in (ID_COLUMN,) if c in df.columns]).duplicated().sum()
    )

    if LABEL_COLUMN in df.columns:
        counts = df[LABEL_COLUMN].value_counts(dropna=False).to_dict()
        report["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        report["is_balanced"] = bool(
            len(counts) == 2 and abs(counts.get(POSITIVE_LABEL, 0) - counts.get(NEGATIVE_LABEL, 0)) == 0
        )

    if ID_COLUMN in df.columns:
        report["n_unique_urls"] = int(df[ID_COLUMN].nunique())
        duplicated_urls = df[ID_COLUMN][df[ID_COLUMN].duplicated(keep=False)]
        report["n_urls_appearing_multiple_times"] = int(duplicated_urls.nunique())
        if LABEL_COLUMN in df.columns and len(duplicated_urls):
            conflicting = (
                df[df[ID_COLUMN].isin(duplicated_urls.unique())]
                .groupby(ID_COLUMN)[LABEL_COLUMN]
                .nunique()
            )
            report["n_urls_with_conflicting_labels"] = int((conflicting > 1).sum())

    feature_columns = [
        c for c in df.columns if c not in (ID_COLUMN, LABEL_COLUMN)
    ]
    numeric = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    report["n_numeric_features"] = len(numeric)
    report["non_numeric_features"] = [c for c in feature_columns if c not in numeric]
    report["constant_features"] = [
        c for c in numeric if df[c].nunique(dropna=False) <= 1
    ]
    report["features_with_negative_values"] = [
        c for c in numeric if (df[c] < 0).any()
    ]
    return report


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame, cfg: ExperimentConfig) -> DatasetBundle:
    """Apply the documented cleaning decisions and return a modelling frame."""
    report: Dict[str, object] = {"steps": []}
    working = df.copy()

    n_before = len(working)

    # 1. Ambiguous labels: same URL, different classes -> remove the whole group.
    if cfg.drop_duplicate_urls and ID_COLUMN in working.columns and LABEL_COLUMN in working.columns:
        dup_urls = working[ID_COLUMN][working[ID_COLUMN].duplicated(keep=False)].unique()
        conflicting = []
        if len(dup_urls):
            grouped = working[working[ID_COLUMN].isin(dup_urls)].groupby(ID_COLUMN)[LABEL_COLUMN].nunique()
            conflicting = grouped[grouped > 1].index.tolist()
        if conflicting:
            working = working[~working[ID_COLUMN].isin(conflicting)]
        report["steps"].append(
            {
                "step": "ambiguous_url_removal",
                "n_urls_with_conflicting_labels": len(conflicting),
                "rows_removed": n_before - len(working),
            }
        )

    # 2. Remove rows whose *feature vector* is an exact duplicate (identical instance).
    before = len(working)
    feature_columns = [c for c in working.columns if c not in (ID_COLUMN, LABEL_COLUMN)]
    working = working.drop_duplicates(subset=feature_columns, keep="first").reset_index(drop=True)
    report["steps"].append(
        {"step": "duplicate_feature_row_removal", "rows_removed": before - len(working)}
    )

    # 3. Missing values: none exist in this dataset, but handle generically.
    before = len(working)
    numeric_columns = working[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    if working[numeric_columns].isna().any().any():
        working[numeric_columns] = working[numeric_columns].fillna(
            working[numeric_columns].median()
        )
    report["steps"].append(
        {"step": "missing_value_imputation", "rows_removed": 0, "strategy": "median (train-fitted in pipeline)"}
    )

    # 4. Categorical encoding: this dataset is already fully numeric.
    categorical_columns = [
        c for c in feature_columns
        if c not in numeric_columns
    ]
    encoded = {}
    for column in categorical_columns:
        codes, uniques = pd.factorize(working[column], sort=True)
        working[column] = codes
        encoded[column] = list(map(str, uniques))
    report["steps"].append(
        {
            "step": "categorical_encoding",
            "n_categorical_features_found": len(categorical_columns),
            "encoding": encoded,
            "note": "Dataset is fully numeric; no encoding was required beyond the label.",
        }
    )

    # 5. Zero-variance features carry no information.
    constant = [c for c in numeric_columns if working[c].nunique(dropna=False) <= 1]
    if cfg.drop_constant_features and constant:
        working = working.drop(columns=constant)
    report["steps"].append(
        {
            "step": "constant_feature_removal",
            "features_removed": constant,
            "n_removed": len(constant),
        }
    )

    # 6. Label encoding: phishing -> 1, legitimate -> 0.
    y = (working[LABEL_COLUMN] == POSITIVE_LABEL).astype(int)
    X = working.drop(columns=[c for c in (ID_COLUMN, LABEL_COLUMN) if c in working.columns])
    X = X.astype(np.float64)

    report["n_before_cleaning"] = n_before
    report["n_after_cleaning"] = len(X)
    report["n_features_after_cleaning"] = int(X.shape[1])
    report["feature_names"] = list(X.columns)
    report["remaining_missing_values"] = int(X.isna().sum().sum())
    report["class_counts_after"] = {
        "phishing": int((y == 1).sum()),
        "legitimate": int((y == 0).sum()),
    }

    return DatasetBundle(X=X, y=y, report=report)


# --------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------
def make_splits(bundle: DatasetBundle, cfg: ExperimentConfig) -> DataSplits:
    """Stratified train / validation / test split.

    The test split is created once, here, and is never touched again until the
    final evaluation, after feature selection *and* hyper-parameter decisions.
    """
    stratify = bundle.y if cfg.stratify else None

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        bundle.X, bundle.y,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=stratify,
    )

    # Validation is carved out of the remaining data. Its size relative to the
    # full dataset is cfg.val_size, so the fraction of the train+val block is:
    relative_val = cfg.val_size / (1.0 - cfg.test_size)
    stratify_tv = y_trainval if cfg.stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=relative_val,
        random_state=cfg.seed,
        stratify=stratify_tv,
    )

    return DataSplits(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )


# --------------------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------------------
def class_distribution(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts().sort_index()
    total = len(y)
    return pd.DataFrame(
        {
            "class": ["legitimate", "phishing"],
            "n": [int(counts.get(0, 0)), int(counts.get(1, 0))],
            "proportion": [
                round(float(counts.get(0, 0) / total), 4) if total else np.nan,
                round(float(counts.get(1, 0) / total), 4) if total else np.nan,
            ],
        }
    )


def correlation_analysis(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Correlation matrix and highly-correlated pairs, computed on TRAIN ONLY.

    Reducing redundancy is tempting, but dropping correlated features changes the
    feature set and would confound the feature-selection comparison, so pairs above
    ``threshold`` are *reported*, not removed (see README, "Design decisions").
    """
    corr = X_train.corr(method="pearson")

    # Highly correlated pairs (upper triangle only).
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "pearson_r"})
    )
    pairs["abs_r"] = pairs["pearson_r"].abs()
    pairs = pairs[pairs["abs_r"] >= threshold].sort_values("abs_r", ascending=False)

    # Point-biserial correlation of each feature with the target.
    target_corr = (
        X_train.corrwith(y_train)
        .rename("correlation_with_target")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    target_corr["abs_correlation"] = target_corr["correlation_with_target"].abs()
    target_corr = target_corr.sort_values("abs_correlation", ascending=False).reset_index(drop=True)

    return corr, pairs, target_corr  # type: ignore[return-value]


def save_json(payload: Dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def run_preprocessing_pipeline(cfg: ExperimentConfig) -> Tuple[DataSplits, DatasetBundle, Dict[str, object]]:
    """End-to-end Stage 2: load -> validate -> clean -> split -> analyse -> save."""
    df = load_raw_dataset()
    validation = validate_dataset(df)
    bundle = clean_dataset(df, cfg)
    splits = make_splits(bundle, cfg)

    corr, pairs, target_corr = correlation_analysis(
        splits.X_train, splits.y_train, threshold=cfg.correlation_threshold
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(validation, METRICS_DIR / f"dataset_validation{cfg.suffix}.json")
    save_json(bundle.report, METRICS_DIR / f"preprocessing_report{cfg.suffix}.json")

    splits.summary().to_csv(METRICS_DIR / f"split_summary{cfg.suffix}.csv", index=False)
    class_distribution(bundle.y).to_csv(
        METRICS_DIR / f"class_distribution{cfg.suffix}.csv", index=False
    )
    target_corr.to_csv(METRICS_DIR / f"target_correlation{cfg.suffix}.csv", index=False)

    pairs.to_csv(METRICS_DIR / f"highly_correlated_pairs{cfg.suffix}.csv", index=False)
    corr.to_csv(METRICS_DIR / f"correlation_matrix_train{cfg.suffix}.csv")

    summary = {
        "validation": validation,
        "cleaning": bundle.report,
        "splits": splits.summary().to_dict(orient="records"),
        "n_highly_correlated_pairs": int(len(pairs)),
        "top_10_features_by_target_correlation": target_corr.head(10).to_dict(orient="records"),
    }
    return splits, bundle, summary
