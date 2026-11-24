"""
main.py
=======
Staged command-line entry point for the study

    "Lightweight and Explainable Phishing Detection Using Adaptive Feature Selection"

Stages
------
``download``         fetch the published dataset and verify its SHA-256
``verify``           audit the raw file against the reference paper's dimensions
``preprocess``       load -> validate -> clean -> split -> analyse -> save
``baselines``        tune hyper-parameters, evaluate the full-feature model
``feature_selection`` run the six selectors + SLA-FS, rank stability
``adaptive``         sweep subset sizes and apply the adaptive rule
``experiments``      baselines + feature_selection + adaptive + final test (one process)
``visualize``        rebuild every figure from the saved tables
``all``              experiments + visualize

Design rules enforced here
--------------------------
1. **The test split is created once** (in ``preprocessing.make_splits``) and is
   only touched by ``_final_test_evaluation``, which runs after every feature
   and hyper-parameter decision has been made.
2. **Hyper-parameters are tuned once on the training split** and reused for all
   subsets, so subset comparisons are not confounded by per-subset tuning.
3. **Nothing is written unless it was computed.** Every CSV/JSON in
   ``results/`` originates from a measurement in this run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.adaptive_selection import (
    AdaptiveSelectionResult,
    adaptive_select,
    build_k_grid,
    refine_k_grid,
    summarize_across_epsilons,
)
from src.config import (
    FIG_DIR,
    METRICS_DIR,
    RANKINGS_DIR,
    RAW_DATASET,
    ExperimentConfig,
    describe_environment,
    set_global_seed,
)
from src.evaluation import (
    Timer,
    evaluate_estimator,
    format_metrics_table,
    roc_points,
    save_json,
)
from src.explainability import (
    build_slafs_rankings,
    explain_instance_lime,
    mean_abs_shap_table,
    provenance_table,
    shap_contributions_frame,
    slafs_intersection_report,
    slafs_rank_average,
    top_interaction_pairs,
)
from src.feature_selection import (
    SELECTOR_ORDER,
    FeatureRanking,
    SelectionContext,
    compute_stability_table,
    pairwise_ranking_agreement,
    rank_correlation_matrix,
    rankings_summary_table,
    run_all_selectors,
    save_combined_rankings,
    stability_rankings,
)
from src.models import build_estimator, tune_all_models
from src.preprocessing import (
    class_distribution,
    load_raw_dataset,
    run_preprocessing_pipeline,
    validate_dataset,
)

# --------------------------------------------------------------------------------------
# Dataset provenance constants (see docs/DATASET_SOURCE_REGISTER.md)
# --------------------------------------------------------------------------------------
MENDELEY_DATASET_ID = "c2gw7fy2j4"
MENDELEY_VERSION = 3
MENDELEY_API = (
    f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET_ID}"
    f"/files?folder_id=root&version={MENDELEY_VERSION}"
)
MENDELEY_FILE_URL = (
    f"https://data.mendeley.com/public-files/datasets/{MENDELEY_DATASET_ID}"
    "/files/{file_id}/file_downloaded"
)
EXPECTED_SHA256 = "21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c"

# The SLA-FS intersection variant is defined relative to a reference subset size.
SLAFS_REFERENCE_K = 15

# Cap on the number of subset sizes evaluated per (method, model): the sweep is a
# curve estimate, not a 1..p grid search, and this keeps the run laptop-sized.
MAX_SWEEP_POINTS = 14


# ======================================================================================
# Stage 1 - acquisition and verification
# ======================================================================================
def _fetch_json(url: str, timeout: int = 120) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "phishing-research/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_download(force: bool = False) -> Dict[str, object]:
    """Download the published CSV from Mendeley Data and verify its checksum."""
    print("=" * 78)
    print("STAGE 1  Dataset acquisition")
    print("=" * 78)

    metadata: Dict[str, object] = {
        "dataset": "Web page phishing detection",
        "authors": "Hannousse, A. & Yahiouche, S. (2021)",
        "repository": "Mendeley Data",
        "doi": f"10.17632/{MENDELEY_DATASET_ID}.{MENDELEY_VERSION}",
        "version": MENDELEY_VERSION,
        "api_url": MENDELEY_API,
        "expected_sha256": EXPECTED_SHA256,
    }

    if RAW_DATASET.exists() and not force:
        print(f"  file already present: {RAW_DATASET.name}")
        actual = _sha256(RAW_DATASET)
        metadata.update({"bytes": RAW_DATASET.stat().st_size, "sha256": actual,
                         "sha256_matches": actual == EXPECTED_SHA256, "downloaded": False})
        print(f"  sha256 {actual}")
        print(f"  checksum match: {metadata['sha256_matches']}")
        save_json(metadata, RAW_DATASET.parent / "dataset_metadata.json")
        return metadata

    listing = _fetch_json(MENDELEY_API)
    files = listing if isinstance(listing, list) else listing.get("files", [])  # type: ignore[union-attr]
    if not files:
        raise RuntimeError(f"Mendeley returned no files for {MENDELEY_API}")

    candidates = [f for f in files if str(f.get("filename", "")).lower().endswith(".csv")]
    chosen = next((f for f in candidates if "dataset_B" in str(f.get("filename", ""))), None)
    chosen = chosen or candidates[0]

    file_id = chosen["id"]
    file_name = chosen.get("filename", "dataset.csv")
    print(f"  source file: {file_name} ({int(chosen.get('size', 0)):,} bytes)")
    metadata.update({"source_filename": file_name, "source_file_id": file_id,
                     "api_reported_size": chosen.get("size")})

    RAW_DATASET.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = RAW_DATASET.with_suffix(".part")
    request = urllib.request.Request(
        MENDELEY_FILE_URL.format(file_id=file_id),
        headers={"User-Agent": "phishing-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=300) as response, open(tmp_path, "wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    tmp_path.replace(RAW_DATASET)

    actual = _sha256(RAW_DATASET)
    metadata.update({
        "bytes": RAW_DATASET.stat().st_size,
        "sha256": actual,
        "sha256_matches": actual == EXPECTED_SHA256,
        "downloaded": True,
    })
    print(f"  saved to {RAW_DATASET}")
    print(f"  sha256 {actual}")
    print(f"  checksum match: {metadata['sha256_matches']}")
    if not metadata["sha256_matches"]:
        print("  [warning] checksum differs from the file used in this study.")
    save_json(metadata, RAW_DATASET.parent / "dataset_metadata.json")
    return metadata


def stage_verify() -> Dict[str, object]:
    """Audit the raw file against the dimensions reported in the reference paper."""
    print("=" * 78)
    print("STAGE 2  Dataset verification")
    print("=" * 78)
    frame = load_raw_dataset()
    report = validate_dataset(frame)
    report["sha256"] = _sha256(RAW_DATASET)

    print(f"  shape                 : {report['n_rows']:,} rows x {report['n_columns']} columns")
    print(f"  matches paper (11430 x 87 + url + status): {report['matches_paper_dimensions']}")
    print(f"  class counts          : {report['class_counts']}")
    print(f"  balanced              : {report['is_balanced']}")
    print(f"  missing values        : {report['missing_values']}")
    print(f"  duplicate rows        : {report['duplicate_rows']}")
    print(f"  duplicate feature rows: {report['duplicate_feature_rows']}")
    print(f"  unique URLs           : {report.get('n_unique_urls')}")
    print(f"  constant features     : {report['constant_features']}")
    print(f"  non-numeric features  : {report['non_numeric_features']}")

    save_json(report, METRICS_DIR / "dataset_verification.json")
    return report


# ======================================================================================
# Shared experiment helpers
# ======================================================================================
def _tuned_params(tuned: Dict[str, Dict[str, object]], model: str) -> Dict[str, object]:
    entry = tuned.get(model) or {}
    return dict(entry.get("params") or {})


def _sweep_grid(n_features: int, cfg: ExperimentConfig) -> List[int]:
    """Coarse, capped subset-size grid (always contains 1 and n_features)."""
    grid = build_k_grid(n_features, cfg.k_step)
    if len(grid) <= MAX_SWEEP_POINTS:
        return grid
    positions = np.unique(np.linspace(0, len(grid) - 1, MAX_SWEEP_POINTS).round().astype(int))
    return [grid[i] for i in positions]


def _evaluate_subset(
    model: str,
    params: Dict[str, object],
    cfg: ExperimentConfig,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    features: Sequence[str],
) -> Dict[str, float]:
    """Fit on ``X_fit`` and evaluate on ``X_eval`` using only ``features``."""
    estimator = build_estimator(model, cfg, params)
    return evaluate_estimator(
        estimator,
        X_eval[list(features)],
        y_eval,
        fit=True,
        X_fit=X_fit[list(features)],
        y_fit=y_fit,
    )


def load_rankings(path: Path) -> Dict[str, FeatureRanking]:
    """Rebuild ranking objects from the wide ``feature_rankings`` CSV."""
    frame = pd.read_csv(path)
    rankings: Dict[str, FeatureRanking] = {}
    for column in frame.columns:
        if not column.startswith("rank_"):
            continue
        method = column[len("rank_"):]
        subset = frame[["feature", column]].dropna().sort_values(column)
        score_column = f"score_{method}"
        if score_column in frame.columns:
            scores = dict(zip(subset["feature"], subset[score_column].fillna(0.0)))
        else:
            scores = {f: float(-i) for i, f in enumerate(subset["feature"])}
        rankings[method] = FeatureRanking(
            method=method, features=list(subset["feature"]), scores=scores
        )
    return rankings


# ======================================================================================
# Stages 3-5 - baselines, feature selection, adaptive selection
# ======================================================================================
def _full_feature_baseline(
    cfg: ExperimentConfig,
    splits,
    tuned: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    """Validation metrics of every classifier on the complete feature set."""
    rows = []
    features = list(splits.X_train.columns)
    print("  full-feature baseline (validation split)")
    for model in cfg.model_names:
        result = _evaluate_subset(
            model, _tuned_params(tuned, model), cfg,
            splits.X_train, splits.y_train, splits.X_val, splits.y_val, features,
        )
        row = {**result, "model": model, "method": "Full",
               "n_features": len(features), "phase": "validation"}
        rows.append(row)
        print(f"    {model:14s} F1={row['f1']:.4f}  AUC={row['roc_auc']:.4f}  "
              f"fit={row['train_time_s']:.1f}s")
    return pd.DataFrame(rows)


def _run_stability(
    ctx: SelectionContext,
    cfg: ExperimentConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Jaccard stability of the top-k sets, plus the protocol actually used."""
    # A cheaper but identical-in-kind protocol for the repeats; recorded explicitly
    # so the stability numbers are never confused with the headline rankings.
    stability_cfg = replace(
        cfg,
        lime_num_samples=min(120, cfg.lime_num_samples),
        shap_max_samples=min(400, cfg.shap_max_samples),
        rfe_step=max(2, cfg.rfe_step),
        perm_cv_folds=2,
        perm_n_repeats=2,
    )
    protocol = {
        "repeats": int(stability_cfg.stability_repeats),
        "subsample_fraction": float(stability_cfg.stability_fraction),
        "selector_settings": {
            "lime_num_samples": stability_cfg.lime_num_samples,
            "shap_max_samples": stability_cfg.shap_max_samples,
            "rfe_step": stability_cfg.rfe_step,
            "perm_cv_folds": stability_cfg.perm_cv_folds,
            "perm_n_repeats": stability_cfg.perm_n_repeats,
        },
        "note": (
            "Repeats use a reduced-cost selector configuration. Stability measures the "
            "robustness of each selector's ordering, not its headline ranking."
        ),
    }

    print("  stability analysis")
    collected = stability_rankings(ctx, methods=SELECTOR_ORDER, cfg=stability_cfg)

    # SLA-FS_rankavg stability follows directly from the SHAP/LIME repeats.
    if collected.get("SHAP") and collected.get("LIME"):
        derived = []
        for shap_list, lime_list in zip(collected["SHAP"], collected["LIME"]):
            if not shap_list or not lime_list:
                continue
            blended = slafs_rank_average(
                FeatureRanking("SHAP", shap_list, {}), FeatureRanking("LIME", lime_list, {})
            )
            derived.append(blended.features)
        if derived:
            collected["SLA-FS_rankavg"] = derived

    k_values = [k for k in (5, 10, 15, 20, 30, 40) if k <= ctx.n_features]
    table = compute_stability_table(collected, k_values)
    return table, protocol


def _sla_variants(rankings: Dict[str, FeatureRanking], k: int) -> Dict[str, FeatureRanking]:
    variants = build_slafs_rankings(rankings, k=k)
    for name, ranking in variants.items():
        print(f"    {name:22s} |intersection/top-k| head: {', '.join(ranking.top_k(4))}")
    return variants


def _sweep_and_select(
    cfg: ExperimentConfig,
    splits,
    rankings: Dict[str, FeatureRanking],
    baseline: pd.DataFrame,
    tuned: Dict[str, Dict[str, object]],
) -> Tuple[pd.DataFrame, List[AdaptiveSelectionResult]]:
    """Evaluate each (method, model) over a capped k-grid and apply the rule.

    Returns the validation curves and the adaptive decisions. No classifier is
    cached here: the test stage refits the decided feature sets once each on
    train+validation, which is a different fit from these validation fits.
    """
    f1_full = dict(zip(baseline["model"], baseline["f1"]))
    n_features = splits.X_train.shape[1]
    grid = _sweep_grid(n_features, cfg)

    curve_rows: List[Dict[str, object]] = []
    decisions: List[AdaptiveSelectionResult] = []

    for method, ranking in rankings.items():
        if ranking.n_features == 0:
            print(f"  [skip] {method}: no ranking available")
            continue
        for model in cfg.model_names:
            if model not in f1_full or not np.isfinite(f1_full[model]):
                continue
            params = _tuned_params(tuned, model)

            print(f"  sweep {method:22s} {model:14s}", end="", flush=True)
            rows: List[Dict[str, object]] = []
            evaluated: Dict[int, float] = {}

            for k in grid:
                features = ranking.top_k(k)
                result = _evaluate_subset(
                    model, params, cfg, splits.X_train, splits.y_train,
                    splits.X_val, splits.y_val, features,
                )
                rows.append({**result, "model": model, "method": method,
                             "n_features": int(k), "phase": "validation"})
                evaluated[int(k)] = float(result["f1"])

            curve = pd.DataFrame(rows)
            coarse = {
                epsilon: adaptive_select(
                    curve, ranking.features, f1_full[model], epsilon,
                    method=method, model=model, patience=cfg.patience,
                )
                for epsilon in cfg.epsilons
            }

            # Refine around the smallest coarse solution so a finer k is not missed.
            if cfg.refine:
                target = min(p.selected_k for p in coarse.values())
                extras = [k for k in refine_k_grid(target, n_features, max(2, cfg.k_step))
                          if k not in evaluated]
                for k in extras:
                    features = ranking.top_k(k)
                    result = _evaluate_subset(
                        model, params, cfg, splits.X_train, splits.y_train,
                        splits.X_val, splits.y_val, features,
                    )
                    rows.append({**result, "model": model, "method": method,
                                 "n_features": int(k), "phase": "validation"})
                    evaluated[int(k)] = float(result["f1"])
                if extras:
                    curve = pd.DataFrame(rows)

            for epsilon in cfg.epsilons:
                decision = adaptive_select(
                    curve, ranking.features, f1_full[model], epsilon,
                    method=method, model=model, patience=cfg.patience,
                )
                decisions.append(decision)
                if decision.fallback_used:
                    tag = "no k met tolerance -> full set"
                else:
                    tag = f"k={decision.selected_k} ({decision.reduction_pct:.0f}% fewer)"
                print(f"\n      eps={epsilon:.2f}: {tag}", end="")

            curve_rows.extend(rows)
            print()

    curves = pd.DataFrame(curve_rows)
    return curves, decisions


def _final_test_evaluation(
    cfg: ExperimentConfig,
    splits,
    decision_rows: pd.DataFrame,
    selected_features: Dict[Tuple[str, str, float], List[str]],
    tuned: Dict[str, Dict[str, object]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit every decided configuration on train+val and evaluate **once** on test."""
    X_trainval = pd.concat([splits.X_train, splits.X_val], axis=0)
    y_trainval = pd.concat([splits.y_train, splits.y_val], axis=0)
    n_full = splits.X_train.shape[1]

    rows: List[Dict[str, object]] = []
    roc_rows: List[Dict[str, object]] = []
    cache: Dict[Tuple[str, Tuple[str, ...]], Dict[str, object]] = {}

    # (a) full-feature baselines
    jobs: List[Tuple[str, str, float, List[str]]] = [
        (model, "Full", float("nan"), list(X_trainval.columns)) for model in cfg.model_names
    ]
    # (b) adaptive selections
    for _, row in decision_rows.iterrows():
        key = (row["model"], row["method"], float(row["epsilon"]))
        features = selected_features.get(key)
        if features:
            jobs.append((row["model"], row["method"], float(row["epsilon"]), list(features)))

    print(f"  final test evaluation: {len(jobs)} configurations")
    for model, method, epsilon, features in jobs:
        signature = (model, tuple(features))
        if signature not in cache:
            # One fit per configuration: the same fitted estimator produces the
            # metrics and the ROC curve (no second, throw-away refit).
            estimator = build_estimator(model, cfg, _tuned_params(tuned, model))
            with Timer() as timer:
                estimator.fit(X_trainval[features], y_trainval)
            result = evaluate_estimator(estimator, splits.X_test[features], splits.y_test)
            result = {"train_time_s": timer.elapsed, **result}
            fpr, tpr, auc = roc_points(estimator, splits.X_test[features], splits.y_test)
            cache[signature] = {"result": result, "fpr": fpr, "tpr": tpr, "auc": auc}
            print(f"    {model:14s} {method:22s} k={len(features):2d}  "
                  f"F1={result['f1']:.4f}  AUC={auc:.4f}")

        entry = cache[signature]
        rows.append({
            **entry["result"],
            "model": model,
            "method": method,
            "epsilon": epsilon,
            "n_features": len(features),
            "reduction_pct": round(100.0 * (1.0 - len(features) / n_full), 2),
            "phase": "test",
        })

        # ROC points are stored for the headline configurations only.
        if method == "Full" or (np.isfinite(epsilon) and abs(epsilon - 0.02) < 1e-12):
            label = f"{model} | {method} | k={len(features)}"
            for x, y in zip(entry["fpr"], entry["tpr"]):
                roc_rows.append({"label": label, "model": model, "method": method,
                                 "n_features": len(features), "auc": entry["auc"],
                                 "fpr": float(x), "tpr": float(y)})

    table = pd.DataFrame(rows)
    roc = pd.DataFrame(roc_rows)

    # Confusion matrices: baselines + one reduced model per classifier.
    confusion_rows = []
    for model in cfg.model_names:
        full_row = table[(table["model"] == model) & (table["method"] == "Full")]
        if not full_row.empty:
            confusion_rows.append(full_row.iloc[0].to_dict())
        reduced = table[(table["model"] == model) & (table["method"] != "Full")]
        if not reduced.empty:
            best = reduced.sort_values("n_features").iloc[0]
            confusion_rows.append(best.to_dict())
    confusion = pd.DataFrame(confusion_rows)
    return table, roc, confusion


# ======================================================================================
# Stage 6 - full experiment
# ======================================================================================
def stage_experiments(cfg: ExperimentConfig, interactions: bool = False) -> Dict[str, object]:
    print("=" * 78)
    print("STAGE 6  Comparative experiments".center(78))
    print("=" * 78)

    set_global_seed(cfg.seed)
    cfg.save()
    save_json(
        {
            "environment": describe_environment(),
            "config": cfg.to_dict(),
            "provenance": provenance_table().to_dict(orient="records"),
            "slafs_reference_k": SLAFS_REFERENCE_K,
            "sweep_points": MAX_SWEEP_POINTS,
        },
        METRICS_DIR / f"run_metadata{cfg.suffix}.json",
    )

    # ---- preprocessing ------------------------------------------------------
    print("\n[1/6] Preprocessing and splitting")
    splits, bundle, _ = run_preprocessing_pipeline(cfg)
    print(f"  train {len(splits.X_train):,} | validation {len(splits.X_val):,} | "
          f"test {len(splits.X_test):,}")
    print(f"  features used: {splits.X_train.shape[1]}")

    class_distribution(bundle.y).to_csv(METRICS_DIR / f"class_distribution{cfg.suffix}.csv", index=False)

    # ---- tuning + baselines -------------------------------------------------
    print("\n[2/6] Hyper-parameter tuning (training split, CV) and baselines")
    tuned = tune_all_models(splits.X_train, splits.y_train, cfg)
    save_json(tuned, METRICS_DIR / f"tuning{cfg.suffix}.json")
    baseline = _full_feature_baseline(cfg, splits, tuned)
    baseline.to_csv(METRICS_DIR / f"full_feature_baseline{cfg.suffix}.csv", index=False)

    # ---- feature selection --------------------------------------------------
    print("\n[3/6] Feature selection (six methods + SLA-FS)")
    ctx = SelectionContext(
        X_train=splits.X_train, y_train=splits.y_train, cfg=cfg,
        X_val=splits.X_val, y_val=splits.y_val,
    )
    rankings = run_all_selectors(ctx)
    print(f"  surrogate fit time: {ctx.surrogate_fit_s:.1f}s")
    rankings.update(_sla_variants(rankings, k=SLAFS_REFERENCE_K))

    combined = save_combined_rankings(
        rankings,
        RANKINGS_DIR / f"feature_rankings{cfg.suffix}.csv",
        RANKINGS_DIR / f"feature_selection_summary{cfg.suffix}.csv",
    )
    summary_table = rankings_summary_table(rankings)
    summary_table["surrogate_fit_s"] = round(ctx.surrogate_fit_s, 3)
    summary_table.to_csv(RANKINGS_DIR / f"feature_selection_summary{cfg.suffix}.csv", index=False)

    rank_correlation_matrix(rankings).to_csv(
        RANKINGS_DIR / f"rank_correlation_spearman{cfg.suffix}.csv"
    )
    pairwise_ranking_agreement(rankings, k=SLAFS_REFERENCE_K).to_csv(
        RANKINGS_DIR / f"topk_jaccard_agreement{cfg.suffix}.csv"
    )
    if "SHAP" in rankings and "LIME" in rankings:
        slafs_intersection_report(
            rankings["SHAP"], rankings["LIME"],
            k_values=[5, 10, 15, 20, 25, 30, 40],
        ).to_csv(RANKINGS_DIR / f"slafs_intersection_report{cfg.suffix}.csv", index=False)

    print(f"  rankings saved: {combined.shape[1] - 1} columns over {len(combined)} features")

    # SHAP artefacts (needed by the visualisation stage, computed once here)
    shap_ranking = rankings.get("SHAP")
    if shap_ranking is not None and shap_ranking.extras.get("shap_values") is not None:
        values = np.asarray(shap_ranking.extras["shap_values"])
        sample_X = shap_ranking.extras["sample_X"]
        pd.DataFrame(values, columns=list(sample_X.columns)).to_csv(
            RANKINGS_DIR / f"shap_values_sample{cfg.suffix}.csv", index=False
        )
        sample_X.reset_index(drop=True).to_csv(
            RANKINGS_DIR / f"shap_sample_X{cfg.suffix}.csv", index=False
        )
        mean_abs_shap_table(values, list(sample_X.columns)).to_csv(
            RANKINGS_DIR / f"shap_mean_abs{cfg.suffix}.csv", index=False
        )
        if interactions:
            print("  computing bounded SHAP interaction pairs (optional)")
            pairs = top_interaction_pairs(
                ctx.fitted_surrogate(), sample_X, list(sample_X.columns), max_samples=40
            )
            pairs.to_csv(RANKINGS_DIR / f"shap_interaction_pairs{cfg.suffix}.csv", index=False)

    # ---- stability ----------------------------------------------------------
    print("\n[4/6] Feature-selection stability (Jaccard across repeats)")
    stability_table, protocol = _run_stability(ctx, cfg)
    stability_table.to_csv(METRICS_DIR / f"stability{cfg.suffix}.csv", index=False)
    save_json(protocol, METRICS_DIR / f"stability_protocol{cfg.suffix}.json")
    print(format_metrics_table(stability_table[stability_table["k"] == 10].rename(
        columns={"mean_jaccard": "f1"})) if not stability_table.empty else "  (none)")

    # ---- adaptive sweeps ----------------------------------------------------
    print("\n[5/6] Subset sweeps and adaptive selection")
    curves, decisions = _sweep_and_select(cfg, splits, rankings, baseline, tuned)
    curves.to_csv(METRICS_DIR / f"validation_curves{cfg.suffix}.csv", index=False)

    decision_rows = summarize_across_epsilons(decisions)
    provenance = {
        **{name: "ESTABLISHED" for name in SELECTOR_ORDER},
        "SLA-FS_rankavg": "MODIFICATION",
        "SLA-FS_scoreblend": "MODIFICATION",
        "SLA-FS_intersection": "REPRODUCTION",
    }
    decision_rows["provenance"] = decision_rows["method"].map(provenance)
    decision_rows.to_csv(METRICS_DIR / f"adaptive_selection{cfg.suffix}.csv", index=False)

    selected_features: Dict[Tuple[str, str, float], List[str]] = {}
    feature_rows = []
    for decision in decisions:
        key = (decision.model, decision.method, float(decision.epsilon))
        selected_features[key] = list(decision.selected_features)
        feature_rows.append({
            "method": decision.method, "model": decision.model,
            "epsilon": decision.epsilon, "n_features": decision.selected_k,
            "provenance": provenance.get(decision.method, "ESTABLISHED"),
            "features": ";".join(decision.selected_features),
        })
    pd.DataFrame(feature_rows).to_csv(
        RANKINGS_DIR / f"adaptive_selected_features{cfg.suffix}.csv", index=False
    )
    print(f"  {len(decisions)} adaptive decisions recorded")

    # ---- final test ---------------------------------------------------------
    print("\n[6/6] Final evaluation on the held-out test split (first and only use)")
    test_table, roc_table, confusion = _final_test_evaluation(
        cfg, splits, decision_rows, selected_features, tuned
    )
    test_table.to_csv(METRICS_DIR / f"final_test_metrics{cfg.suffix}.csv", index=False)
    roc_table.to_csv(METRICS_DIR / f"roc_curves{cfg.suffix}.csv", index=False)
    confusion.to_csv(METRICS_DIR / f"confusion_matrices{cfg.suffix}.csv", index=False)

    # ---- LIME example on a test instance ------------------------------------
    try:
        final_model = build_estimator("RandomForest", cfg, _tuned_params(tuned, "RandomForest"))
        final_model.fit(splits.X_train, splits.y_train)
        phishing_rows = splits.y_test[splits.y_test == 1]
        if len(phishing_rows):
            instance = splits.X_test.loc[phishing_rows.index[0]]
            _, table = explain_instance_lime(
                final_model, splits.X_train, list(splits.X_train.columns),
                instance, cfg, num_features=15,
            )
            table.to_csv(RANKINGS_DIR / f"lime_example{cfg.suffix}.csv", index=False)
            print("  LIME example explanation saved")
    except Exception as exc:  # pragma: no cover
        print(f"  [warn] LIME example could not be produced: {exc}")

    print("\n  Test-set summary (full vs adaptive at epsilon=0.02)")
    view = test_table[
        (test_table["method"] == "Full") | (np.abs(test_table["epsilon"] - 0.02) < 1e-12)
    ]
    print(format_metrics_table(view))

    return {
        "baseline": baseline,
        "rankings": rankings,
        "curves": curves,
        "decisions": decision_rows,
        "test": test_table,
        "roc": roc_table,
        "confusion": confusion,
        "stability": stability_table,
        "summary": summary_table,
    }


# ======================================================================================
# Stage 7 - figures
# ======================================================================================
def _read(name: str, cfg: ExperimentConfig, index_col=None) -> Optional[pd.DataFrame]:
    path = METRICS_DIR / f"{name}{cfg.suffix}.csv"
    if not path.exists():
        path = RANKINGS_DIR / f"{name}{cfg.suffix}.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=index_col)
    except Exception:
        return None


def stage_visualize(cfg: ExperimentConfig) -> List[Path]:
    """Rebuild every figure from the persisted tables."""
    from src import visualization as viz

    print("=" * 78)
    print("STAGE 7  Figures".center(78))
    print("=" * 78)

    produced: List[Path] = []

    def _add(path):
        if path is not None:
            produced.append(path)
            print(f"  {Path(path).name}")

    # ---- dataset description ------------------------------------------------
    _add(viz.plot_class_distribution(_read("class_distribution", cfg), cfg))
    _add(viz.plot_target_correlation(_read("target_correlation", cfg), cfg=cfg))
    _add(viz.plot_correlated_pairs(_read("highly_correlated_pairs", cfg), cfg))
    # The correlation matrix is stored with its feature names as the index.
    _add(viz.plot_correlation_structure(
        _read("correlation_matrix_train", cfg, index_col=0), cfg))

    # ---- rankings -----------------------------------------------------------
    rankings_frame = _read("feature_rankings", cfg)
    if rankings_frame is not None and not rankings_frame.empty:
        rank_columns = [c for c in rankings_frame.columns if c.startswith("rank_")]
        ranks = rankings_frame.set_index("feature")[rank_columns]
        ranks.columns = [c[len("rank_"):] for c in rank_columns]
        ranks = ranks.apply(pd.to_numeric, errors="coerce")
        _add(viz.plot_rank_heatmap(ranks, cfg=cfg))
        _add(viz.plot_rank_correlation(ranks.corr(method="spearman"), cfg))

        long_rows = []
        for column in rankings_frame.columns:
            if not column.startswith("score_"):
                continue
            method = column[len("score_"):]
            part = rankings_frame[["feature", column]].dropna()
            long_rows.append(pd.DataFrame(
                {"feature": part["feature"], "method": method, "score": part[column]}
            ))
        if long_rows:
            _add(viz.plot_feature_importance(pd.concat(long_rows, ignore_index=True), cfg=cfg))

    _add(viz.plot_execution_times(_read("feature_selection_summary", cfg), cfg))

    # ---- SHAP / LIME --------------------------------------------------------
    shap_values = _read("shap_values_sample", cfg)
    shap_sample = _read("shap_sample_X", cfg)
    if shap_values is not None and shap_sample is not None and len(shap_values) == len(shap_sample):
        _add(viz.plot_shap_summary(shap_values.to_numpy(), shap_sample, cfg=cfg))
        contributions = shap_contributions_frame(
            shap_values.to_numpy(), list(shap_sample.columns), row=0, top_n=15
        )
        _add(viz.plot_shap_contributions(contributions, cfg))

    _add(viz.plot_lime_explanation(_read("lime_example", cfg), cfg))

    # ---- performance --------------------------------------------------------
    _add(viz.plot_features_vs_performance(_read("validation_curves", cfg), cfg))
    # One row per (model, method) at the default tolerance, plus the full baselines.
    test_table = _read("final_test_metrics", cfg)
    if test_table is not None and "epsilon" in test_table.columns:
        keep = (test_table["method"] == "Full") | (
            np.abs(test_table["epsilon"].fillna(-1.0) - 0.02) < 1e-12
        )
        _add(viz.plot_f1_comparison(test_table[keep], cfg,
                                    title_note=" (epsilon = 0.02)"))
    else:
        _add(viz.plot_f1_comparison(test_table, cfg))
    _add(viz.plot_roc_curves(_read("roc_curves", cfg), cfg))
    _add(viz.plot_confusion_matrices(_read("confusion_matrices", cfg), cfg))
    _add(viz.plot_model_cost(_read("final_test_metrics", cfg), cfg))

    # ---- stability ----------------------------------------------------------
    stability = _read("stability", cfg)
    _add(viz.plot_stability(stability, cfg))
    _add(viz.plot_stability_bars(stability, k=10, cfg=cfg))

    # ---- adaptive selection -------------------------------------------------
    adaptive = _read("adaptive_selection", cfg)
    curves = _read("validation_curves", cfg)
    if adaptive is not None and curves is not None:
        primary = "RandomForest" if "RandomForest" in set(curves["model"]) else \
            str(curves["model"].iloc[0])
        for method in adaptive["method"].unique():
            _add(viz.plot_adaptive_selection(curves, adaptive, primary, method, cfg))
        _add(viz.plot_adaptive_summary(adaptive, cfg))
        _add(viz.plot_reduction_vs_drop(adaptive, cfg))

    print(f"\n  {len(produced)} figures written to {FIG_DIR}")
    return produced


# ======================================================================================
# CLI
# ======================================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight and explainable phishing detection with adaptive feature selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py --stage all\n"
            "  python main.py --stage all --quick\n"
            "  python main.py --stage experiments --epsilons 0.01 0.02 0.05\n"
        ),
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["download", "verify", "preprocess", "baselines",
                 "feature_selection", "adaptive", "experiments", "visualize", "all"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke test; results are NOT reportable")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--models", type=str, default=None,
                        help="comma-separated subset of RandomForest,XGBoost,LightGBM,KNN")
    parser.add_argument("--no-lightgbm", action="store_true")
    parser.add_argument("--no-tuning", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--interactions", action="store_true",
                        help="also compute bounded SHAP interaction pairs (slower)")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(fast_mode=bool(args.quick), seed=int(args.seed))
    cfg.epsilons = [float(e) for e in args.epsilons]
    if args.no_lightgbm:
        cfg.include_lightgbm = False
        cfg.model_names = [m for m in cfg.model_names if m != "LightGBM"]
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        cfg.model_names = [m for m in cfg.model_names if m in wanted]
        cfg.include_lightgbm = "LightGBM" in cfg.model_names
    if args.no_tuning:
        cfg.tune_hyperparameters = False
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)

    if args.stage == "download":
        stage_download(force=args.force_download)
        return 0
    if args.stage == "verify":
        stage_verify()
        return 0
    if args.stage == "preprocess":
        run_preprocessing_pipeline(cfg)
        print(f"  artefacts written to {METRICS_DIR}")
        return 0
    if args.stage == "visualize":
        stage_visualize(cfg)
        return 0
    if args.stage == "all":
        stage_experiments(cfg, interactions=args.interactions)
        stage_visualize(cfg)
        return 0

    # Remaining stages need the dataset and a split.
    if not RAW_DATASET.exists():
        print("Dataset missing; run `python main.py --stage download` first.")
        return 1

    if args.stage == "baselines":
        splits, _, _ = run_preprocessing_pipeline(cfg)
        tuned = tune_all_models(splits.X_train, splits.y_train, cfg)
        save_json(tuned, METRICS_DIR / f"tuning{cfg.suffix}.json")
        baseline = _full_feature_baseline(cfg, splits, tuned)
        baseline.to_csv(METRICS_DIR / f"full_feature_baseline{cfg.suffix}.csv", index=False)
        print(format_metrics_table(baseline))
        return 0

    if args.stage == "feature_selection":
        splits, _, _ = run_preprocessing_pipeline(cfg)
        ctx = SelectionContext(X_train=splits.X_train, y_train=splits.y_train, cfg=cfg,
                               X_val=splits.X_val, y_val=splits.y_val)
        rankings = run_all_selectors(ctx)
        rankings.update(_sla_variants(rankings, k=SLAFS_REFERENCE_K))
        save_combined_rankings(
            rankings,
            RANKINGS_DIR / f"feature_rankings{cfg.suffix}.csv",
            RANKINGS_DIR / f"feature_selection_summary{cfg.suffix}.csv",
        )
        print(rankings_summary_table(rankings)[["method", "selection_time_s", "provenance"]]
              .to_string(index=False))
        return 0

    if args.stage == "adaptive":
        print("The adaptive stage needs the tuning results and rankings.")
        print("Run `python main.py --stage experiments` to produce them in one process.")
        return 1

    if args.stage == "experiments":
        stage_experiments(cfg, interactions=args.interactions)
        return 0

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
