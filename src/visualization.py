"""
visualization.py
================
Publication-quality figures. Every function is a thin, defensively-written
wrapper around Matplotlib/Seaborn that

* applies one shared academic style (serif type, no top/right spines, light grid,
  300 dpi, PDF + PNG export),
* reads only plain DataFrames/arrays persisted by the experiment stage, so
  ``python main.py --stage visualize`` can regenerate every figure without
  refitting a model,
* returns the PNG path (or ``None`` when the input table is empty, so a missing
  artefact never crashes the run).

No figure in this module contains hard-coded numbers: if a table is empty the
figure is skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import ExperimentConfig, FIG_DIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# --------------------------------------------------------------------------------------
# Shared style
# --------------------------------------------------------------------------------------
METHOD_ORDER: List[str] = [
    "SHAP",
    "LIME",
    "Permutation",
    "MutualInfo",
    "RFE",
    "RFImportance",
    "SLA-FS_rankavg",
    "SLA-FS_scoreblend",
    "SLA-FS_intersection",
]

MODEL_ORDER: List[str] = ["RandomForest", "XGBoost", "LightGBM", "KNN"]

COLORS: Dict[str, str] = {
    "SHAP": "#1b6ca8",
    "LIME": "#e07b39",
    "Permutation": "#2e8b57",
    "MutualInfo": "#8a4b9c",
    "RFE": "#b3312c",
    "RFImportance": "#4c5a67",
    "SLA-FS_rankavg": "#0f9b9b",
    "SLA-FS_scoreblend": "#c9a227",
    "SLA-FS_intersection": "#7a5195",
    "Full": "#111111",
    "RandomForest": "#1b6ca8",
    "XGBoost": "#e07b39",
    "LightGBM": "#2e8b57",
    "KNN": "#8a4b9c",
}

PROVENANCE_MARKERS: Dict[str, str] = {
    "ESTABLISHED": "o",
    "REPRODUCTION": "s",
    "MODIFICATION": "^",
    "PROPOSED": "D",
}


def apply_academic_style() -> None:
    """Serif typography and print-friendly defaults for every figure."""
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
        }
    )
    sns.set_palette("colorblind")


apply_academic_style()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _save(fig, name: str, cfg: Optional[ExperimentConfig] = None) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = getattr(cfg, "suffix", "") if cfg is not None else ""
    stem = f"{name}{suffix}"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(png_path, dpi=getattr(cfg, "figure_dpi", 300) or 300, bbox_inches="tight")
    if cfg is None or getattr(cfg, "save_pdf", False):
        fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path


def _ordered(values: Iterable[str], preferred: Sequence[str]) -> List[str]:
    present = [v for v in preferred if v in set(values)]
    extras = sorted(set(values) - set(present))
    return present + extras


def _color(key: str) -> str:
    if key in COLORS:
        return COLORS[key]
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return palette[abs(hash(key)) % len(palette)]


def _short(feature: str, limit: int = 30) -> str:
    return feature if len(feature) <= limit else feature[: limit - 1] + "\u2026"


# --------------------------------------------------------------------------------------
# 1. Feature importance
# --------------------------------------------------------------------------------------
def plot_feature_importance(
    importance_long: pd.DataFrame,
    top_n: int = 15,
    cfg: Optional[ExperimentConfig] = None,
    methods: Optional[Sequence[str]] = None,
    filename: str = "feature_importance_by_method",
) -> Optional[Path]:
    """Grid of horizontal bars, one panel per selector.

    ``importance_long`` needs ``feature``, ``method`` and ``score`` columns.
    Scores are min-max normalised *within* each method so panels are visually
    comparable despite different units (SHAP values vs Gini impurity vs MI).

    Ordering uses the **signed** score (higher = more important) because not all
    selectors emit non-negative numbers: ``SLA-FS_rankavg`` stores ``-mean rank``
    and permutation importance can be negative. Sorting by ``|score|`` would
    invert those panels.
    """
    if importance_long is None or importance_long.empty:
        return None
    frame = importance_long.copy()
    present = _ordered(frame["method"].unique(), methods or METHOD_ORDER)
    n_panels = len(present)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.9 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, method in zip(axes, present):
        subset = frame[frame["method"] == method].copy()
        if subset.empty:
            ax.axis("off")
            continue
        subset = subset.dropna(subset=["score"])
        if subset.empty:
            ax.axis("off")
            continue
        top = subset.nlargest(top_n, "score").iloc[::-1]
        values = top["score"].to_numpy(dtype=float)
        span = values.max() - values.min()
        normalised = (values - values.min()) / span if span > 0 else np.full_like(values, 1.0)
        ax.barh([_short(f, 26) for f in top["feature"]], normalised,
                color=_color(method), edgecolor="white", linewidth=0.4)
        ax.set_title(method)
        ax.set_xlabel("normalised importance")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x")

    for ax in axes[n_panels:]:
        ax.axis("off")
    fig.suptitle(f"Top-{top_n} features per selection method", y=1.005, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, filename, cfg)


def plot_rank_heatmap(
    ranks: pd.DataFrame,
    top_n: int = 20,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "feature_rank_heatmap",
) -> Optional[Path]:
    """Heatmap of each feature's rank under every method (1 = best)."""
    if ranks is None or ranks.empty:
        return None
    view = ranks.copy()
    view["_best"] = view.min(axis=1)
    view = view.nsmallest(top_n, "_best").drop(columns="_best")
    columns = _ordered(view.columns, METHOD_ORDER)

    fig, ax = plt.subplots(figsize=(0.55 * len(columns) + 3.0, 0.28 * len(view) + 1.8))
    sns.heatmap(
        view[columns], ax=ax, cmap="viridis_r", annot=True, fmt=".0f",
        linewidths=0.4, linecolor="white", cbar_kws={"label": "rank (1 = most important)"},
        annot_kws={"size": 6},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Feature rank by method (top {len(view)} features)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, filename, cfg)


def plot_rank_correlation(
    correlation: pd.DataFrame,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "rank_correlation_spearman",
) -> Optional[Path]:
    """Spearman agreement between the full rankings of the methods."""
    if correlation is None or correlation.empty:
        return None
    columns = _ordered(correlation.columns, METHOD_ORDER)
    view = correlation.loc[columns, columns]

    fig, ax = plt.subplots(figsize=(0.6 * len(columns) + 3.0, 0.55 * len(columns) + 2.2))
    sns.heatmap(
        view, ax=ax, cmap="RdYlBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f",
        linewidths=0.5, linecolor="white", square=True,
        cbar_kws={"label": "Spearman rho"},
        annot_kws={"size": 7},
    )
    ax.set_title("Agreement between feature rankings")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    return _save(fig, filename, cfg)


# --------------------------------------------------------------------------------------
# 2. SHAP and LIME
# --------------------------------------------------------------------------------------
def plot_shap_summary(
    values: np.ndarray,
    X_sample: pd.DataFrame,
    top_n: int = 15,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "shap_summary_beeswarm",
) -> Optional[Path]:
    """Beeswarm of SHAP values for the most important features.

    Implemented directly in Matplotlib (rather than ``shap.summary_plot``) so the
    figure is reproducible across shap versions and matches the paper style.
    """
    if values is None or X_sample is None or len(X_sample) == 0:
        return None
    features = list(X_sample.columns)
    importance = np.abs(np.asarray(values)).mean(axis=0)
    order = np.argsort(importance)[::-1][:top_n][::-1]

    fig, ax = plt.subplots(figsize=(6.2, 0.34 * len(order) + 1.9))
    rng = np.random.RandomState(0)
    for row, index in enumerate(order):
        shap_column = np.asarray(values)[:, index]
        feature_values = np.asarray(X_sample.iloc[:, index], dtype=float)
        # Each row is colour-scaled on its own 5th-95th percentile range, as in the
        # standard SHAP beeswarm. Pooling raw values would let one high-magnitude
        # feature (e.g. google_index in the 1e7 range) flatten every other row.
        low = np.nanpercentile(feature_values, 5)
        high = np.nanpercentile(feature_values, 95)
        if not np.isfinite(high) or high - low <= 0:
            scaled = np.full_like(feature_values, 0.5)
        else:
            scaled = np.clip((feature_values - low) / (high - low), 0.0, 1.0)
        jitter = rng.uniform(-0.28, 0.28, size=len(shap_column))
        ax.scatter(
            shap_column, row + jitter, c=scaled, cmap="coolwarm", vmin=0.0, vmax=1.0,
            s=7, alpha=0.75, linewidths=0, rasterized=True,
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_short(features[i], 28) for i in order], fontsize=7)
    ax.axvline(0.0, color="#444444", linewidth=0.7, linestyle="--")
    ax.set_xlabel("SHAP value (impact on phishing probability)")
    ax.set_title(f"SHAP summary: top {len(order)} features")
    ax.grid(axis="x")

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=matplotlib.colors.Normalize(0.0, 1.0))
    sm.set_array(np.array([0.0, 1.0]))
    colorbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    colorbar.set_label("feature value (low \u2192 high, scaled per feature)", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    return _save(fig, filename, cfg)


def plot_shap_contributions(
    contributions: pd.DataFrame,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "shap_instance_contributions",
) -> Optional[Path]:
    """Per-instance SHAP contributions (waterfall-style, signed)."""
    if contributions is None or contributions.empty:
        return None
    view = contributions.iloc[::-1]
    colors = ["#b3312c" if v > 0 else "#1b6ca8" for v in view["shap_value"]]

    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(view) + 1.6))
    ax.barh([_short(f, 28) for f in view["feature"]], view["shap_value"],
            color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.set_xlabel("SHAP contribution (red = pushes towards phishing)")
    ax.set_title("Single-instance explanation")
    ax.tick_params(axis="y", labelsize=7)
    return _save(fig, filename, cfg)


def plot_lime_explanation(
    explanation: pd.DataFrame,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "lime_instance_explanation",
) -> Optional[Path]:
    """Local LIME weights for one prediction."""
    if explanation is None or explanation.empty:
        return None
    view = explanation.sort_values("weight").reset_index(drop=True)
    colors = ["#b3312c" if w > 0 else "#1b6ca8" for w in view["weight"]]

    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(view) + 1.6))
    ax.barh([_short(f, 28) for f in view["feature"]], view["weight"],
            color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.set_xlabel("LIME local weight (red = supports phishing)")
    ax.set_title("LIME explanation of one prediction")
    ax.tick_params(axis="y", labelsize=7)
    return _save(fig, filename, cfg)


# --------------------------------------------------------------------------------------
# 3. Classification performance
# --------------------------------------------------------------------------------------
def plot_confusion_matrices(
    table: pd.DataFrame,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "confusion_matrices",
    max_panels: int = 9,
) -> Optional[Path]:
    """Small-multiple confusion matrices (counts + row-normalised shading)."""
    if table is None or table.empty:
        return None
    view = table.head(max_panels)
    ncols = 3
    nrows = int(np.ceil(len(view) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, view.iterrows()):
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalised = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
        ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if normalised[i, j] > 0.5 else "#222222")
        ax.set_xticks([0, 1], ["legit", "phish"])
        ax.set_yticks([0, 1], ["legit", "phish"])
        ax.set_xlabel("predicted", fontsize=7)
        ax.set_ylabel("actual", fontsize=7)
        label = f"{row.get('model', '')}"
        if pd.notna(row.get("method", np.nan)):
            label += f"\n{row['method']}"
        if pd.notna(row.get("n_features", np.nan)):
            label += f" (k={int(row['n_features'])})"
        ax.set_title(label, fontsize=8)
        ax.grid(False)

    for ax in axes[len(view):]:
        ax.axis("off")
    fig.suptitle("Confusion matrices on the held-out test split", y=1.0, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, filename, cfg)


def plot_roc_curves(
    roc: pd.DataFrame,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "roc_curves",
    max_curves: int = 12,
) -> Optional[Path]:
    """ROC curves; ``roc`` needs ``fpr``, ``tpr`` and a ``label`` column."""
    if roc is None or roc.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    labels = list(roc["label"].unique())[:max_curves]
    for label in labels:
        subset = roc[roc["label"] == label]
        auc = float(subset["auc"].iloc[0]) if "auc" in subset else float("nan")
        ax.plot(subset["fpr"], subset["tpr"], label=f"{label} (AUC={auc:.3f})", linewidth=1.3)
    ax.plot([0, 1], [0, 1], linestyle=":", color="#666666", linewidth=1.0, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves on the held-out test split")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right", fontsize=6.5)
    return _save(fig, filename, cfg)


def plot_f1_comparison(
    metrics: pd.DataFrame,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "f1_comparison_models_methods",
    title_note: str = "",
) -> Optional[Path]:
    """Grouped bars: F1 per selection method, grouped by model.

    A (model, method) pair can legitimately appear more than once in the test
    table (one row per epsilon). Duplicates are collapsed with ``max`` so the
    chart shows the best configuration of that pair; the table remains the
    authoritative record of every row.
    """
    if metrics is None or metrics.empty:
        return None
    frame = metrics.dropna(subset=["f1"]).copy()
    methods = _ordered(frame["method"].unique(), METHOD_ORDER)
    models = _ordered(frame["model"].unique(), MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(max(6.5, 0.85 * len(methods) + 2.5), 3.8))
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(methods))
    for offset, model in enumerate(models):
        subset = frame[frame["model"] == model].groupby("method")["f1"].max()
        values = [float(subset.get(m, np.nan)) for m in methods]
        ax.bar(x + offset * width - 0.4 + width / 2, values, width,
               label=model, color=_color(model), edgecolor="white", linewidth=0.4)

    ax.set_xticks(x, [_short(m, 18) for m in methods], rotation=20, ha="right")
    ax.set_ylabel("F1-score (phishing class)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("F1-score by selection method and classifier" + title_note)
    ax.legend(ncol=len(models), loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=7)
    ax.grid(axis="y")
    return _save(fig, filename, cfg)


def plot_features_vs_performance(
    curves: pd.DataFrame,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "features_vs_performance",
) -> Optional[Path]:
    """F1 versus number of retained features, faceted by classifier."""
    if curves is None or curves.empty:
        return None
    models = _ordered(curves["model"].unique(), MODEL_ORDER)
    ncols = 2
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols / 1.4, 3.3 * nrows),
                             sharey=True, squeeze=False)
    axes = axes.ravel()

    for ax, model in zip(axes, models):
        subset = curves[curves["model"] == model]
        for method in _ordered(subset["method"].unique(), METHOD_ORDER):
            line = subset[subset["method"] == method].sort_values("n_features")
            ax.plot(line["n_features"], line["f1"], marker="o", markersize=2.5,
                    label=method, color=_color(method), linewidth=1.2)
        ax.set_title(model)
        ax.set_xlabel("number of retained features (k)")
        ax.set_ylabel("validation F1")
        ax.set_ylim(min(0.5, subset["f1"].min() - 0.02), 1.005)
        ax.grid(True)

    for ax in axes[len(models):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Performance versus feature count (validation split)", y=1.0,
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, filename, cfg)


# --------------------------------------------------------------------------------------
# 4. Cost and stability
# --------------------------------------------------------------------------------------
def plot_execution_times(
    summary: pd.DataFrame,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "feature_selection_times",
) -> Optional[Path]:
    """Wall-clock cost of each selector (measured, not estimated)."""
    if summary is None or summary.empty or "selection_time_s" not in summary:
        return None
    frame = summary.dropna(subset=["selection_time_s"]).copy()
    if frame.empty:
        return None
    frame["_order"] = frame["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    frame = frame.sort_values("_order")

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    bars = ax.bar([_short(m, 18) for m in frame["method"]], frame["selection_time_s"],
                  color=[_color(m) for m in frame["method"]], edgecolor="white", linewidth=0.4)
    for bar, value in zip(bars, frame["selection_time_s"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}s",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("selection time (s)")
    ax.set_title("Cost of each feature-selection method")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y")
    return _save(fig, filename, cfg)


def plot_model_cost(
    table: pd.DataFrame,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "model_training_inference_cost",
) -> Optional[Path]:
    """Training and inference cost per classifier (full vs reduced feature set)."""
    if table is None or table.empty:
        return None
    frame = table.copy()
    required = {"model", "train_time_s", "inference_time_per_sample_ms"}
    if not required.issubset(frame.columns):
        return None

    grouped = (
        frame.groupby("model")[["train_time_s", "inference_time_per_sample_ms"]]
        .median()
        .reindex(_ordered(frame["model"].unique(), MODEL_ORDER))
        .dropna(how="all")
    )
    if grouped.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, column, label in zip(
        axes,
        ["train_time_s", "inference_time_per_sample_ms"],
        ["training time (s)", "inference time (ms/sample)"],
    ):
        ax.bar(grouped.index, grouped[column],
               color=[_color(m) for m in grouped.index], edgecolor="white", linewidth=0.4)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.suptitle("Computational cost per classifier (median over evaluated subsets)",
                 y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, filename, cfg)


def plot_stability(
    stability: pd.DataFrame,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "selection_stability_curve",
) -> Optional[Path]:
    """Jaccard stability of the top-k set as k grows."""
    if stability is None or stability.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for method in _ordered(stability["method"].unique(), METHOD_ORDER):
        subset = stability[stability["method"] == method].sort_values("k")
        if subset.empty:
            continue
        ax.plot(subset["k"], subset["mean_jaccard"], marker="o", markersize=3,
                label=method, color=_color(method), linewidth=1.3)
        ax.fill_between(subset["k"],
                        subset["mean_jaccard"] - subset["std_jaccard"].fillna(0),
                        subset["mean_jaccard"] + subset["std_jaccard"].fillna(0),
                        color=_color(method), alpha=0.10, linewidth=0)
    ax.set_xlabel("subset size k")
    ax.set_ylabel("mean pairwise Jaccard similarity")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Feature-selection stability across repeated subsamples")
    ax.legend(fontsize=6.5, ncol=2)
    return _save(fig, filename, cfg)


def plot_stability_bars(
    stability: pd.DataFrame,
    k: int = 10,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "selection_stability_bars",
) -> Optional[Path]:
    """Stability bars at one headline subset size."""
    if stability is None or stability.empty:
        return None
    subset = stability[stability["k"] == k]
    if subset.empty:
        return None
    subset = subset.assign(_order=subset["method"].map(
        {m: i for i, m in enumerate(METHOD_ORDER)})).sort_values("_order")

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    errors = subset["std_jaccard"].fillna(0).to_numpy()
    ax.bar([_short(m, 18) for m in subset["method"]], subset["mean_jaccard"],
           yerr=errors, capsize=3, color=[_color(m) for m in subset["method"]],
           edgecolor="white", linewidth=0.4)
    ax.set_ylabel("mean Jaccard similarity")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Stability of the top-{k} feature set")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y")
    return _save(fig, filename, cfg)


# --------------------------------------------------------------------------------------
# 5. Adaptive selection
# --------------------------------------------------------------------------------------
def plot_adaptive_selection(
    curve: pd.DataFrame,
    adaptive: pd.DataFrame,
    model: str,
    method: str,
    cfg: Optional[ExperimentConfig] = None,
    filename: Optional[str] = None,
) -> Optional[Path]:
    """One panel per (model, method): the F1(k) curve and the chosen k per epsilon."""
    if curve is None or curve.empty:
        return None
    subset = curve[(curve["model"] == model) & (curve["method"] == method)].sort_values("n_features")
    if subset.empty:
        return None
    picks = adaptive[(adaptive["model"] == model) & (adaptive["method"] == method)]

    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.plot(subset["n_features"], subset["f1"], marker="o", markersize=3,
            color=_color(method), linewidth=1.4, label="validation F1(k)")

    if not picks.empty:
        f1_full = float(picks["f1_full"].iloc[0])
        ax.axhline(f1_full, color="#111111", linestyle="--", linewidth=0.9,
                   label=f"full-feature F1 = {f1_full:.4f}")
        shades = ["#cccccc", "#999999", "#666666"]
        for index, (_, row) in enumerate(picks.sort_values("epsilon").iterrows()):
            ax.axhline(f1_full - row["epsilon"], color=shades[index % len(shades)],
                       linestyle=":", linewidth=0.9,
                       label=f"tolerance epsilon={row['epsilon']:.2f}")
            ax.axvline(row["selected_k"], color=shades[index % len(shades)],
                       linestyle="-.", linewidth=0.9)
            ax.annotate(f"k={int(row['selected_k'])}",
                        xy=(row["selected_k"], f1_full - row["epsilon"]),
                        xytext=(4, -10), textcoords="offset points", fontsize=7)

    ax.set_xlabel("number of retained features (k)")
    ax.set_ylabel("validation F1")
    ax.set_title(f"Adaptive selection: {method} on {model}")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    return _save(fig, filename or f"adaptive_curve_{model}_{method}", cfg)


def plot_adaptive_summary(
    adaptive: pd.DataFrame,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "adaptive_summary",
) -> Optional[Path]:
    """Selected k and achieved reduction, one panel per tolerance epsilon."""
    if adaptive is None or adaptive.empty:
        return None
    epsilons = sorted(adaptive["epsilon"].unique())
    models = _ordered(adaptive["model"].unique(), MODEL_ORDER)

    fig, axes = plt.subplots(1, len(epsilons), figsize=(4.3 * len(epsilons), 3.4),
                             sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, epsilon in zip(axes, epsilons):
        slab = adaptive[adaptive["epsilon"] == epsilon]
        methods = _ordered(slab["method"].unique(), METHOD_ORDER)
        width = 0.8 / max(len(models), 1)
        x = np.arange(len(methods))
        for offset, model in enumerate(models):
            line = slab[slab["model"] == model].set_index("method")
            values = [line["selected_k"].get(m, np.nan) for m in methods]
            ax.bar(x + offset * width - 0.4 + width / 2, values, width,
                   label=model, color=_color(model), edgecolor="white", linewidth=0.4)
        ax.set_xticks(x, [_short(m, 16) for m in methods], rotation=25, ha="right")
        ax.set_title(f"epsilon = {epsilon:.2f}")
        ax.set_ylabel("selected k")
        ax.grid(axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models), fontsize=7,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Adaptive feature selection: subset size per tolerance", y=1.02,
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, filename, cfg)


def plot_reduction_vs_drop(
    adaptive: pd.DataFrame,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "adaptive_reduction_vs_cost",
) -> Optional[Path]:
    """Feature reduction (%) against the F1 actually lost, per method and model."""
    if adaptive is None or adaptive.empty:
        return None
    frame = adaptive.dropna(subset=["reduction_pct", "f1_drop"])
    if frame.empty:
        return None

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for method in _ordered(frame["method"].unique(), METHOD_ORDER):
        subset = frame[frame["method"] == method]
        ax.scatter(subset["reduction_pct"], subset["f1_drop"], s=26,
                   color=_color(method), label=method, alpha=0.85,
                   marker=PROVENANCE_MARKERS.get(subset["provenance"].iloc[0], "o")
                   if "provenance" in subset else "o")
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=0.8)
    ax.set_xlabel("feature reduction (%)")
    ax.set_ylabel("F1 change versus full feature set")
    ax.set_title("Reduction versus performance change")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(True)
    return _save(fig, filename, cfg)


# --------------------------------------------------------------------------------------
# 6. Dataset description
# --------------------------------------------------------------------------------------
def plot_class_distribution(
    distribution: pd.DataFrame,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "class_distribution",
) -> Optional[Path]:
    """Bar chart of the class balance."""
    if distribution is None or distribution.empty or "n" not in distribution:
        return None
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    bars = ax.bar(distribution["class"], distribution["n"],
                  color=["#1b6ca8", "#b3312c"][: len(distribution)],
                  edgecolor="white", linewidth=0.5)
    total = int(distribution["n"].sum())
    for bar, value in zip(bars, distribution["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value,
                f"{int(value):,}\n({100 * value / total:.1f}%)",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("number of samples")
    ax.set_title("Class distribution")
    ax.set_ylim(0, distribution["n"].max() * 1.18)
    ax.grid(axis="y")
    return _save(fig, filename, cfg)


def plot_target_correlation(
    target: pd.DataFrame,
    top_n: int = 25,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "target_correlation",
) -> Optional[Path]:
    """Strongest linear associations with the phishing label (train split only)."""
    if target is None or target.empty or "abs_correlation" not in target:
        return None
    view = target.head(top_n).iloc[::-1]
    colors = ["#b3312c" if v > 0 else "#1b6ca8" for v in view["correlation_with_target"]]

    fig, ax = plt.subplots(figsize=(5.4, 0.28 * len(view) + 1.6))
    ax.barh([_short(f, 28) for f in view["feature"]], view["correlation_with_target"],
            color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.set_xlabel("point-biserial correlation with the phishing label")
    ax.set_title(f"Top {len(view)} features by target correlation (training split)")
    ax.tick_params(axis="y", labelsize=7)
    return _save(fig, filename, cfg)


def plot_correlation_structure(
    correlation: pd.DataFrame,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "feature_correlation_structure",
) -> Optional[Path]:
    """Full train-split feature correlation matrix (no annotations)."""
    if correlation is None or correlation.empty:
        return None
    size = min(9.0, max(5.0, 0.09 * len(correlation)))
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    sns.heatmap(correlation, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1,
                cbar_kws={"label": "Pearson r", "shrink": 0.7},
                xticklabels=False, yticklabels=False, rasterized=True)
    ax.set_title("Feature correlation structure (training split)")
    ax.set_xlabel("87 candidate features")
    ax.set_ylabel("87 candidate features")
    ax.grid(False)
    return _save(fig, filename, cfg)


def plot_correlated_pairs(
    pairs: pd.DataFrame,
    cfg: Optional[ExperimentConfig] = None,
    filename: str = "highly_correlated_pairs",
    top_n: int = 20,
) -> Optional[Path]:
    """Redundant feature pairs above the reporting threshold."""
    if pairs is None or pairs.empty:
        return None
    view = pairs.head(top_n).iloc[::-1]
    labels = [f"{_short(a, 18)} \u2013 {_short(b, 18)}"
              for a, b in zip(view["feature_a"], view["feature_b"])]

    fig, ax = plt.subplots(figsize=(5.6, 0.3 * len(view) + 1.5))
    ax.barh(labels, view["abs_r"], color="#4c5a67", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("|Pearson r|")
    ax.set_title(f"{len(pairs)} feature pairs with |r| above threshold")
    ax.tick_params(axis="y", labelsize=7)
    return _save(fig, filename, cfg)
