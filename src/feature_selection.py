"""
feature_selection.py
====================
The six feature-ranking techniques required by the study, behind one uniform
interface:

    SHAP | LIME | Permutation | MutualInfo | RFE | RFImportance

Every selector returns a :class:`FeatureRanking`, i.e. *the complete ordering of
all features* plus the raw importance score behind that ordering. Returning the
full ordering (rather than a single subset) is what makes the adaptive
selection stage possible: the same ranking can be truncated at any ``k``.

Fair-comparison rules
---------------------
* **One surrogate model.** SHAP, LIME, permutation importance and Gini
  importance all inspect the *same* fitted Random Forest
  (``SURROGATE_TREES`` trees, seed = ``cfg.seed``), so differences between the
  rankings come from the selector, not from the model.
* **Training data only.** No selector in this module ever receives the test
  split. Permutation importance is measured on held-out folds of the *training*
  split (``cfg.perm_cv_folds``), never on test.
* **Execution time is measured, not estimated.** ``elapsed_s`` is wall-clock
  time around the selector itself. ``RFImportance`` is essentially free once
  the surrogate exists; its cost is therefore reported separately (see
  ``SelectionContext.surrogate_fit_s``) so the comparison table is not
  misleading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from config import ExperimentConfig, set_global_seed
from evaluation import Timer, jaccard, stability_metrics

try:  # LIME is optional at runtime but required by the study
    import lime  # noqa: F401
    from lime.lime_tabular import LimeTabularExplainer

    HAS_LIME = True
except Exception:  # pragma: no cover
    HAS_LIME = False

SURROGATE_TREES = 300
POSITIVE_CLASS = 1


# --------------------------------------------------------------------------------------
# Ranking container
# --------------------------------------------------------------------------------------
@dataclass
class FeatureRanking:
    """An ordered list of features plus the scores that produced the order.

    Attributes
    ----------
    method : str
        Short method name (see ``SELECTORS``), used as a column/legend label.
    features : list of str
        Complete ordering, most important first.
    scores : dict
        Raw importance per feature (units depend on the method).
    elapsed_s : float
        Wall-clock seconds spent by the selector.
    provenance : str
        ``ESTABLISHED`` (literature method), ``REPRODUCTION`` or ``MODIFICATION``.
    """

    method: str
    features: List[str]
    scores: Dict[str, float]
    elapsed_s: float = 0.0
    provenance: str = "ESTABLISHED"
    description: str = ""
    notes: str = ""
    extras: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set = set()
        ordered: List[str] = []
        for feature in self.features:
            if feature not in seen:
                seen.add(feature)
                ordered.append(feature)
        self.features = ordered

    # -- basic accessors ------------------------------------------------------
    @property
    def n_features(self) -> int:
        return len(self.features)

    def top_k(self, k: int) -> List[str]:
        return self.features[: max(0, int(k))]

    def rank_series(self) -> pd.Series:
        """1-based rank per feature (1 = most important)."""
        return pd.Series(np.arange(1, len(self.features) + 1), index=self.features)

    def score_series(self) -> pd.Series:
        return pd.Series({f: float(self.scores.get(f, np.nan)) for f in self.features})

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_scores(
        cls,
        method: str,
        scores: Dict[str, float],
        **kwargs,
    ) -> "FeatureRanking":
        """Order features by descending score (ties broken alphabetically)."""
        clean = {k: float(v) for k, v in scores.items()}
        ordered = sorted(clean, key=lambda f: (-clean[f], f))
        return cls(method=method, features=ordered, scores=clean, **kwargs)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def subsample_frame(X: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    """Deterministic row subsample (used to bound SHAP/LIME cost)."""
    if n is None or n >= len(X):
        return X
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(X), size=int(n), replace=False))
    return X.iloc[idx]


def build_surrogate(cfg: ExperimentConfig, n_estimators: int = SURROGATE_TREES) -> RandomForestClassifier:
    """The single Random Forest inspected by SHAP / LIME / permutation / Gini."""
    set_global_seed(cfg.seed)
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
    )


@dataclass
class SelectionContext:
    """Everything a selector is allowed to see: training data + config.

    The validation split is carried for convenience (e.g. inspecting
    permutation importance out-of-sample) but the test split is never attached.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    cfg: ExperimentConfig
    X_val: Optional[pd.DataFrame] = None
    y_val: Optional[pd.Series] = None
    surrogate: Optional[RandomForestClassifier] = None
    surrogate_fit_s: float = 0.0

    @property
    def feature_names(self) -> List[str]:
        return list(self.X_train.columns)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def fitted_surrogate(self) -> RandomForestClassifier:
        """Fit the shared surrogate once, lazily, and time the fit."""
        if self.surrogate is None:
            with Timer() as timer:
                model = build_surrogate(self.cfg)
                model.fit(self.X_train, self.y_train)
            self.surrogate = model
            self.surrogate_fit_s = timer.elapsed
        return self.surrogate


# --------------------------------------------------------------------------------------
# 1. SHAP
# --------------------------------------------------------------------------------------
def rank_shap(ctx: SelectionContext) -> FeatureRanking:
    """Global feature importance = mean absolute SHAP value over the data.

    ``TreeExplainer`` is used because the surrogate is a tree ensemble, which
    makes exact Shapley values tractable. The importance of feature *j* is
    ``mean_i |phi_ij|`` over a random sample of ``cfg.shap_max_samples``
    training rows.
    """
    from .explainability import shap_values_for  # local import avoids a cycle

    model = ctx.fitted_surrogate()
    sample = subsample_frame(ctx.X_train, ctx.cfg.shap_max_samples, ctx.cfg.seed)

    with Timer() as timer:
        values = shap_values_for(model, sample)
    importance = np.abs(values).mean(axis=0)

    ranking = FeatureRanking.from_scores(
        "SHAP",
        dict(zip(ctx.feature_names, importance)),
        elapsed_s=timer.elapsed,
        provenance="ESTABLISHED",
        description="Mean |SHAP value| from TreeExplainer on the shared Random Forest surrogate",
        notes=f"n_explained={len(sample)}",
    )
    ranking.extras["sample_X"] = sample
    ranking.extras["shap_values"] = values
    return ranking


# --------------------------------------------------------------------------------------
# 2. LIME
# --------------------------------------------------------------------------------------
def _match_feature_label(label: str, names: List[str], name_set: set) -> Optional[str]:
    """LIME may decorate labels (``"x <= 0.5"``); recover the column name."""
    if label in name_set:
        return label
    best: Optional[str] = None
    for name in names:
        if label.startswith(name) and (best is None or len(name) > len(best)):
            best = name
    return best


def _lime_fallback_importance(
    model,
    X_train: pd.DataFrame,
    instances: pd.DataFrame,
    feature_names: List[str],
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Documented fallback: local linear surrogates on Gaussian perturbations.

    Used only if the ``lime`` package cannot be imported. It reproduces LIME's
    essential idea (sample the neighbourhood, weight by proximity, fit a sparse
    linear model, aggregate |weight|) without the package dependency.
    """
    rng = np.random.RandomState(seed)
    means = X_train.mean().to_numpy(dtype=float)
    stds = X_train.std().to_numpy(dtype=float)
    stds[stds == 0] = 1.0
    d = len(feature_names)
    accumulated = np.zeros(d)

    for i in range(len(instances)):
        x = instances.iloc[i].to_numpy(dtype=float)
        Z = means + rng.normal(size=(n_samples, d)) * stds
        Z[0] = x
        distance = (((Z - x) / stds) ** 2).sum(axis=1) / d
        weights = np.exp(-distance / 2.0)
        Z[0] = x
        with np.errstate(all="ignore"):
            proba = model.predict_proba(
                pd.DataFrame(Z, columns=list(feature_names))
            )[:, POSITIVE_CLASS]
        design = np.hstack([Z, np.ones((len(Z), 1))])
        weighted = design * weights[:, None]
        coef, *_ = np.linalg.lstsq(weighted, proba * weights, rcond=None)
        accumulated += np.abs(coef[:-1])

    return accumulated / max(len(instances), 1)


def rank_lime(ctx: SelectionContext) -> FeatureRanking:
    """Aggregated LIME: mean |local weight| over ``cfg.lime_num_samples`` rows.

    LIME is local, so a *global* ranking requires aggregation. Following the
    SLA-FS idea, the absolute weight each feature receives in each local linear
    surrogate is averaged across explained instances.
    """
    model = ctx.fitted_surrogate()
    names = ctx.feature_names
    name_set = set(names)
    n_instances = int(min(ctx.cfg.lime_num_samples, len(ctx.X_train)))
    instances = subsample_frame(ctx.X_train, n_instances, ctx.cfg.seed)

    notes = f"n_explained={len(instances)}, num_samples={ctx.cfg.lime_num_samples}"
    accumulated = np.zeros(len(names))

    with Timer() as timer:
        if HAS_LIME:
            explainer = LimeTabularExplainer(
                training_data=np.asarray(ctx.X_train, dtype=float),
                feature_names=names,
                class_names=["legitimate", "phishing"],
                mode="classification",
                discretize_continuous=False,
                random_state=ctx.cfg.seed,
            )

            def predict_fn(matrix: np.ndarray) -> np.ndarray:
                # LIME perturbs as a NumPy array; restore the column names so the
                # surrogate sees exactly the layout it was fitted on.
                return model.predict_proba(pd.DataFrame(matrix, columns=names))

            for i in range(len(instances)):
                row = np.asarray(instances.iloc[i], dtype=float)
                explanation = explainer.explain_instance(
                    row,
                    predict_fn,
                    num_features=len(names),
                    num_samples=ctx.cfg.lime_num_samples,
                    labels=(POSITIVE_CLASS,),
                )
                for label, weight in explanation.as_list(label=POSITIVE_CLASS):
                    feature = _match_feature_label(label, names, name_set)
                    if feature is not None:
                        accumulated[names.index(feature)] += abs(float(weight))
            accumulated /= max(len(instances), 1)
            method_note = notes
        else:  # pragma: no cover - exercised only without the lime package
            accumulated = _lime_fallback_importance(
                model, ctx.X_train, instances, names, ctx.cfg.lime_num_samples, ctx.cfg.seed
            )
            method_note = notes + ", used documented local-surrogate fallback (lime unavailable)"

    return FeatureRanking.from_scores(
        "LIME",
        dict(zip(names, accumulated)),
        elapsed_s=timer.elapsed,
        provenance="ESTABLISHED",
        description="Mean |LIME local weight| aggregated over explained training instances",
        notes=method_note,
    )


# --------------------------------------------------------------------------------------
# 3. Permutation importance
# --------------------------------------------------------------------------------------
def rank_permutation(ctx: SelectionContext) -> FeatureRanking:
    """Fold-averaged permutation importance (F1 drop when a column is shuffled).

    Measured on held-out folds of the training split so the importance reflects
    generalisation, not memorisation. Negative values are meaningful ("shuffling
    this feature helps") and are kept, which is why the ranking is built from the
    raw score rather than its absolute value.
    """
    cfg = ctx.cfg
    X, y = ctx.X_train, ctx.y_train
    names = ctx.feature_names
    folds = max(2, int(cfg.perm_cv_folds))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.seed)

    accumulated = np.zeros(len(names))
    completed = 0

    with Timer() as timer:
        for train_idx, holdout_idx in splitter.split(X, y):
            model = build_surrogate(cfg)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            result = permutation_importance(
                model,
                X.iloc[holdout_idx],
                y.iloc[holdout_idx],
                scoring="f1",
                n_repeats=cfg.perm_n_repeats,
                random_state=cfg.seed,
                n_jobs=cfg.n_jobs,
            )
            accumulated += result.importances_mean
            completed += 1
    accumulated /= max(completed, 1)

    return FeatureRanking.from_scores(
        "Permutation",
        dict(zip(names, accumulated)),
        elapsed_s=timer.elapsed,
        provenance="ESTABLISHED",
        description="Mean drop in F1 when a feature is permuted, averaged over training CV folds",
        notes=f"cv_folds={completed}, n_repeats={cfg.perm_n_repeats}, scoring=f1",
    )


# --------------------------------------------------------------------------------------
# 4. Mutual information
# --------------------------------------------------------------------------------------
def rank_mutual_information(ctx: SelectionContext) -> FeatureRanking:
    """Univariate mutual information between each feature and the label."""
    with Timer() as timer:
        scores = mutual_info_classif(
            ctx.X_train,
            ctx.y_train,
            random_state=ctx.cfg.seed,
            n_neighbors=ctx.cfg.mi_n_neighbors,
        )
    return FeatureRanking.from_scores(
        "MutualInfo",
        dict(zip(ctx.feature_names, scores)),
        elapsed_s=timer.elapsed,
        provenance="ESTABLISHED",
        description="Univariate mutual information with the label (k-NN estimator)",
        notes=f"n_neighbors={ctx.cfg.mi_n_neighbors}",
    )


# --------------------------------------------------------------------------------------
# 5. Recursive Feature Elimination
# --------------------------------------------------------------------------------------
def rank_rfe(ctx: SelectionContext) -> FeatureRanking:
    """RFE down to one feature; ``ranking_`` gives a complete ordering.

    The estimator inside RFE is a Random Forest, matching the surrogate family
    used by the other model-based selectors.
    """
    names = ctx.feature_names
    estimator = build_surrogate(ctx.cfg, n_estimators=200)

    with Timer() as timer:
        selector = RFE(estimator=estimator, n_features_to_select=1, step=int(ctx.cfg.rfe_step))
        selector.fit(ctx.X_train, ctx.y_train)

    elimination = np.asarray(selector.ranking_, dtype=float)
    # NOTE: after the last elimination round RFE refits its estimator on the
    # surviving feature(s) only, so ``estimator_.feature_importances_`` no longer
    # spans all p columns. Ties are therefore broken with the shared surrogate's
    # importances, which are defined over the full feature set.
    tie_break = np.asarray(ctx.fitted_surrogate().feature_importances_, dtype=float)
    importance = tie_break

    # Smallest `ranking_` value = survived longest = most important.
    order = sorted(range(len(names)), key=lambda i: (elimination[i], -importance[i]))
    features = [names[i] for i in order]
    scores = {names[i]: float(importance[i]) for i in order}

    return FeatureRanking(
        method="RFE",
        features=features,
        scores=scores,
        elapsed_s=timer.elapsed,
        provenance="ESTABLISHED",
        description="Recursive Feature Elimination with a Random Forest, down to 1 feature",
        notes=f"step={ctx.cfg.rfe_step}, n_elimination_rounds={int(elimination.max())}",
    )


# --------------------------------------------------------------------------------------
# 6. Random Forest (Gini) importance
# --------------------------------------------------------------------------------------
def rank_rf_importance(ctx: SelectionContext) -> FeatureRanking:
    """Mean decrease in impurity from the shared surrogate (no extra fitting)."""
    model = ctx.fitted_surrogate()
    with Timer() as timer:
        importance = np.asarray(model.feature_importances_, dtype=float)

    return FeatureRanking.from_scores(
        "RFImportance",
        dict(zip(ctx.feature_names, importance)),
        elapsed_s=timer.elapsed,
        provenance="ESTABLISHED",
        description="Mean decrease in Gini impurity from the shared Random Forest surrogate",
        notes="Importance is free once the surrogate is fitted; see surrogate_fit_s",
    )


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
SELECTORS: Dict[str, Callable[[SelectionContext], FeatureRanking]] = {
    "SHAP": rank_shap,
    "LIME": rank_lime,
    "Permutation": rank_permutation,
    "MutualInfo": rank_mutual_information,
    "RFE": rank_rfe,
    "RFImportance": rank_rf_importance,
}

SELECTOR_ORDER: List[str] = list(SELECTORS)


def run_all_selectors(
    ctx: SelectionContext,
    methods: Optional[Sequence[str]] = None,
    progress: bool = True,
) -> Dict[str, FeatureRanking]:
    """Run the requested selectors and return ``{method: FeatureRanking}``."""
    wanted = list(methods or SELECTOR_ORDER)
    rankings: Dict[str, FeatureRanking] = {}

    for index, name in enumerate(wanted, start=1):
        if progress:
            print(f"  [{index}/{len(wanted)}] {name} ...", end=" ", flush=True)
        try:
            ranking = SELECTORS[name](ctx)
            rankings[name] = ranking
            if progress:
                print(f"{ranking.elapsed_s:7.1f}s  (top3: {', '.join(ranking.top_k(3))})")
        except Exception as exc:  # a single selector failing must not abort the run
            if progress:
                print(f"FAILED: {type(exc).__name__}: {exc}")
            rankings[name] = FeatureRanking(
                method=name,
                features=[],
                scores={},
                elapsed_s=float("nan"),
                provenance="ESTABLISHED",
                description=f"Failed: {type(exc).__name__}: {exc}",
            )
    return rankings


# --------------------------------------------------------------------------------------
# Persistence and cross-method analysis
# --------------------------------------------------------------------------------------
def save_combined_rankings(
    rankings: Dict[str, FeatureRanking],
    path,
    score_path=None,
) -> pd.DataFrame:
    """Wide table: ``feature | rank_<method> ... | score_<method> ...``."""
    frame: Optional[pd.DataFrame] = None
    for name, ranking in rankings.items():
        if ranking.n_features == 0:
            continue
        part = pd.concat(
            [ranking.rank_series().rename(f"rank_{name}"),
             ranking.score_series().rename(f"score_{name}")],
            axis=1,
        )
        frame = part if frame is None else frame.join(part, how="outer")

    if frame is None:
        frame = pd.DataFrame(columns=["feature"])
    frame = frame.sort_values(
        sorted([c for c in frame.columns if c.startswith("rank_")])[0]
    ).reset_index().rename(columns={"index": "feature"})
    frame.to_csv(path, index=False)

    if score_path is not None:
        rankings_summary_table(rankings).to_csv(score_path, index=False)
    return frame


def rankings_summary_table(rankings: Dict[str, FeatureRanking]) -> pd.DataFrame:
    """One row per method: cost, provenance and the top-10 features."""
    rows = []
    for name, ranking in rankings.items():
        rows.append(
            {
                "method": name,
                "n_features_ranked": ranking.n_features,
                "selection_time_s": round(float(ranking.elapsed_s), 3),
                "provenance": ranking.provenance,
                "description": ranking.description,
                "notes": ranking.notes,
                "top_10_features": ";".join(ranking.top_k(10)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "method", key=lambda s: s.map({m: i for i, m in enumerate(SELECTOR_ORDER + [
            "SLA-FS_rankavg", "SLA-FS_scoreblend", "SLA-FS_intersection"
        ])})
    ).reset_index(drop=True)


def rank_correlation_matrix(rankings: Dict[str, FeatureRanking]) -> pd.DataFrame:
    """Spearman correlation between the rankings of different methods.

    A feature missing from a ranking is placed last, so correlations stay
    defined when a method selects a strict subset.
    """
    series = {}
    for name, ranking in rankings.items():
        if ranking.n_features:
            series[name] = ranking.rank_series()
    if not series:
        return pd.DataFrame()
    ranks = pd.concat(series, axis=1)
    ranks = ranks.fillna(ranks.max() + 1.0)
    return ranks.corr(method="spearman")


def stability_rankings(
    ctx: SelectionContext,
    methods: Optional[Sequence[str]] = None,
    cfg: Optional[ExperimentConfig] = None,
    progress: bool = True,
) -> Dict[str, List[List[str]]]:
    """Re-run each selector on random subsamples to measure ranking stability.

    Returns ``{method: [ranking_repeat_1, ranking_repeat_2, ...]}``. Stability is
    then summarised with Jaccard similarity of the top-k sets
    (:func:`src.evaluation.stability_metrics`).

    ``cfg`` lets the caller use a cheaper protocol (fewer LIME samples, coarser
    RFE step) for the repeats. That protocol is recorded in the run metadata so
    the stability numbers are never mistaken for the headline rankings.
    """
    cfg = cfg or ctx.cfg
    wanted = list(methods or SELECTOR_ORDER)
    repeats = max(2, int(cfg.stability_repeats))
    fraction = float(cfg.stability_fraction)
    rng = np.random.RandomState(cfg.seed)
    n = len(ctx.X_train)
    size = max(2, int(round(fraction * n)))

    collected: Dict[str, List[List[str]]] = {name: [] for name in wanted}

    for repeat in range(repeats):
        idx = np.sort(rng.choice(n, size=size, replace=False))
        subsample_ctx = SelectionContext(
            X_train=ctx.X_train.iloc[idx].reset_index(drop=True),
            y_train=ctx.y_train.iloc[idx].reset_index(drop=True),
            cfg=cfg,
        )
        if progress:
            print(f"  stability repeat {repeat + 1}/{repeats} (n={size})")
        for name in wanted:
            try:
                collected[name].append(SELECTORS[name](subsample_ctx).features)
            except Exception as exc:  # pragma: no cover
                if progress:
                    print(f"    [warn] {name} failed on repeat {repeat + 1}: {exc}")
                collected[name].append([])
    return collected


def compute_stability_table(
    stability: Dict[str, List[List[str]]],
    k_values: Sequence[int],
) -> pd.DataFrame:
    """Long table of Jaccard stability: one row per (method, k)."""
    rows = []
    for name, ranking_lists in stability.items():
        usable = [r for r in ranking_lists if r]
        for k in k_values:
            if any(k > len(r) for r in usable):
                continue
            summary = stability_metrics(usable, int(k))
            rows.append(
                {
                    "method": name,
                    "k": int(k),
                    "mean_jaccard": summary.get("mean_jaccard"),
                    "std_jaccard": summary.get("std_jaccard"),
                    "min_jaccard": summary.get("min_jaccard"),
                    "n_pairs": summary.get("n_pairs", 0),
                    "n_runs": len(usable),
                }
            )
    return pd.DataFrame(rows)


def pairwise_ranking_agreement(rankings: Dict[str, FeatureRanking], k: int) -> pd.DataFrame:
    """Jaccard overlap of the top-k sets between every pair of methods."""
    names = [n for n, r in rankings.items() if r.n_features]
    matrix = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            matrix.loc[a, b] = jaccard(rankings[a].top_k(k), rankings[b].top_k(k))
    return matrix
