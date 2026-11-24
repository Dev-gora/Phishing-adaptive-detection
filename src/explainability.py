"""
explainability.py
=================
Explainable-AI utilities and the **SLA-FS** feature-fusion stage.

What the reference paper does
-----------------------------
Shafin (2025) proposes *SLA-FS*: SHAP supplies a **global** view of feature
importance, LIME supplies **aggregated local** explanations, and the two are
fused into one feature ranking from which a compact subset is taken.

What the paper does *not* state
-------------------------------
The exact fusion arithmetic and the final ``k`` are not specified precisely
enough to be reproduced verbatim. Three variants are therefore implemented and
**explicitly labelled**:

======================  ============  ==================================================
Method                  Provenance    Fusion rule
======================  ============  ==================================================
SLA-FS_rankavg          MODIFICATION  mean of the two rank vectors, re-sorted
SLA-FS_scoreblend       MODIFICATION  min-max-normalised score blend (``alpha`` weight)
SLA-FS_intersection     REPRODUCTION  keep features in *both* top-k lists (paper wording)
======================  ============  ==================================================

``MODIFICATION`` means "inspired by the paper but our own operationalisation, not
verified against the paper's code". ``REPRODUCTION`` means "the literal
intersection rule the paper describes". This distinction is carried into every
results table via the ``provenance`` column and is restated in the README.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import ExperimentConfig
from .evaluation import Timer, jaccard
from .feature_selection import (
    POSITIVE_CLASS,
    FeatureRanking,
    _match_feature_label,
)

# --------------------------------------------------------------------------------------
# Provenance register
# --------------------------------------------------------------------------------------
METHOD_PROVENANCE: Dict[str, Dict[str, str]] = {
    "SHAP": {
        "provenance": "ESTABLISHED",
        "reference": "Lundberg & Lee (2017), NeurIPS 30, 4765-4774",
        "description": "Global importance = mean absolute SHAP value.",
        "implementation_note": "shap.TreeExplainer on the shared Random Forest surrogate.",
    },
    "LIME": {
        "provenance": "ESTABLISHED",
        "reference": "Ribeiro, Singh & Guestrin (2016), KDD '16, 1135-1144",
        "description": "Local linear surrogate; aggregated to a global score.",
        "implementation_note": "Mean |weight| over explained training instances.",
    },
    "Permutation": {
        "provenance": "ESTABLISHED",
        "reference": "Breiman (2001), Machine Learning 45(1), 5-32; Fisher, Rudin & Dominici (2019), JMLR 20(177), 1-81",
        "description": "Drop in F1 when a feature is randomly permuted.",
        "implementation_note": "Averaged over stratified folds of the training split.",
    },
    "MutualInfo": {
        "provenance": "ESTABLISHED",
        "reference": "Cover & Thomas (2006), Elements of Information Theory, 2nd ed.; Kraskov et al. (2004), Phys. Rev. E 69, 066138",
        "description": "Univariate mutual information with the label.",
        "implementation_note": "sklearn mutual_info_classif, k-NN estimator.",
    },
    "RFE": {
        "provenance": "ESTABLISHED",
        "reference": "Guyon, Weston, Barnhill & Vapnik (2002), Machine Learning 46, 389-422",
        "description": "Recursive elimination using Random Forest importances.",
        "implementation_note": "Elimination order gives a complete ranking.",
    },
    "RFImportance": {
        "provenance": "ESTABLISHED",
        "reference": "Breiman (2001), Machine Learning 45(1), 5-32",
        "description": "Mean decrease in Gini impurity.",
        "implementation_note": "Read directly from the shared surrogate.",
    },
    "SLA-FS_rankavg": {
        "provenance": "MODIFICATION",
        "reference": "Shafin (2025), Data Science and Management 8(2), 127-136, DOI 10.1016/j.dsm.2024.08.004",
        "description": "Fusion of SHAP and aggregated-LIME rankings by mean rank.",
        "implementation_note": "Our operationalisation: the paper does not give an explicit formula.",
    },
    "SLA-FS_scoreblend": {
        "provenance": "MODIFICATION",
        "reference": "Shafin (2025), DOI 10.1016/j.dsm.2024.08.004",
        "description": "Fusion by weighted blend of min-max-normalised importance scores.",
        "implementation_note": "Our operationalisation: alpha = 0.5 by default, configurable.",
    },
    "SLA-FS_intersection": {
        "provenance": "REPRODUCTION",
        "reference": "Shafin (2025), DOI 10.1016/j.dsm.2024.08.004",
        "description": "Intersection of the SHAP and LIME top-k lists, used as the selected subset.",
        "implementation_note": "Matches the paper's stated 'ranking/intersection' rule; k must be fixed.",
    },
}


def provenance_table() -> pd.DataFrame:
    """One row per method, for the paper's methodology table."""
    rows = [{"method": name, **meta} for name, meta in METHOD_PROVENANCE.items()]
    order = list(METHOD_PROVENANCE)
    frame = pd.DataFrame(rows)
    frame["_order"] = frame["method"].map({m: i for i, m in enumerate(order)})
    return frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# SHAP helpers
# --------------------------------------------------------------------------------------
def _normalise_shap(raw) -> np.ndarray:
    """Coerce every SHAP output flavour into ``(n_samples, n_features)``.

    Different shap versions return a list ``[class0, class1]``, a 3-D array
    ``(n, p, 2)``, or an ``Explanation`` object. All are reduced to the positive
    (phishing) class.
    """
    if hasattr(raw, "values"):
        raw = raw.values
    if isinstance(raw, list):
        array = np.asarray(raw[POSITIVE_CLASS] if len(raw) > POSITIVE_CLASS else raw[0], dtype=float)
    else:
        array = np.asarray(raw, dtype=float)

    if array.ndim == 3:
        array = array[:, :, POSITIVE_CLASS] if array.shape[2] > POSITIVE_CLASS else array[:, :, 0]
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def shap_values_for(model, X: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP values for the phishing class, shape ``(n, p)``."""
    import shap

    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    explainer = shap.TreeExplainer(model)
    try:
        raw = explainer.shap_values(frame, check_additivity=False)
    except TypeError:  # older/newer signature without the keyword
        raw = explainer.shap_values(frame)
    return _normalise_shap(raw)


def mean_abs_shap_table(values: np.ndarray, feature_names: Sequence[str]) -> pd.DataFrame:
    """Global ranking table: ``mean |SHAP|``, signed mean and rank."""
    importance = np.abs(values).mean(axis=0)
    signed = values.mean(axis=0)
    frame = pd.DataFrame(
        {
            "feature": list(feature_names),
            "mean_abs_shap": importance,
            "mean_shap": signed,
            "mean_feature_value": np.nan,
        }
    )
    frame = frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def shap_contributions_frame(
    values: np.ndarray,
    feature_names: Sequence[str],
    row: int = 0,
    top_n: int = 15,
) -> pd.DataFrame:
    """Per-instance contributions, sorted by magnitude (waterfall-style chart)."""
    contributions = np.asarray(values)[row]
    frame = pd.DataFrame({"feature": list(feature_names), "shap_value": contributions})
    frame["abs_shap"] = frame["shap_value"].abs()
    frame = frame.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def top_interaction_pairs(
    model,
    X_sample: pd.DataFrame,
    feature_names: Sequence[str],
    max_samples: int = 40,
    top_n: int = 15,
) -> pd.DataFrame:
    """Strongest pairwise SHAP interactions (optional, cost-bounded).

    TreeSHAP interaction values are O(p^2) per instance, so this is intentionally
    restricted to a small sample. Returns an empty frame if the calculation is not
    supported by the installed shap version -- interaction analysis is a
    supplementary view, not part of the main results.
    """
    columns = ["feature_a", "feature_b", "mean_abs_interaction"]
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        sample = X_sample.iloc[:max_samples]
        raw = explainer.shap_interaction_values(sample)
        matrix = np.asarray(raw[POSITIVE_CLASS] if isinstance(raw, list) else raw, dtype=float)
        if matrix.ndim == 4:
            matrix = matrix[:, :, :, POSITIVE_CLASS]
        mean_abs = np.abs(matrix).mean(axis=0)
        np.fill_diagonal(mean_abs, 0.0)
        upper = np.triu(mean_abs, k=1)
        idx = np.dstack(np.unravel_index(np.argsort(upper.ravel())[::-1][:top_n], upper.shape))[0]
        rows = [
            {
                "feature_a": feature_names[i],
                "feature_b": feature_names[j],
                "mean_abs_interaction": float(upper[i, j]),
            }
            for i, j in idx
            if upper[i, j] > 0
        ]
        return pd.DataFrame(rows, columns=columns)
    except Exception as exc:  # pragma: no cover - depends on shap version
        return pd.DataFrame(columns=columns).assign(note=str(exc))


# --------------------------------------------------------------------------------------
# LIME helpers
# --------------------------------------------------------------------------------------
def explain_instance_lime(
    model,
    X_train: pd.DataFrame,
    feature_names: Sequence[str],
    instance: pd.Series,
    cfg: ExperimentConfig,
    num_features: int = 15,
) -> Tuple[object, pd.DataFrame]:
    """Explain one prediction and return the explanation plus a tidy table."""
    from lime.lime_tabular import LimeTabularExplainer

    names = list(feature_names)
    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_train, dtype=float),
        feature_names=names,
        class_names=["legitimate", "phishing"],
        mode="classification",
        discretize_continuous=False,
        random_state=cfg.seed,
    )
    row = np.asarray(instance, dtype=float)
    explanation = explainer.explain_instance(
        row,
        model.predict_proba,
        num_features=int(num_features),
        num_samples=cfg.lime_num_samples,
        labels=(POSITIVE_CLASS,),
    )
    return explanation, lime_explanation_table(explanation, names, num_features)


def lime_explanation_table(
    explanation,
    feature_names: Sequence[str],
    k: int = 15,
) -> pd.DataFrame:
    """``as_list`` -> tidy frame ``(feature, weight)`` for the phishing class."""
    names = list(feature_names)
    name_set = set(names)
    rows = []
    for label, weight in explanation.as_list(label=POSITIVE_CLASS):
        feature = _match_feature_label(label, names, name_set) or str(label)
        rows.append({"feature": feature, "weight": float(weight)})
    frame = pd.DataFrame(rows, columns=["feature", "weight"])
    if frame.empty:
        return frame
    frame["abs_weight"] = frame["weight"].abs()
    frame["direction"] = np.where(frame["weight"] >= 0, "towards phishing", "towards legitimate")
    frame = frame.sort_values("abs_weight", ascending=False).head(int(k)).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


# --------------------------------------------------------------------------------------
# SLA-FS: fusion of SHAP (global) and aggregated LIME (local)
# --------------------------------------------------------------------------------------
def _rank_positions(ranking: FeatureRanking) -> pd.Series:
    return ranking.rank_series().astype(float)


def _minmax(series: pd.Series) -> pd.Series:
    low, high = float(series.min()), float(series.max())
    if not np.isfinite(low) or not np.isfinite(high) or high - low <= 0:
        return pd.Series(0.5, index=series.index)
    return (series - low) / (high - low)


def slafs_rank_average(
    shap: FeatureRanking,
    lime: FeatureRanking,
    method: str = "SLA-FS_rankavg",
) -> FeatureRanking:
    """MODIFICATION: fuse by the arithmetic mean of the two rank vectors."""
    with Timer() as timer:
        features = list(dict.fromkeys(list(shap.features) + list(lime.features)))
        shap_pos = _rank_positions(shap).reindex(features)
        lime_pos = _rank_positions(lime).reindex(features)
        shap_pos = shap_pos.fillna(float(shap.n_features) + 1.0)
        lime_pos = lime_pos.fillna(float(lime.n_features) + 1.0)
        blended = (shap_pos + lime_pos) / 2.0

    ranking = FeatureRanking(
        method=method,
        features=sorted(features, key=lambda f: (blended[f], f)),
        # Store negative mean rank so "higher score = better" holds everywhere.
        scores={f: float(-blended[f]) for f in features},
        elapsed_s=timer.elapsed,
        provenance="MODIFICATION",
        description="SLA-FS fusion by mean of the SHAP and LIME rank vectors (our operationalisation)",
        notes=f"shap_n={shap.n_features}, lime_n={lime.n_features}",
    )
    ranking.extras["mean_rank"] = blended.to_dict()
    return ranking


def slafs_score_blend(
    shap: FeatureRanking,
    lime: FeatureRanking,
    alpha: float = 0.5,
    method: str = "SLA-FS_scoreblend",
) -> FeatureRanking:
    """MODIFICATION: weighted blend of min-max-normalised importance scores."""
    with Timer() as timer:
        features = list(dict.fromkeys(list(shap.features) + list(lime.features)))
        shap_score = _minmax(shap.score_series()).reindex(features).fillna(0.0)
        lime_score = _minmax(lime.score_series()).reindex(features).fillna(0.0)
        blended = alpha * shap_score + (1.0 - alpha) * lime_score

    return FeatureRanking(
        method=method,
        features=sorted(features, key=lambda f: (-blended[f], f)),
        scores={f: float(blended[f]) for f in features},
        elapsed_s=timer.elapsed,
        provenance="MODIFICATION",
        description=f"SLA-FS fusion by min-max score blend (alpha={alpha} on SHAP)",
        notes=f"alpha={alpha}",
    )


def _intersection(shap: FeatureRanking, lime: FeatureRanking, k: int) -> List[str]:
    return [f for f in shap.top_k(k) if f in set(lime.top_k(k))]


def slafs_ranking_for_k(
    shap: FeatureRanking,
    lime: FeatureRanking,
    k: int,
    method: str = "SLA-FS_intersection",
) -> FeatureRanking:
    """REPRODUCTION of the paper's intersection rule, extended to a full ranking.

    The intersection of the SHAP and LIME top-k lists becomes the head of the
    ranking; the remaining features follow in rank-average order. Ordering the
    tail (instead of discarding it) lets the adaptive stage evaluate *smaller*
    subsets than ``k`` without rerunning the fusion.
    """
    base = slafs_rank_average(shap, lime)
    with Timer() as timer:
        selected = _intersection(shap, lime, k)
    tail = [f for f in base.features if f not in set(selected)]

    ranking = FeatureRanking(
        method=method,
        features=list(selected) + tail,
        scores={f: float(base.scores.get(f, 0.0)) for f in selected + tail},
        elapsed_s=base.elapsed_s + timer.elapsed,
        provenance="REPRODUCTION",
        description=f"SLA-FS intersection of SHAP and LIME top-{k}, then rank-average order",
        notes=f"k={k}, |intersection|={len(selected)}",
    )
    ranking.extras["intersection"] = selected
    ranking.extras["k"] = int(k)
    return ranking


def slafs_intersection_report(
    shap: FeatureRanking,
    lime: FeatureRanking,
    k_values: Sequence[int],
) -> pd.DataFrame:
    """How the intersection size shrinks as k grows (reported, not tuned)."""
    rows = []
    for k in k_values:
        if k > shap.n_features or k > lime.n_features:
            continue
        selected = _intersection(shap, lime, int(k))
        rows.append(
            {
                "k": int(k),
                "n_shap_topk": int(k),
                "n_lime_topk": int(k),
                "n_intersection": len(selected),
                "jaccard_shap_lime_topk": jaccard(shap.top_k(k), lime.top_k(k)),
                "selected_features": ";".join(selected),
            }
        )
    return pd.DataFrame(rows)


def build_slafs_rankings(
    rankings: Dict[str, FeatureRanking],
    k: int = 15,
    alpha: float = 0.5,
) -> Dict[str, FeatureRanking]:
    """Add the three SLA-FS variants to a dict of base rankings.

    ``k`` is the reference subset size at which the intersection variant is
    defined. It is recorded in the run metadata because, unlike the other
    variants, its identity depends on ``k``.
    """
    if "SHAP" not in rankings or "LIME" not in rankings:
        return {}
    shap, lime = rankings["SHAP"], rankings["LIME"]
    if shap.n_features == 0 or lime.n_features == 0:
        return {}

    variants = {
        "SLA-FS_rankavg": slafs_rank_average(shap, lime),
        "SLA-FS_scoreblend": slafs_score_blend(shap, lime, alpha=alpha),
        "SLA-FS_intersection": slafs_ranking_for_k(shap, lime, k),
    }
    return variants
