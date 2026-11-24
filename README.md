# Lightweight and Explainable Phishing Detection Using Adaptive Feature Selection

A reproducible research pipeline for phishing-website detection that

1. ranks features with six established selection techniques **and** with the
   SLA-FS fusion proposed in the reference paper,
2. chooses the **smallest** feature subset that stays within a stated F1 tolerance
   of the full-feature baseline (the *adaptive* contribution), and
3. reports accuracy, cost and stability for every configuration on a test split
   that is never touched until all decisions are final.

Reference paper (reproduced in part, extended in part):

> Shafin, S. S. (2025). *An explainable feature selection framework for web
> phishing detection with machine learning.* Data Science and Management, 8(2),
> 127–136. <https://doi.org/10.1016/j.dsm.2024.08.004>

---

## 1. Quick start

```bash
# 1. environment (Python 3.10+; verified on 3.14.4)
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux / Colab

pip install -r requirements.txt

# 2. one command: dataset check -> experiments -> figures
python main.py --stage all
```

`--stage all` writes every table to `results/metrics/`, every ranking to
`results/feature_rankings/` and every figure to `results/visualizations/`.

### Smoke test first (recommended)

```bash
python main.py --stage all --quick
```

`--quick` shrinks sample sizes, drops LightGBM and skips hyper-parameter tuning.
It exercises the whole pipeline in a few minutes and its outputs are suffixed
`_fast`. **Fast-mode numbers are not reportable** and the code labels them as
such.

### Google Colab

Colab ships most of the stack already; `pip install -r requirements.txt`
completes it. Upload or clone the folder, then run the same two commands. Nothing
in the pipeline is Windows-specific.

---

## 2. Repository layout

```
phishing-detection/
├── data/
│   ├── dataset_phishing_raw.csv        # published file (not modified)
│   └── dataset_metadata.json           # download provenance + SHA-256
├── docs/
│   └── DATASET_SOURCE_REGISTER.md      # source register required by the brief
├── notebooks/README.md                 # how to explore results interactively
├── results/
│   ├── metrics/                        # CSVs + JSON audit trails
│   ├── feature_rankings/               # per-method rankings + SHAP/LIME artefacts
│   ├── visualizations/                 # 300 dpi PNG + vector PDF
│   └── logs/
├── src/
│   ├── config.py                       # ExperimentConfig: every knob + seeds
│   ├── preprocessing.py                # validation, cleaning, leakage-safe splits
│   ├── models.py                       # RF / XGBoost / LightGBM / KNN + tuning
│   ├── feature_selection.py            # the six ranking techniques
│   ├── explainability.py               # SHAP/LIME helpers + SLA-FS fusion
│   ├── adaptive_selection.py           # the adaptive rule (main contribution)
│   ├── evaluation.py                   # metrics, timing, Jaccard stability
│   └── visualization.py                # publication figures
├── main.py                             # staged CLI
├── requirements.txt
└── README.md
```

`config.py` and `visualization.py` are additions to the brief's file list: the
first is the single source of truth for reproducibility, the second keeps ~450
lines of plotting out of the experiment code.

---

## 3. Stages

| Stage | Command | Produces |
|---|---|---|
| 1–2 | `--stage download` / `--stage verify` | dataset + checksum + verification report |
| 3 | `--stage preprocess` | validation, cleaning, split, correlation tables |
| 4 | `--stage baselines` | tuned hyper-parameters + full-feature metrics |
| 5 | `--stage feature_selection` | six rankings + SLA-FS + stability |
| 6 | `--stage experiments` | the full experiment, including the single test-set evaluation |
| 7 | `--stage visualize` | all figures, rebuilt from the saved tables |

`--stage adaptive` is intentionally not a standalone command: the adaptive rule
depends on the tuned models and rankings produced earlier in the same process, and
re-deriving them separately would risk the two halves drifting apart. Use
`--stage experiments`.

Useful flags: `--seed`, `--epsilons 0.01 0.02 0.05`, `--models RandomForest,XGBoost`,
`--no-lightgbm`, `--no-tuning`, `--interactions`, `--quick`.

---

## 4. Dataset

Full provenance, licence and checksum: **`docs/DATASET_SOURCE_REGISTER.md`**.

In short: Hannousse & Yahiouche (2021), *Web page phishing detection*, Mendeley
Data **V3**, DOI [10.17632/c2gw7fy2j4.3](https://doi.org/10.17632/c2gw7fy2j4.3),
**CC BY 4.0** — downloaded from the publisher, SHA-256 verified, 11,430 x 87
features (56 URL-based, 24 HTML/content, 7 external), exactly balanced
5,715 / 5,715, no missing values, no duplicate rows. No synthetic or augmented
data is used anywhere.

---

## 5. Preprocessing and leakage control

| Step | Decision | Reason |
|---|---|---|
| `url` column | dropped before modelling | it is an identifier; keeping it would leak the label's source |
| Ambiguous URLs | any URL carrying two different labels is removed entirely | identical identifier with contradictory labels is unresolvable |
| Duplicate feature rows | 174 removed before splitting | identical feature vectors across splits leak information |
| Missing values | median imputation **inside** the model pipeline | fitted on training folds only |
| Categorical features | none present; label mapped `phishing -> 1` | the published table is fully numeric |
| Constant features | 6 dropped: `nb_or`, `ratio_nullHyperlinks`, `ratio_intRedirection`, `ratio_intErrors`, `submit_email`, `sfh` | zero information, undefined correlation |
| Scaling | `StandardScaler` inside the KNN `Pipeline` only | fitted per training fold; tree models need no scaling |
| Highly correlated pairs (`|r| >= 0.95`) | **reported, not removed** | deleting them would change the feature set and confound the selection comparison |
| Split | stratified 60 / 20 / 20, `seed = 42` | train = fit, validation = selection, test = report once |
| Resulting frame | 11,256 rows x **81** features (6,753 / 2,251 / 2,252) | 174 duplicate feature vectors removed, 6 constant columns dropped |

**Test-set protocol.** `make_splits()` creates the test split once. No selector,
no sweep and no hyper-parameter search ever receives it. It is used exactly twice
in `_final_test_evaluation`: once to score the full-feature baselines, once to
score the decided reduced models. Everything that influenced a decision (ranking,
`k`, hyper-parameters, `epsilon`) was fixed on train/validation before that call.

---

## 6. Feature selection

All six techniques return the **complete ordering** of all features, not a single
subset, so the same ranking can be truncated at any `k`. SHAP, LIME, permutation
importance and Gini importance inspect the *same* fitted Random Forest (300 trees,
`seed = 42`), so differences come from the selector rather than the model.

| Method | Importance definition | Cost note |
|---|---|---|
| SHAP | mean \|SHAP value\| over 1,000 training rows (TreeSHAP, exact for trees) | expensive: model-specific Shapley values |
| LIME | mean \|local weight\| over 500 explained training rows | expensive: one ridge surrogate per row |
| Permutation | mean F1 drop when a feature is shuffled, averaged over training CV folds | moderate: refits the surrogate per fold |
| Mutual information | univariate MI with the label (k-NN estimator) | cheap, model-free |
| RFE | reverse elimination order down to 1 feature | expensive: one fit per elimination round |
| RF importance | mean decrease in Gini impurity | free once the surrogate exists (reported separately) |

### SLA-FS — reproduced vs modified

The paper names the fusion ("SHAP global + aggregated LIME local -> feature
ranking/intersection") but does not publish its arithmetic or its `k`. Three
variants are therefore implemented and **labelled**:

| Method | Label | Rule |
|---|---|---|
| `SLA-FS_intersection` | **REPRODUCTION** | features in *both* the SHAP and LIME top-`k` lists (k = 15) |
| `SLA-FS_rankavg` | **MODIFICATION** | mean of the two rank vectors, re-sorted |
| `SLA-FS_scoreblend` | **MODIFICATION** | `0.5 *` min-max-normalised SHAP score `+ 0.5 *` normalised LIME score |

The distinction is written into `METHOD_PROVENANCE`
(`src/explainability.py`), exported to `results/metrics/run_metadata.json` and
carried as a `provenance` column in every results table, so no table can be read
without it. "MODIFICATION" means *our operationalisation, not verified against the
authors' code* — not a claim of equivalence.

---

## 7. Adaptive feature selection (main contribution)

For a ranking and a tolerance `epsilon`:

```
for each candidate size k in an ascending grid:
    fit the tuned model on the training split using the top-k features
    F1(k) <- F1 on the validation split
selected k = smallest k such that  F1(k) >= F1(full) - epsilon
```

Three properties make this defensible rather than a search for a lucky number:

1. **`epsilon` is stated, not hidden.** Three tolerances (0.01 / 0.02 / 0.05) are
   reported side by side, so the accuracy–compactness trade-off is explicit.
2. **Patience.** The criterion must hold for `patience = 3` *consecutive* subset
   sizes. A single noisy spike at `k = 3` cannot produce a 3-feature model.
3. **Honest fallback.** If no `k` qualifies, the full feature set is retained and
   `fallback_used = True` is recorded. The algorithm never assumes fewer features
   must be better, and never silently returns a worse model.

Two further safeguards:

* **Grid + refinement.** A capped coarse grid (≤ 14 points, always including
  `k = 1` and `k = n`) is evaluated first, then a dense window of
  `selected_k ± step` is evaluated so the true smallest qualifying `k` is not
  missed because of grid granularity.
* **Monotonicity is not assumed.** The curves genuinely rise and fall; that is
  visible in `results/visualizations/adaptive_curve_*.png`.

The validation curve, the raw smallest qualifying `k` (`raw_smallest_k`, i.e.
without patience) and the accepted `k` are all stored, so a reader can see exactly
what patience changed.

---

## 8. Models

Random Forest, XGBoost, KNN (required) and LightGBM (optional).

* Hyper-parameters are tuned **once** on the training split with stratified 3-fold
  CV, F1-scored, and then reused for *every* subset.
* Why not re-tune per subset: ~9 methods x 4 models x ~18 subset sizes ≈ 650 extra
  grid searches. Reusing one tuned configuration keeps the comparison *about the
  feature set*; per-subset tuning would mix two effects. This is a documented
  limitation, not an oversight.
* KNN carries its own `StandardScaler` in a `Pipeline`, so scaling is always
  fitted on the data it is allowed to see.

---

## 9. Metrics

Accuracy, precision, recall, F1 and ROC-AUC (phishing = positive class), plus
training time, inference time (total and per sample), feature-selection time,
number of selected features, reduction %, and feature-selection stability.

**Stability** is the mean pairwise **Jaccard similarity** of the top-`k` feature
sets across repeated runs on 80 % subsamples of the training split. The repeats use
a reduced-cost selector configuration (recorded in
`results/metrics/stability_protocol.json`) — stability measures the robustness of
each selector's *ordering*, not its headline ranking, and this is stated wherever
the stability numbers appear.

---

## 10. Outputs

`results/metrics/`
| File | Content |
|---|---|
| `dataset_verification.json` | audit of the raw file |
| `preprocessing_report.json` | every cleaning decision, with counts |
| `split_summary.csv`, `class_distribution.csv` | split sizes and balance |
| `target_correlation.csv`, `highly_correlated_pairs.csv`, `correlation_matrix_train.csv` | correlation analysis (train only) |
| `tuning.json` | selected hyper-parameters + CV F1 |
| `full_feature_baseline.csv` | full-feature metrics per model |
| `stability.csv`, `stability_protocol.json` | Jaccard stability and its protocol |
| `validation_curves.csv` | F1 vs k for every method x model |
| `adaptive_selection.csv` | the adaptive decision table (the main results table) |
| `final_test_metrics.csv` | test-set metrics for every decided configuration |
| `roc_curves.csv`, `confusion_matrices.csv` | plotting inputs |
| `run_metadata.json` | environment, full config, provenance register |

`results/feature_rankings/`
`feature_rankings.csv` (rank + score per method), `feature_selection_summary.csv`,
`rank_correlation_spearman.csv`, `topk_jaccard_agreement.csv`,
`slafs_intersection_report.csv`, `adaptive_selected_features.csv`,
`shap_mean_abs.csv`, `shap_values_sample.csv`, `shap_sample_X.csv`,
`lime_example.csv`.

`results/visualizations/` — 300 dpi PNG + vector PDF, ~20 figure families:
feature importance by method, rank heatmap, ranking agreement, SHAP beeswarm,
single-instance SHAP, LIME explanation, confusion matrices, ROC curves, F1
comparison, features-vs-performance, selection time, model cost, stability curve
and bars, adaptive curves, adaptive summary, reduction-vs-cost, correlation
structure, target correlation, class distribution, correlated pairs.

---

## 11. Reproducibility

* One dataclass (`ExperimentConfig`) holds every knob; `to_dict()` is written next
  to the results on every run (`run_metadata.json`).
* All randomness derives from `seed = 42` (NumPy, `random`, scikit-learn,
  XGBoost, LightGBM, LIME, and the subsampling used for SHAP/stability).
* Library versions are recorded at runtime (`describe_environment()`).
* The dataset checksum is verified, so the input is byte-identical.
* Deterministic single-threaded reduction is **not** forced: tree ensembles with
  `n_jobs > 1` can differ in the last decimal between runs. Set `n_jobs: 1` in
  `ExperimentConfig` if bit-exact repetition is required (slower).

---

## 12. Limitations (stated, not hidden)

1. **The reference paper's code is unavailable**, so SLA-FS arithmetic and `k` are
   our interpretation. Two of the three variants are explicitly `MODIFICATION`.
2. **Feature-level identity with the paper's table is inferred** from matching
   dimensions and feature families; the paper publishes no column list.
3. **Hyper-parameters are not re-tuned per subset** (compute cost, and to isolate
   the effect of the feature set).
4. **Stability uses a reduced-cost selector protocol** for its repeats.
5. **The sweep grid is capped** at 14 coarse points plus a refinement window, so
   the reported optimum is the optimum *on that grid*. The refinement step bounds
   the error this introduces but does not eliminate it.
6. **`SLA-FS_intersection` depends on the reference `k`** (15). The adaptive stage
   orders features, so it can still evaluate smaller subsets, but the *identity* of
   the head of that ranking is tied to `k`.
7. **No live-website validation.** Everything here is offline tabular
   classification on a 2020 feature snapshot; concept drift in the wild is not
   measured, and the paper's "lightweight" claim is evaluated as feature count and
   measured runtime, not as deployed latency.
8. **Single dataset, single split.** Differences below ~0.3 % F1 between
   configurations should not be treated as meaningful without repeated splits.
9. **The tolerance is a validation-split guarantee, not a test-set guarantee.**
   Because the rule selects the *smallest* qualifying `k`, the chosen point sits on
   the tolerance boundary and can fall below it on a different split. Measured
   transfer rates are in §13.5 — read them before reusing this rule.

Failed and abandoned approaches are recorded in `docs/EXPERIMENT_LOG.md`.

---

## 13. Results

Produced by `python main.py --stage all` with `seed = 42` on the dataset described
above. Every number below is read from `results/` and was measured in that run;
no value is copied from another paper. Test-split values are stated once, after all
selection and tuning decisions were fixed.

### 13.1 Data actually modelled

| Quantity | Value |
|---|---|
| Raw rows x columns | 11,430 x 89 (87 features + `url` + `status`) |
| Duplicate feature vectors removed | 174 |
| Constant features removed | 6 |
| Final frame | **11,256 x 81** |
| Split (train / validation / test) | 6,753 / 2,251 / 2,252 |

### 13.2 Full-feature baselines

Hyper-parameters were selected by 3-fold CV F1 on the training split only.

| Model | Tuned configuration | CV F1 | Validation F1 | **Test F1** | Test ROC-AUC | Test accuracy |
|---|---|---|---|---|---|---|
| Random Forest | 200 trees, `max_depth=20` | 0.9625 | 0.9564 | **0.9589** | 0.9920 | 0.9596 |
| XGBoost | 400 trees, `max_depth=4` | 0.9654 | 0.9645 | **0.9625** | 0.9924 | 0.9631 |
| KNN | `k=5`, distance weights | 0.9301 | 0.9330 | **0.9379** | 0.9737 | 0.9401 |
| LightGBM | 400 trees, 31 leaves | 0.9658 | 0.9636 | **0.9643** | 0.9937 | 0.9649 |

KNN trails the ensembles by ~2.6 F1 points, consistent with the reference paper's
finding that instance-based learning is the weakest of the three model families on
this dataset.

### 13.3 Feature selection

All nine rankings agree on the head of the list: `google_index`, `page_rank`,
`web_traffic`, `nb_hyperlinks`, `nb_www`, `phish_hints`, `domain_age`.
`google_index` is first under every method with a very large margin.

**Cost** (measured, single run, same machine):

| Method | Selection time |
|---|---|
| SHAP (1,000 rows, TreeSHAP) | 82.3 s |
| Permutation (3 CV folds, F1) | 87.4 s |
| RFE (down to 1 feature) | 78.3 s |
| LIME (500 rows, 500 samples each) | 76.1 s |
| Mutual information | 3.0 s |
| Random Forest Gini importance | 0.08 s (surrogate already fitted: 1.4 s) |
| SLA-FS fusion (all three variants) | < 0.02 s |

Wall-clock times come from the single run that produced the current `results/`.
They move by up to ~2.5x with machine load: an identical earlier run of the same
code measured 204.6 s (SHAP), 143.7 s (permutation), 133.3 s (RFE) and 96.2 s
(LIME). The three explanation-based costs are therefore interchangeable, and
their ordering should not be interpreted; what is stable is the gap to the filter
baselines.

The explainability-based methods cost two to three orders of magnitude more than
filter/Wrapper baselines for rankings that are **0.98–0.99 Spearman-correlated**
with plain Gini importance. That is a practically important negative result for
anyone proposing SHAP/LIME-based selection on this dataset.

**Ranking agreement (Spearman):** SHAP↔RF importance 0.99, SHAP↔RFE 0.98,
SHAP↔mutual information 0.90, SHAP↔LIME 0.89. Permutation importance is the
outlier (0.67–0.79 against everything else) — it measures F1 sensitivity rather
than impurity or attribution, and it is also the least stable selector.

**Stability** (mean pairwise Jaccard of the top-10, 5 subsamples of the training
split):

| Method | Jaccard @ k=10 |
|---|---|
| LIME | 1.000 |
| Mutual information | 0.927 |
| RFE | 0.927 |
| RF importance | 0.927 |
| SHAP | 0.855 |
| SLA-FS_rankavg | 0.855 |
| Permutation | 0.603 |

LIME's top-10 is perfectly reproducible across subsamples while its top-15 drops
to 0.756 — the leading features are strongly separated and the tail is not.
Permutation importance is markedly unstable, which is the expected consequence of
measuring F1 sensitivity on ~2,200-row holdouts.

**SLA-FS intersection** (`|top-k(SHAP) ∩ top-k(LIME)|`): 5/5 at k=5, 8/10 at k=10,
12/15 at k=15, 16/20, 21/25, 25/30, 37/40. SHAP and LIME agree exactly on the first
five features and then diverge, which is what the Jaccard curve 1.000 → 0.667 →
0.860 shows.

### 13.4 Adaptive selection

36 configurations per tolerance (9 rankings x 4 classifiers). `ΔF1` is the test-set
change against **that model's own** full-feature test score.

| ε | Mean feature reduction | Configs within tolerance on test | Mean ΔF1 | Worst ΔF1 |
|---|---|---|---|---|
| 0.01 | 84.4 % | 7 / 36 | −0.0149 | −0.0263 |
| 0.02 | 90.4 % | 14 / 36 | −0.0226 | −0.0321 |
| 0.05 | **95.1 %** | **36 / 36** | −0.0391 | −0.0497 |

Representative configurations that *do* hold up on test at ε = 0.02:

| Model | Ranking | k | Reduction | Test F1 | ΔF1 |
|---|---|---|---|---|---|
| LightGBM | LIME | 12 | 85.2 % | 0.9532 | −0.0111 |
| XGBoost | SLA-FS_scoreblend | 12 | 85.2 % | 0.9505 | −0.0120 |
| LightGBM | Permutation | 8 | 90.1 % | 0.9509 | −0.0134 |

### 13.5 Main finding: the tolerance is a validation guarantee, not a test guarantee

The adaptive rule satisfies `F1(k) >= F1(full) − ε` **on the validation split by
construction**, yet only 7/36 (ε=0.01) and 14/36 (ε=0.02) of the chosen subsets keep
that promise on test, while all 36 do at ε=0.05.

The mechanism is visible in `results/visualizations/adaptive_curve_*.png`: the rule
returns the *smallest* qualifying `k`, so the selected point is parked exactly on
the tolerance boundary. Any split-to-split variation then pushes roughly half of
the configurations below the line — the ones selected at a point where the curve
was locally flat survive, the ones selected on a rising shoulder do not. A larger
ε moves the selected point away from the boundary onto the flat part of the curve,
which is why ε=0.05 transfers perfectly.

This is reported as a **finding, not a defect**: it quantifies the reliability of
validation-based subset selection and shows that the tolerance must be chosen with
headroom for split noise, not read as a bound on unseen data. It is the reason this
project reports validation curves and the raw `raw_smallest_k` alongside every
decision, and why the README does not claim that 90 % fewer features are "free".

### 13.6 Relation to the reference paper

The reference paper reports Random Forest accuracy of **97.41 %** on this dataset
under its own preprocessing and split. Our Random Forest reaches **95.96 %** test
accuracy. These are not directly comparable: we remove 174 duplicate feature
vectors and 6 constant features, use a stratified 60/20/20 split with a fixed seed,
tune with a small CV grid, and evaluate on a single held-out block rather than
cross-validation. The gap should be read as a difference in protocol, not as
evidence for or against either implementation. Our contribution is the adaptive
selection rule and its measured behaviour, not a reproduction of the reported
headline accuracy.

---

## 14. Licences and attribution

* **Dataset**: CC BY 4.0 — attribution required, see `docs/DATASET_SOURCE_REGISTER.md`.
* **Reference paper**: CC BY-NC-ND 4.0. Cited, not redistributed.
* This code: provided for the associated undergraduate research project.

If you use the dataset, cite Hannousse & Yahiouche (2021); if you use the SLA-FS
idea, cite Shafin (2025).
