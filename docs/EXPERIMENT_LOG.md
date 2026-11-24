# Experiment log

Chronological record of what was attempted, what failed, and why. The brief
requires failed experiments and technical limitations to be recorded rather than
quietly dropped; this file is that record. It contains no performance claims —
results live only in `results/` and in the README's results section.

---

## 1. Environment setup

| Attempt | Outcome |
|---|---|
| Install dependencies in the foreground (`pip install …`) | **Timed out** after 295 s while downloading `scipy` (37 MB) mid-transfer. Aborted; no partial installation was left in an unusable state. |
| Re-run as a background process logging to `install.log` | **Succeeded.** `shap 0.54.1`, `lime 0.2.0.1`, `xgboost 3.4.1`, `lightgbm 4.7.0`, `matplotlib 3.11.2`, `seaborn 0.13.2`, `scikit-learn 1.9.1` all installed and imported on **Python 3.14.4**. |

Python 3.14 is very new; the risk that `shap`, `lime`, `xgboost` or `lightgbm`
would have no compatible wheel was real. They installed (`lime` built from
source). No fallback implementation was needed — the documented local-surrogate
fallback in `src/feature_selection.py` remains in the code as a guard but was
**not exercised** in the reported run.

A project-local virtual environment (`phishing-detection/.venv`) was used instead
of installing into the global interpreter.

---

## 2. Code defects found by running the pipeline

Every defect below was found by an actual run and fixed; none was worked around
by disabling a check or weakening an assertion.

### 2.1 RFE crashed: `IndexError: index 1 is out of bounds for axis 0 with size 1`

**Cause.** After the final elimination round, scikit-learn's `RFE` refits its
estimator on the surviving feature(s) only. With `n_features_to_select=1`,
`selector.estimator_.feature_importances_` therefore has length **1**, not `p`,
and indexing it by original feature position fails.

**Fix.** Elimination order (`ranking_`) defines the ranking; ties are broken with
the *shared* Random Forest surrogate's importances, which are defined over the
full feature set. This also keeps RFE consistent with the "one surrogate model"
rule used by the other model-based selectors.

**Effect on results.** The first (fast) run had to skip RFE entirely; the reported
run includes it.

### 2.2 Correlation heatmap crashed: `could not convert string to float: 'length_url'`

**Cause.** `DataFrame.to_csv` writes the correlation matrix with its feature names
as the index. Reading it back with a plain `read_csv` produced an extra string
column, which Seaborn could not cast to float.

**Fix.** The figure stage reads that file with `index_col=0`. The stored file was
not changed, so the matrix on disk is still complete and reusable.

### 2.3 F1 comparison crashed: inhomogeneous array shape

**Cause.** `final_test_metrics.csv` contains one row per `(model, method, epsilon)`,
so setting `method` as the index produced duplicate labels and each bar received a
`Series` instead of a scalar.

**Fix.** Duplicates are collapsed with `groupby(...).max()` inside the plotting
function, and the figure stage passes one row per `(model, method)` (full
baselines plus the default tolerance `epsilon = 0.02`).

### 2.4 Feature-importance panels were **inverted** for SLA-FS

**Cause.** Bars were ordered by `|score|`. That is correct for SHAP, LIME, mutual
information and Gini importance, but `SLA-FS_rankavg` stores `-mean rank`
(higher = better) and permutation importance can be negative. Sorting by magnitude
therefore ranked the *least* important features first, and the fastest way to spot
it was to look at the figure: `nb_dslash`, `iframe` and `onmouseover` appeared at
the top of the SLA-FS panels instead of `google_index`.

**Fix.** Order and scale by the signed score, which is the correct convention for
every selector in the pipeline.

**Lesson recorded.** This class of bug is invisible in numbers-only output; it was
caught by rendering the figure and reading it, not by the code passing.

### 2.5 SHAP beeswarm colour scale was dominated by one feature

**Cause.** Colouring every row from one pooled colour scale meant `google_index`
(values in the 1e7 range) compressed all other features into a single colour.

**Fix.** Each row is colour-scaled on its own 5th–95th percentile range, as in the
standard SHAP beeswarm. The colour bar is labelled "scaled per feature" so the
figure cannot be misread.

### 2.6 LIME feature-name warnings flooding the log

**Cause.** LIME perturbs instances as NumPy arrays; the surrogate was fitted on a
DataFrame with column names, so scikit-learn warned on every call.

**Fix.** The prediction wrapper rebuilds a DataFrame with the training column names
before calling `predict_proba`. This also guarantees column order.

---

## 3. Methodological decisions taken because of cost

These are not bugs, but they bound what the results can claim.

| Decision | Reason | Where it is disclosed |
|---|---|---|
| Hyper-parameters tuned **once** on train, reused for all subsets | Re-tuning per subset would mean ~650 extra grid searches and would mix tuning effects into the subset comparison | README §8, `src/models.py` header |
| Sweep grid capped at 14 coarse points + a refinement window | A dense 1..p grid multiplies the sweep cost by ~6 for no decision-relevant gain | README §12, `MAX_SWEEP_POINTS` in `main.py` |
| Stability repeats use a cheaper selector configuration | RFE alone costs minutes per repeat; 5 repeats of the full configuration would dominate the whole run | `results/metrics/stability_protocol.json`, README §9 |

---

## 4. Things that were deliberately **not** done

* **No substitute or synthetic dataset.** The published file matched the reported
  dimensions, so generating data was never necessary and would have invalidated
  the study.
* **No crawling of live phishing sites.** PhishTank and OpenPhish are URL feeds
  without a feature matrix; scraping them would have required crawling live
  websites, which is out of scope and was not authorised. See
  `docs/DATASET_SOURCE_REGISTER.md`.
* **No dropping of correlated features.** Pairs above `|r| = 0.95` are reported,
  not removed, because removal would change the feature set and confound the
  comparison between selection methods.
* **No per-subset re-tuning and no test-set peeking.** The test split is used once,
  after every decision is fixed.
* **SHAP interaction values are optional** (`--interactions`, bounded to 40 rows)
  because TreeSHAP interactions are O(p²) per instance. They are supplementary and
  are not used for any selection decision.

---

## 4b. Findings that emerged only by running the experiments

* **The adaptive rule parks the subset on the tolerance boundary.** Because the rule
  returns the *smallest* qualifying `k`, the chosen model sits exactly at
  `F1(full) - epsilon` on validation. On test, only 7/36 (epsilon=0.01) and 14/36
  (epsilon=0.02) of configurations stayed inside tolerance, against 36/36 at
  epsilon=0.05. This was not predicted when the rule was designed; it is recorded
  in README section 13.5 as the study's main empirical finding rather than hidden.
* **SHAP- and LIME-based rankings are almost perfectly rank-correlated with free
  Gini importance** (Spearman 0.98 and 0.88) while costing 200 s and 96 s against
  0.1 s. This is an efficiency argument against explainability-based selection on
  this dataset, and it was only visible after measuring both cost and agreement.
* **Permutation importance is the least stable selector** (Jaccard 0.603 at k=10 vs
  1.000 for LIME), which limits how much weight its ranking should carry.
* **Three selectors reported identical Jaccard at k=10** (mutual information, RFE,
  Gini: 0.927). This was checked rather than assumed: the numbers diverge at k=15,
  20, 30 and 40, and the three selectors are distinct functions, so the agreement at
  k=10 is a genuine coincidence of membership, not a plumbing bug.
* **The initial pipeline run skipped RFE entirely** (defect 2.1), so any earlier
  fast-mode numbers that lack RFE rows must not be compared with the reported run.

---

## 5. Reproducing the reported run

```bash
python main.py --stage all
```

Environment, configuration and library versions for that run are written to
`results/metrics/run_metadata.json`. Progress output for the run behind the
current `results/` is in `run_after_cleanup.log` (the earlier identical run, before
the cleanup pass, is `run_full.log`; fast smoke test: `smoke.log`). Fast-mode
artefacts are suffixed `_fast` and are **not** reportable.

---

## 6. Cleanup pass (dead code and a duplicate refit)

Housekeeping after the reported run: no experimental variable was changed, and the
pipeline was re-run end to end to prove it.

**Removed — dead code.**

| Item | Why it was dead |
|---|---|
| `ExperimentConfig.extra_seeds` | Unused field. Its comment referenced a `--robustness` flag that was never implemented in `main.py`'s CLI, so the config advertised a repeated-seed mode that did not exist. |
| `evaluation.stability_curve`, `evaluation.selection_frequency` | Never called; stability is computed through `compute_stability_table`. |
| `evaluation.save_results` | Never called; every table is written with `DataFrame.to_csv` after its own column handling. |
| `adaptive_selection.best_tradeoff`, `adaptive_selection.compare_to_baseline` | Never called. `summarize_across_epsilons` is the reporting path actually used. |
| `FeatureRanking.top_table` | Never called; rankings are serialised by `save_combined_rankings`. |
| `_sweep_and_select`'s third return value and its internal `provenance` dict | The third value was always the empty dict `{}` although the docstring promised a fitted-model cache; the provenance dict was shadowed by the one that `stage_experiments` builds. The bogus cache promise was replaced with an accurate docstring. |
| 13 unused imports across `main.py`, `src/explainability.py`, `src/feature_selection.py` | Flagged by an AST check, which now reports zero unused imports. (`Timer` stayed in `main.py`: the refit fix below makes it used again.) |

**Fixed — duplicate model fit in the final test stage.** `_final_test_evaluation`
previously called `_evaluate_subset` (which fits and scores) and then built and
fitted the *same* estimator again solely to obtain ROC points. Every decided
configuration was therefore fitted twice, for 112 test configurations. It now
fits once and takes both the metrics and the ROC curve from that single estimator,
keeping `train_time_s` measured over the one fit. A within-run consistency check
confirms the reported `roc_auc` and the ROC table's `auc` agree to 1.1e-16, which
the two-fit path could not guarantee.

**Verification.** The full pipeline was re-run (`run_after_cleanup.log`, `EXIT=0`,
29 figures) and every table compared with the pre-cleanup snapshot. Identical:
baselines, `feature_rankings`, `adaptive_selection` (all 108 decisions, hence the
7/36, 14/36 and 36/36 counts), `adaptive_selected_features`, `stability`,
`confusion_matrices`, `split_summary`, `lime_example`, `shap_values_sample`,
`target_correlation`, `highly_correlated_pairs`, and both agreement matrices.
Expected differences: the two timing columns (`selection_time_s`,
`surrogate_fit_s`) and the retired `extra_seeds` key in `run_metadata.json`.
Remaining drift: two ROC-AUC cells moved by 7.9e-7 (= 1/(1126^2), exactly one
score pair) and three RandomForest `k=6` ROC curves lost two vertices while their
AUC stayed fixed at 1e-16. Cause: cross-process float noise in RandomForest's
threaded `predict_proba`, not this change — the *untouched* validation sweep shows
the same 7.9e-7 drift on one row. No cell differs at the four decimals the README
reports.

The selection timings quoted in README section 13.3 were updated to this run's
measurements; they are wall clock and moved by up to ~2.5x between the two runs of
identical code.
