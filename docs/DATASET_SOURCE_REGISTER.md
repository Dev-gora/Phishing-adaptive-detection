# Dataset source register

This register records the provenance of every dataset that entered the study, as
required by the project brief. No data in this repository was generated,
simulated, augmented or hand-edited: the modelling frame is derived only from the
published file described below.

---

## 1. Dataset used

| Field | Value |
|---|---|
| **Dataset name** | Web page phishing detection |
| **Authors** | Hannousse, A. & Yahiouche, S. |
| **Repository** | Mendeley Data |
| **DOI** | [10.17632/c2gw7fy2j4.3](https://doi.org/10.17632/c2gw7fy2j4.3) |
| **Version** | 3 |
| **Landing page** | https://data.mendeley.com/datasets/c2gw7fy2j4/3 |
| **File used** | `dataset_B_05_2020.csv` |
| **Licence** | CC BY 4.0 |
| **Companion paper** | Hannousse, A., & Yahiouche, S. (2021). *Towards benchmark datasets for machine learning based website phishing detection: An experimental study.* Engineering Applications of Artificial Intelligence, 104, 104347. |
| **Preprint** | arXiv:2010.12847 |
| **Local path** | `data/dataset_phishing_raw.csv` |
| **Size** | 3,661,166 bytes |
| **SHA-256** | `21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c` |
| **Dimensions** | 11,430 rows x 89 columns = 87 features + `url` (identifier) + `status` (label) |
| **Class balance** | 5,715 legitimate / 5,715 phishing (exactly balanced) |
| **Missing values** | 0 |
| **Exact duplicate rows** | 0 |
| **Duplicate feature vectors** | 174 (identical 87-dimensional vectors under different URLs) |

### Why this file

The reference paper for this project is:

> Shafin, S. S. (2025). *An explainable feature selection framework for web
> phishing detection with machine learning.* Data Science and Management, 8(2),
> 127–136. https://doi.org/10.1016/j.dsm.2024.08.004 (open access, CC BY-NC-ND 4.0)

It reports **11,430 samples x 87 features** for a "web phishing dataset" with
URL-based, HTML-content and external-service features. The Hannousse & Yahiouche
dataset matches those dimensions exactly (11,430 x 87 features, plus identifier
and label columns), uses the same three feature families, and is the only public
dataset with that exact shape. The dataset was therefore taken from the
**publisher** (Mendeley Data) and not from a third-party mirror, and the file
checksum was verified before any experiment was run.

### Feature families (as published)

| Family | Count |
|---|---|
| URL-based | 56 |
| HTML/content-based | 24 |
| External-service-based | 7 |
| **Total** | **87** |

Six features are constant across all 11,430 rows and are dropped during
preprocessing (documented in `src/preprocessing.py`): `nb_or`,
`ratio_nullHyperlinks`, `ratio_intRedirection`, `ratio_intErrors`, `submit_email`
and `sfh`. Together with the removal of 174 duplicate feature vectors this yields
the final modelling frame: **11,256 rows x 81 features**, split 6,753 / 2,251 /
2,252 (train / validation / test). All of these counts are machine-generated into
`results/metrics/preprocessing_report.json` on every run, not transcribed.

### Verification commands

```bash
python main.py --stage download   # re-fetch from Mendeley and check SHA-256
python main.py --stage verify     # audit dimensions, balance, duplicates
```

Both stages write their reports to `results/metrics/`.

---

## 2. Sources investigated but not used

Each source in the brief was examined before choosing a dataset. None was
discarded for convenience; each was ruled out for a specific reason.

| Source | What it provides | Why it is not the modelling dataset |
|---|---|---|
| **UCI Machine Learning Repository** | Several phishing-related sets (e.g. *Phishing Websites* 11,055 x 30, *Website Phishing* 1,353 x 9) | Different sample count, feature count and feature semantics from the reference paper's 11,430 x 87. Using it would change the experiment, not reproduce it. |
| **Kaggle** | Re-uploads of the Hannousse & Yahiouche file, plus unrelated sets | Mirrors are unversioned third-party copies; provenance and byte-identity cannot be guaranteed. The authoritative publisher copy was used instead. |
| **PhishTank** | A live feed of verified phishing URLs | Provides URLs only, with no precomputed URL/HTML/external feature matrix and no legitimate-class control group. Raw URLs are also not a drop-in replacement for an 87-feature table; producing features would mean crawling live sites, which this project does not do. |
| **OpenPhish** | A live feed of phishing URLs | Same limitation as PhishTank: URL lists, no feature matrix, no balanced legitimate class. |
| **UNB CIC datasets** | Network-traffic and URL datasets (e.g. CIC-IDS, URL-based phishing sets) | Different collection methodology (traffic captures / different URL feature extraction), different dimensions and label semantics. Not the dataset behind the reported 11,430 x 87 table. |

**No web scraping was performed.** The one file that was downloaded came from a
public, licensed data repository through its official API, under a CC BY 4.0
licence that permits reuse with attribution.

---

## 3. Differences between the original and the data actually used

There is no substitute dataset, so there is no original-vs-replacement gap to
document. Two things must still be stated precisely, because they are *not* the
same kind of claim:

1. **Dimension match is verified.** 11,430 x 87 features is confirmed from the
   file itself (`results/metrics/dataset_verification.json`), not assumed.
2. **Feature-level identity is inferred, not verified.** The reference paper does
   not publish its column list, its train/test split, or its preprocessing code.
   The match on sample count, feature count and feature families is strong
   evidence that the same table was used, but it cannot be proven from the paper
   alone. Every result in this repository is therefore our own measurement on the
   public file, and is reported as such.

---

## 4. Reference paper — reported results (for comparison only)

The following numbers are **quoted from the reference paper's abstract**, not
produced by this project. They are listed here so that our results can be placed
in context, and they are never presented as our findings.

| Model | Accuracy reported by Shafin (2025) |
|---|---|
| Random Forest | 97.41% |
| XGBoost | 97.21% |
| KNN | 84.51% |

The paper reports SLA-FS as outperforming information gain, chi-square and RFE on
the same dataset. Our pipeline reproduces the *idea* (SHAP global + aggregated
LIME local -> ranking/intersection), not the authors' unpublished code; see
`METHOD_PROVENANCE` in `src/explainability.py` and the methodology section of the
README for the exact distinction between reproduction and modification.

---

## 5. How to cite

```bibtex
@article{hannousse2021phishing,
  title   = {Web page phishing detection},
  author  = {Hannousse, Abdelhakim and Yahiouche, Salima},
  year    = {2021},
  note    = {Mendeley Data, V3},
  doi     = {10.17632/c2gw7fy2j4.3}
}

@article{shafin2025slafs,
  title   = {An explainable feature selection framework for web phishing
             detection with machine learning},
  author  = {Shafin, Sakib Shahriar},
  journal = {Data Science and Management},
  volume  = {8},
  number  = {2},
  pages   = {127--136},
  year    = {2025},
  doi     = {10.1016/j.dsm.2024.08.004}
}
```
