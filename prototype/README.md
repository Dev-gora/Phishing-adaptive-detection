# Phishing Detection Prototype

A complete prototype implementation of "Lightweight and Explainable Phishing Detection Using Adaptive Feature Selection" with a backend API and web frontend that uses the actual machine learning models from the research project.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Phishing Detection Prototype                 │
├─────────────────────┬─────────────────────┬─────────────────┤
│     Frontend        │      Backend API    │   Project Src   │
│  (React + CDN)      │  (FastAPI + Uvicorn)│  (Python)       │
│  port: 3000         │  port: 8000         │  main.py        │
├─────────────────────┼─────────────────────┼─────────────────┤
│  • React components │  • Dataset download │  • Config       │
│  • State management │  • Preprocessing    │  • Models       │
│  • API calls        │  • Feature selection│  • Evaluation   │
│  • Result display   │  • Adaptive FS      │  • Visualization│
└─────────────────────┴─────────────────────┴─────────────────┘
```

## Models Used

The prototype implements four machine learning models from the research project:

| Model | Type | Key Characteristics |
|-------|------|---------------------|
| **RandomForest** | Tree ensemble | 300 trees, Gini impurity, `max_features='sqrt'`. Provides built-in feature importance. |
| **XGBoost** | Gradient boosting | 300 trees, histogram method, subsample=0.9, colsample_bytree=0.9. Handles missing values natively. |
| **LightGBM** | Leaf-wise boosting | 300 trees, 31 leaves, faster than XGBoost. Uses GOSS and EFB for efficiency. |
| **KNN** | Instance-based | Euclidean distance with StandardScaler pipeline. k=5 (tunable). Lazy learner - no training. |

## Feature Selection Methods

Eight methods are compared in the prototype:

1. **SHAP** - SHAP-based feature importance (TreeExplainer)
2. **LIME** - Local Interpretable Model-agnostic Explanations
3. **Permutation** - Permutation importance with CV
4. **Mutual Information** - m=3 neighbors MI score
5. **RFE** - Recursive Feature Elimination (step=1)
6. **RF Gini** - Random Forest Gini importance
7. **SLA-FS_rankavg** - SLA-FS rank average (MODIFICATION)
8. **SLA-FS_scoreblend** - SLA-FS score blend (MODIFICATION)
9. **SLA-FS_intersection** - SLA-FS intersection (REPRODUCTION)
10. **Adaptive Selection** - F1(k) ≥ F1(full) - ε with ε ∈ {0.01, 0.02, 0.05}, patience=3

## Adaptive Feature Selection

The core contribution: find the smallest subset of features k such that:

```
F1(k) ≥ F1(full) - ε
```

where patience=3 (3 consecutive subset sizes must satisfy the tolerance before acceptance).

Tested ε values: 0.01 (strictest), 0.02 (default), 0.05 (lenient)

Results in feature reduction: typically 60-80% fewer features with ≤0.02 F1 loss.

## API Endpoints

### GET /
Root endpoint with API information

### GET /health
Health check

### GET /config
Returns the ExperimentConfig as JSON

### GET /environment
Returns runtime environment (library versions, etc.)

### POST /download-dataset
Download and verify the Mendeley dataset (sha256: 21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c)

### POST /preprocess
Run data preprocessing and train/test split (60/20/20 stratified)

### POST /train-models
Train all models with hyperparameter tuning on training split

### POST /feature-selection
Run all feature selection methods and return rankings

### POST /adaptive-selection
Run adaptive feature selection with tolerance-based subset selection

### POST /final-evaluation
Evaluate selected configurations on held-out test set (single fit, no data leakage)

### POST /explain-instance
LIME explanation of a single prediction instance

### POST /shap-summary
SHAP summary statistics and top features

## Deployment

### Backend (FastAPI + Uvicorn)

```bash
cd phishing-detection/prototype/backend
pip install uvicorn fastapi "xgboost<2.0" lightgbm shap lime pandas numpy scikit-learn
uvicorn main:app --host 127.0.0.1 --port 8000
```

### Frontend (React + CDN)

```bash
cd phishing-detection/prototype/frontend
# Open index.html in browser or serve with any static server
# The app connects to http://127.0.0.1:8000 by default
```

### Combined (using this repo's main.py)

The main `phishing-detection/main.py` already includes a `--stage all` command that runs the full pipeline. The prototype API endpoints are additive - they can be run alongside the main research pipeline.

## Sample Input

The prototype includes a sample CSV from the research dataset at:
`phishing-detection/data/sample_url_analysis.csv`

Contains URL features for phishing detection demo.

## Key Design Rules (from main.py)

1. **Test split created once** and only touched by final evaluation
2. **Hyper-parameters tuned once** on training split, reused for all subsets
3. **No output written** unless computed during the run
4. **All randomness driven by cfg.seed** (default: 42)
5. **Nothing seen by selectors** that ever touches the test split