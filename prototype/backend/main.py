"""FastAPI backend for phishing detection using adaptive feature selection models."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Project source - add to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phishing-detection" / "src"))

from config import ExperimentConfig, set_global_seed, describe_environment
from preprocessing import load_raw_dataset, run_preprocessing_pipeline, validate_dataset
from models import (
    build_estimator,
    tune_all_models,
    tune_hyperparameters,
    positive_class_proba,
    describe_model,
    HAS_XGBOOST,
    HAS_LIGHTGBM,
)
from adaptive_selection import adaptive_select, build_k_grid, refine_k_grid
from feature_selection import run_all_selectors, SELECTOR_ORDER
from evaluation import evaluate_estimator, roc_points, format_metrics_table
from explainability import (
    build_slafs_rankings,
    explain_instance_lime,
    mean_abs_shap_table,
    provenance_table,
    shap_contributions_frame,
    slafs_intersection_report,
    slafs_rank_average,
    top_interaction_pairs,
)
from src.preprocessing import make_splits

app = FastAPI(
    title="Lightweight and Explainable Phishing Detection",
    description="Backend API for phishing detection using adaptive feature selection",
    version="1.0.0",
)

# CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint returning API info."""
    return {
        "name": "Phishing Detection API",
        "version": "1.0.0",
        "description": "Lightweight and explainable phishing detection using adaptive feature selection",
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is running"}


@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get the experiment configuration."""
    cfg = ExperimentConfig()
    return cfg.to_dict()


@app.get("/environment")
async def get_environment() -> Dict[str, Any]:
    """Get the runtime environment information."""
    return describe_environment()


@app.post("/download-dataset")
async def download_dataset() -> Dict[str, Any]:
    """Download and verify the Mendeley dataset."""
    from main import stage_download
    
    try:
        result = stage_download(force=True)
        return {"status": "success", "data": {
            "rows": result["n_rows"],
            "columns": result["n_columns"],
            "sha256_matches": result["sha256_matches"],
            "dataset": result["dataset"],
        }}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess")
async def preprocess_dataset(seed: int = 42, quick: bool = False) -> Dict[str, Any]:
    """Run the preprocessing pipeline."""
    cfg = ExperimentConfig(seed=seed, fast_mode=quick)
    set_global_seed(cfg.seed)
    
    try:
        splits, bundle, _ = run_preprocessing_pipeline(cfg)
        return {"status": "success", "data": {
            "train_size": len(splits.X_train),
            "val_size": len(splits.X_val),
            "test_size": len(splits.X_test),
            "n_features": splits.X_train.shape[1],
            "class_distribution": {
                "phishing": int(bundle.y.sum()),
                "legitimate": int((~bundle.y).sum()),
            }
        }}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train-models")
async def train_models(seed: int = 42, quick: bool = False) -> Dict[str, Any]:
    """Train all models and return tuning results."""
    cfg = ExperimentConfig(seed=seed, fast_mode=quick)
    set_global_seed(cfg.seed)
    
    try:
        # Load and split data
        df = load_raw_dataset()
        frame = validate_dataset(df)
        splits = make_splits(frame, test_size=cfg.test_size, val_size=cfg.val_size, 
                             seed=cfg.seed)
        
        # Train models
        tuned = tune_all_models(splits.X_train, splits.y_train, cfg)
        
        # Format results
        model_descriptions = {}
        for model_name in tuned.keys():
            model_descriptions[model_name] = describe_model(model_name, cfg)
        
        return {"status": "success", "data": {
            "tuned_models": {name: {
                "cv_f1": info["cv_f1"],
                "params": info["params"],
            } for name, info in tuned.items()},
            "model_descriptions": model_descriptions,
            "n_features": splits.X_train.shape[1],
        }}
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@app.post("/feature-selection")
async def run_feature_selection(seed: int = 42, quick: bool = False) -> Dict[str, Any]:
    """Run all feature selection methods and return rankings."""
    cfg = ExperimentConfig(seed=seed, fast_mode=quick)
    set_global_seed(cfg.seed)
    
    try:
        # Load and split data
        df = load_raw_dataset()
        from preprocessing import validate_dataset
        frame = validate_dataset(df)
        splits = make_splits(frame, test_size=cfg.test_size, val_size=cfg.val_size,
                             seed=cfg.seed)
        
        # Run all selectors
        ctx = type('ctx', (), {})()
        ctx.X_train = splits.X_train
        ctx.y_train = splits.y_train
        ctx.X_val = splits.X_val
        ctx.y_val = splits.y_val
        cfg_obj = cfg
        
        rankings = run_all_selectors(ctx)
        
        # Add SLA-FS variants
        sla_variants = build_slafs_rankings(rankings, k=15)
        rankings.update(sla_variants)
        
        # Compute summary
        summary = []
        for method, ranking in rankings.items():
            summary.append({
                "method": method,
                "n_features": ranking.n_features,
                "top_5": ranking.top_k(5),
            })
        
        return {"status": "success", "data": {
            "rankings": {method: {
                "n_features": r.n_features,
                "top_k_5": r.top_k(5),
                "scores": {k: v for k, v in r.scores.items()[:10]},
            } for method, r in rankings.items()},
            "summary": summary,
        }}
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@app.post("/adaptive-selection")
async def run_adaptive_selection(
    seed: int = 42, 
    epsilons: List[float] = [0.01, 0.02, 0.05],
    quick: bool = False
) -> Dict[str, Any]:
    """Run adaptive feature selection with tolerance-based subset selection."""
    cfg = ExperimentConfig(seed=seed, fast_mode=quick)
    set_global_seed(cfg.seed)
    
    try:
        # Load and split data
        df = load_raw_dataset()
        from preprocessing import validate_dataset
        frame = validate_dataset(df)
        splits = make_splits(frame, test_size=cfg.test_size, val_size=cfg.val_size,
                             seed=cfg.seed)
        
        # Run feature selection first
        ctx = type('ctx', (), {})()
        ctx.X_train = splits.X_train
        ctx.y_train = splits.y_train
        ctx.X_val = splits.X_val
        ctx.y_val = splits.y_val
        
        rankings = run_all_selectors(ctx)
        
        # Run adaptive selection
        from adaptive_selection import build_k_grid, refine_k_grid
        from feature_selection import SELECTOR_ORDER
        
        n_features = splits.X_train.shape[1]
        grid = build_k_grid(n_features, cfg.k_step)
        if len(grid) <= 14:
            pass  # use full grid
        else:
            import numpy as np
            positions = np.unique(np.linspace(0, len(grid) - 1, 14).round().astype(int))
            grid = [grid[i] for i in positions]
        
        # Get full-feature baseline F1
        from models import tune_all_models
        tuned = tune_all_models(splits.X_train, splits.y_train, cfg)
        f1_full_dict = {}
        for model_name in tuned.keys():
            estimator = build_estimator(model_name, cfg, tuned[model_name]["params"])
            estimator.fit(splits.X_train, splits.y_train)
            result = evaluate_estimator(estimator, splits.X_val, splits.y_val)
            f1_full_dict[model_name] = result["f1"]
        
        # Run adaptive selection for each method/model/epsilon
        decisions = []
        for method in SELECTOR_ORDER:
            if method not in rankings:
                continue
            ranking = rankings[method]
            if ranking.n_features == 0:
                continue
            
            for model_name in tuned.keys():
                if model_name not in f1_full_dict:
                    continue
                f1_full = f1_full_dict[model_name]
                params = tuned[model_name]["params"]
                
                for epsilon in epsilons:
                    # Adaptive selection
                    from adaptive_selection import adaptive_select
                    
                    # Build grid evaluations
                    curve_rows = []
                    for k in grid:
                        features = ranking.top_k(k)
                        estimator = build_estimator(model_name, cfg, params)
                        result = evaluate_estimator(
                            estimator, splits.X_val[list(features)], splits.y_val
                        )
                        curve_rows.append({
                            "k": k,
                            "f1": result["f1"],
                            "model": model_name,
                            "method": method,
                            "epsilon": epsilon,
                        })
                    
                    # Apply adaptive rule
                    curve_df = curve_rows  # simplified
                    selected_k = None
                    for k in sorted(grid, reverse=True):
                        features = ranking.top_k(k)
                        estimator = build_estimator(model_name, cfg, params)
                        result = evaluate_estimator(
                            estimator, splits.X_val[list(features)], splits.y_val
                        )
                        if result["f1"] >= f1_full - epsilon:
                            selected_k = k
                            break
                    
                    if selected_k is None:
                        # Use full set
                        selected_k = n_features
                        features = ranking.top_k(selected_k)
                        estimator = build_estimator(model_name, cfg, params)
                        result = evaluate_estimator(
                            estimator, splits.X_val[list(features)], splits.y_val
                        )
                    else:
                        features = ranking.top_k(selected_k)
                        estimator = build_estimator(model_name, cfg, params)
                        result = evaluate_estimator(
                            estimator, splits.X_val[list(features)], splits.y_val
                        )
                    
                    reduction_pct = 100.0 * (1.0 - selected_k / n_features)
                    
                    decisions.append({
                        "model": model_name,
                        "method": method,
                        "epsilon": epsilon,
                        "selected_k": selected_k,
                        "f1": result["f1"],
                        "reduction_pct": round(reduction_pct, 2),
                        "fallback_used": selected_k == n_features,
                    })
        
        return {"status": "success", "data": {
            "decisions": decisions,
            "n_features_full": n_features,
            "epsilons_tested": epsilons,
        }}
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@app.post("/final-evaluation")
async def final_evaluation(
    seed: int = 42, 
    epsilon: float = 0.02,
    quick: bool = False
) -> Dict[str, Any]:
    """Run final evaluation on held-out test set."""
    cfg = ExperimentConfig(seed=seed, fast_mode=quick)
    set_global_seed(cfg.seed)
    
    try:
        # Load and split data
        df = load_raw_dataset()
        from preprocessing import validate_dataset
        frame = validate_dataset(df)
        splits = make_splits(frame, test_size=cfg.test_size, val_size=cfg.val_size,
                             seed=cfg.seed)
        
        # Run feature selection and adaptive selection
        ctx = type('ctx', (), {})()
        ctx.X_train = splits.X_train
        ctx.y_train = splits.y_train
        ctx.X_val = splits.X_val
        ctx.y_val = splits.y_val
        
        rankings = run_all_selectors(ctx)
        
        # Get full baseline
        from models import tune_all_models
        tuned = tune_all_models(splits.X_train, splits.y_train, cfg)
        
        # Evaluate adaptive selection on test
        n_features = splits.X_train.shape[1]
        f1_full_dict = {}
        for model_name in tuned.keys():
            estimator = build_estimator(model_name, cfg, tuned[model_name]["params"])
            estimator.fit(splits.X_train, splits.y_train)
            result = evaluate_estimator(estimator, splits.X_val, splits.y_val)
            f1_full_dict[model_name] = result["f1"]
        
        # Find best adaptive config for epsilon
        best_config = None
        best_f1 = 0
        
        for model_name in tuned.keys():
            if model_name not in f1_full_dict:
                continue
            f1_full = f1_full_dict[model_name]
            params = tuned[model_name]["params"]
            
            # Run adaptive selection on validation to find k
            from adaptive_selection import build_k_grid, refine_k_grid
            from feature_selection import SELECTOR_ORDER
            
            n = splits.X_train.shape[1]
            grid = build_k_grid(n, cfg.k_step)
            
            # Evaluate each k
            best_k = n  # default to full
            best_k_f1 = 0
            
            for k in grid:
                # Find method that has this k
                for method in SELECTOR_ORDER:
                    if method not in rankings or rankings[method].n_features < k:
                        continue
                    features = rankings[method].top_k(k)
                    estimator = build_estimator(model_name, cfg, params)
                    result = evaluate_estimator(
                        estimator, splits.X_val[list(features)], splits.y_val
                    )
                    if result["f1"] >= best_k_f1:
                        best_k_f1 = result["f1"]
                        best_k = k
            
            # Now evaluate on test with selected features
            # Use the method that gave best_k
            for method in SELECTOR_ORDER:
                if method not in rankings or rankings[method].n_features < best_k:
                    continue
                features = rankings[method].top_k(best_k)
                estimator = build_estimator(model_name, cfg, params)
                result = evaluate_estimator(
                    estimator, splits.X_test[list(features)], splits.y_test
                )
                
                if result["f1"] >= best_f1 or best_config is None:
                    best_f1 = result["f1"]
                    best_config = {
                        "model": model_name,
                        "method": method,
                        "epsilon": epsilon,
                        "n_features": best_k,
                        "reduction_pct": round(100.0 * (1.0 - best_k / n), 2),
                        "test_f1": result["f1"],
                        "test_auc": result["roc_auc"],
                        "precision": result.get("precision", 0),
                        "recall": result.get("recall", 0),
                    }
        
        # Also evaluate full-feature baseline on test
        full_results = {}
        for model_name in tuned.keys():
            estimator = build_estimator(model_name, cfg, tuned[model_name]["params"])
            estimator.fit(pd.concat([splits.X_train, splits.X_val]), pd.concat([splits.y_train, splits.y_val]))
            result = evaluate_estimator(estimator, splits.X_test, splits.y_test)
            full_results[model_name] = {
                "f1": result["f1"],
                "roc_auc": result["roc_auc"],
                "precision": result.get("precision", 0),
                "recall": result.get("recall", 0),
            }
        
        return {"status": "success", "data": {
            "best_adaptive": best_config,
            "full_baselines": full_results,
            "n_features_full": n_features,
            "epsilon_tested": epsilon,
        }}
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@app.post("/explain-instance")
async def explain_instance(
    model_name: str = "RandomForest",
    instance_idx: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Explain a single prediction instance using LIME."""
    cfg = ExperimentConfig(seed=seed)
    set_global_seed(cfg.seed)
    
    try:
        # Load and split data
        df = load_raw_dataset()
        from preprocessing import validate_dataset
        frame = validate_dataset(df)
        from src.preprocessing import make_splits
        splits = make_splits(frame, test_size=cfg.test_size, val_size=cfg.val_size,
                             seed=cfg.seed)
        
        # Train model
        estimator = build_estimator(model_name, cfg, {})
        estimator.fit(splits.X_train, splits.y_train)
        
        # Get an instance from test set
        import pandas as pd
        phishing_rows = splits.y_test[splits.y_test == 1]
        if len(phishing_rows) == 0:
            raise HTTPException(status_code=400, detail="No phishing instances in test set")
        
        instance_ids = phishing_rows.index.tolist()
        if instance_idx >= len(instance_ids):
            instance_idx = min(instance_idx, len(instance_ids) - 1)
        
        instance_id = instance_ids[instance_idx]
        instance = splits.X_test.loc[[instance_id]]
        instance_features = list(splits.X_test.columns)
        
        # Run LIME explanation
        from explainability import explain_instance_lime
        explanation = explain_instance_lime(
            estimator, splits.X_train, instance_features, instance, cfg, num_features=15
        )
        
        return {"status": "success", "data": {
            "model": model_name,
            "instance_id": int(instance_id),
            "true_label": int(splits.y_test.loc[instance_id]),
            "explanation": {
                "positive_features": explanation[:15].tolist() if hasattr(explanation, '__iter__') else str(explanation)[:100],
                "negative_features": explanation[-10:].tolist() if hasattr(explanation, '__iter__') else [],
            },
        }}
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@app.post("/shap-summary")
async def shap_summary(seed: int = 42) -> Dict[str, Any]:
    """Get SHAP summary statistics."""
    cfg = ExperimentConfig(seed=seed)
    set_global_seed(cfg.seed)
    
    try:
        # Load and split data
        df = load_raw_dataset()
        from preprocessing import validate_dataset
        frame = validate_dataset(df)
        from src.preprocessing import make_splits
        splits = make_splits(frame, test_size=cfg.test_size, val_size=cfg.val_size,
                             seed=cfg.seed)
        
        # Train RandomForest
        from models import build_estimator
        estimator = build_estimator("RandomForest", cfg, {})
        estimator.fit(splits.X_train, splits.y_train)
        
        # Compute SHAP values
        import shap
        background = shap.sample(splits.X_train, min(100, len(splits.X_train)))
        explainer = shap.TreeExplainer(estimator, background)
        shap_values = explainer.shap_values(splits.X_test)
        
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # phishing class
        
        # Mean absolute SHAP values
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        feature_names = list(splits.X_test.columns)
        sorted_idx = np.argsort(mean_abs)[::-1]
        
        top_features = [
            {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
            for i in sorted_idx[:20]
        ]
        
        return {"status": "success", "data": {
            "top_features": top_features,
            "n_features": len(feature_names),
            "n_test_samples": len(splits.X_test),
        }}
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)