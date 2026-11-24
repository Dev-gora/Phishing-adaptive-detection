"""HTTP API for the phishing-detection prototype.

Serves the static single-page UI and the analysis endpoints it calls. Every
number returned here is computed from the real dataset and real fitted models;
nothing is mocked.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parents[1]
FRONTEND_DIR = PROJECT_ROOT / "prototype" / "frontend" / "public"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from engine import MODEL_LABELS, MODEL_ORDER, engine as _engine_singleton, get_engine  # noqa: E402
import feature_dictionary  # noqa: E402

app = FastAPI(title="Lightweight & Explainable Phishing Detection", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- models
class AnalyzeRequest(BaseModel):
    url: str
    models: Optional[List[str]] = None


class CompareRequest(BaseModel):
    models: List[str] = Field(default_factory=lambda: ["RandomForest", "XGBoost", "KNN"])
    methods: List[str] = Field(default_factory=lambda: ["Full", "SHAP"])
    k: int = 12
    epsilon: float = 0.02


class FeatureRequest(BaseModel):
    url: str
    model: str = "RandomForest"


class ExplainRequest(BaseModel):
    url: Optional[str] = None
    model: str = "RandomForest"


# ---------------------------------------------------------------------------- helpers
def _validate_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="Enter a URL to analyse.")
    if len(value) > 2048:
        raise HTTPException(status_code=400, detail="URL is longer than 2048 characters.")
    candidate = value if "://" in value else "http://" + value
    try:
        parts = urlsplit(candidate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse the URL: {exc}") from exc
    if not parts.netloc or "." not in parts.netloc.split(":")[0] and parts.netloc.split(":")[0] != "localhost":
        raise HTTPException(
            status_code=400,
            detail="That does not look like a URL. Include a hostname, e.g. example.com/login.",
        )
    return value


def _check_models(models: Optional[List[str]], available: List[str]) -> List[str]:
    if not models:
        return [m for m in ("RandomForest", "XGBoost", "KNN") if m in available]
    unknown = [m for m in models if m not in available]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown or unavailable model(s): {', '.join(unknown)}")
    return models


@app.on_event("startup")
def _warm_engine() -> None:
    """Load the dataset and fit the cached models before serving requests.

    Doing this synchronously means the first interactive prediction is fast: the
    browser never sees a half-warm engine whose numbers would include model fits.
    """
    engine = get_engine()
    engine.models_ready.wait(timeout=600)
    engine.rankings_ready.wait(timeout=600)


# --------------------------------------------------------------------------- metadata
@app.get("/api/health")
def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "models": engine.available_models(),
        "url_features": len(engine.url_feature_names),
        "rankings_ready": engine.rankings_ready.is_set(),
        "init_seconds": engine.ready_s,
    }


@app.get("/api/models")
def models() -> dict:
    return get_engine().models_info()


@app.get("/api/dataset-info")
def dataset_info() -> dict:
    engine = get_engine()
    info = engine.dataset_info()
    info["feature_dictionary"] = feature_dictionary.dictionary(engine.bundle.X.columns)
    info["feature_dictionary_all"] = feature_dictionary.dictionary(engine.raw.columns)
    return info


@app.get("/api/feature-importance")
def feature_importance(limit: int = 20) -> dict:
    engine = get_engine()
    available = [m for m in engine.available_models() if m != "KNN"]
    payload = {"method": "mean absolute SHAP value", "models": {}}
    for name in available:
        try:
            payload["models"][name] = engine.global_shap(name, top=limit)
        except Exception as exc:
            payload["models"][name] = {"error": f"{type(exc).__name__}: {exc}"}
    payload["rankings"] = engine.ranking_table()
    return payload


# ------------------------------------------------------------------------ URL analysis
@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    engine = get_engine()
    url = _validate_url(request.url)
    chosen = _check_models(request.models, engine.available_models())
    return engine.predict_url(url, chosen)


@app.post("/api/feature-analysis")
def feature_analysis(request: FeatureRequest) -> dict:
    engine = get_engine()
    url = _validate_url(request.url)
    model = request.model if request.model in engine.available_models() else "RandomForest"
    payload = engine.predict_url(url, [model])
    return {
        "url": url,
        "model": model,
        "model_label": MODEL_LABELS.get(model, model),
        "features": payload["features"],
        "importance_source": payload.get("importance_source"),
        "shap": payload["shap"],
        "lime": payload["lime"],
        "prediction": payload["headline"],
        "external_not_retrieved": [
            "domain_age", "domain_registration_length", "dns_record", "web_traffic",
            "google_index", "page_rank", "statistical_report", "nb_redirection",
            "nb_external_redirection", "domain_in_brand", "brand_in_subdomain",
            "brand_in_path", "random_domain",
        ],
        "note": payload["notes"][0],
    }


@app.post("/api/explain")
def explain(request: ExplainRequest) -> dict:
    engine = get_engine()
    model = request.model if request.model in engine.available_models() else "RandomForest"
    payload: dict = {"model": model, "model_label": MODEL_LABELS.get(model, model)}

    try:
        payload["global"] = engine.global_shap(model, top=20)
    except Exception as exc:
        payload["global"] = {"error": str(exc)}

    if request.url:
        url = _validate_url(request.url)
        instance = engine.predict_url(url, [model])
        payload["instance"] = {
            "url": url,
            "prediction": instance["headline"],
            "shap": instance["shap"],
            "lime": instance["lime"],
            "contributions": {
                "positive": [r for r in instance["shap"]["rows"] if r["value"] > 0][:8],
                "negative": [r for r in instance["shap"]["rows"] if r["value"] < 0][:8],
            },
        }
    return payload


# --------------------------------------------------------------------------- compare
@app.post("/api/compare-models")
def compare(request: CompareRequest) -> dict:
    engine = get_engine()
    available = engine.available_models()
    chosen_models = _check_models(request.models, available)
    known = set(MODEL_ORDER)
    methods = [m for m in request.methods if m]
    if not methods:
        raise HTTPException(status_code=400, detail="Select at least one feature-selection method.")
    if not chosen_models:
        raise HTTPException(status_code=400, detail="Select at least one model.")

    max_k = len(engine.url_feature_names)
    k = max(1, min(int(request.k), max_k))
    epsilon = max(0.0, min(float(request.epsilon), 0.5))
    payload = engine.compare(chosen_models, methods, k, epsilon)
    payload["available_models"] = available
    payload["k_requested"] = k
    payload["max_k"] = max_k
    return payload


# ----------------------------------------------------------------------------- static
if FRONTEND_DIR.exists():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
