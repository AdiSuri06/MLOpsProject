import os
import time
from typing import List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
GIT_SHA = os.getenv("GIT_SHA", "local")

app = FastAPI(title="ML inference api", version=MODEL_VERSION)

_model = None
_load_error: Optional[str] = None
_load_time_ms: Optional[int] = None


class PredictRequest(BaseModel):
    features: conlist(float, min_length=4, max_length=4)


class PredictResponse(BaseModel):
    prediction: int
    model_version: str
    git_sha: str


def load_model() -> None:
    global _model, _load_error, _load_time_ms
    tinitial = time.time()
    try:
        _model = joblib.load(MODEL_PATH)
        _load_error = None
    except Exception as e:
        _model = None
        _load_error = str(e)
    _load_time_ms = int((time.time() - tinitial) * 1000)


@app.on_event("startup")
def _startup():
    load_model()


@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {_load_error}")
    return {"status": "ok", "model_version": MODEL_VERSION, "git_sha": GIT_SHA}


@app.get("/model-info")
def model_info():
    return {
        "model_path": MODEL_PATH,
        "model_version": MODEL_VERSION,
        "git_sha": GIT_SHA,
        "loaded": _model is not None,
        "load_error": _load_error,
        "load_time_ms": _load_time_ms,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException()
    return PredictResponse(prediction=int(_model.predict([req.features])[0]), model_version=MODEL_VERSION, git_sha=GIT_SHA)
