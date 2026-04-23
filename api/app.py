from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.settings import FEATURES, METADATA_PATH, MODEL_PATH
from inference.predictor import ModelBundle, load_model_bundle, predict_probabilities

PREDICTION_EXAMPLE = {
    "events_per_session_avg": 3.2,
    "gun_store_events": 1,
    "reward_events": 5,
    "ml_snapshots_non_empty": 2,
    "playtime_events": 7,
}
BATCH_PREDICTION_EXAMPLE = {
    "items": [
        PREDICTION_EXAMPLE,
        {
            "events_per_session_avg": 1.4,
            "gun_store_events": 0,
            "reward_events": 2,
            "ml_snapshots_non_empty": 1,
            "playtime_events": 3,
        },
    ]
}


class PredictionRequest(BaseModel):
    events_per_session_avg: float = Field(..., description="Average number of events per session", examples=[3.2])
    gun_store_events: float = Field(..., description="Number of gun store events", examples=[1])
    reward_events: float = Field(..., description="Number of reward events", examples=[5])
    ml_snapshots_non_empty: float = Field(..., description="Count of non-empty ML snapshots", examples=[2])
    playtime_events: float = Field(..., description="Number of playtime events", examples=[7])

    model_config = {"json_schema_extra": {"example": PREDICTION_EXAMPLE}}


class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest] = Field(..., description="List of feature rows for batch inference")

    model_config = {"json_schema_extra": {"example": BATCH_PREDICTION_EXAMPLE}}


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., examples=[0.7812])
    predicted_class: int = Field(..., examples=[1])
    decision_threshold: float = Field(..., examples=[0.5185])


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


app = FastAPI(
    title="Churn Inference Service",
    version="4.0.0",
    description="Inference API for the CatBoost churn model. Open `/docs` for Swagger UI and `/redoc` for ReDoc.",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

bundle: ModelBundle | None = None


def get_decision_threshold() -> float:
    if bundle is None:
        raise RuntimeError("Model bundle is not loaded")
    return bundle.decision_threshold


def make_prediction_response(probability: float) -> PredictionResponse:
    threshold = get_decision_threshold()
    return PredictionResponse(
        churn_probability=float(probability),
        predicted_class=int(probability >= threshold),
        decision_threshold=threshold,
    )


@app.on_event("startup")
def startup_event() -> None:
    global bundle
    bundle = load_model_bundle()


@app.get("/health", summary="Health check", description="Returns service status, artifact paths, feature list, and active decision threshold.")
def health() -> dict:
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "metadata_path": str(METADATA_PATH),
        "features": FEATURES,
        "decision_threshold": get_decision_threshold(),
    }


@app.get("/features", summary="List model features", description="Returns the exact feature order expected by the inference model.")
def features() -> dict:
    return {"features": FEATURES, "decision_threshold": get_decision_threshold()}


@app.post("/predict", response_model=PredictionResponse, summary="Predict churn for one user", description="Accepts one feature row and returns churn probability, predicted class, and decision threshold.")
def predict(request: PredictionRequest) -> PredictionResponse:
    probabilities = predict_probabilities(bundle, [request.model_dump()])
    return make_prediction_response(probabilities[0])


@app.post("/predict_batch", response_model=BatchPredictionResponse, summary="Predict churn for multiple users", description="Accepts multiple feature rows and returns predictions in the same order.")
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    if not request.items:
        raise HTTPException(status_code=400, detail="items must not be empty")

    probabilities = predict_probabilities(bundle, [item.model_dump() for item in request.items])
    predictions = [make_prediction_response(probability) for probability in probabilities]
    return BatchPredictionResponse(predictions=predictions)
