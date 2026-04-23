import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from config.settings import FEATURES, METADATA_PATH, MODEL_PATH

DEFAULT_THRESHOLD = 0.5


class ModelBundle:
    def __init__(self, model: CatBoostClassifier, metadata: dict):
        self.model = model
        self.metadata = metadata

    @property
    def decision_threshold(self) -> float:
        return float(self.metadata.get("decision_threshold", DEFAULT_THRESHOLD))


def load_model_bundle(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> ModelBundle:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    metadata = {"features": FEATURES, "decision_threshold": DEFAULT_THRESHOLD}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    return ModelBundle(model=model, metadata=metadata)


def predict_probabilities(bundle: ModelBundle, rows: list[dict]) -> list[float]:
    frame = pd.DataFrame(rows, columns=FEATURES).fillna(0)
    probabilities = bundle.model.predict_proba(frame)[:, 1]
    return [float(probability) for probability in probabilities]
