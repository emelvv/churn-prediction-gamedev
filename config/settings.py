from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "combined_users_features.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.cbm"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

FEATURES = [
    "events_per_session_avg",
    "gun_store_events",
    "reward_events",
    "ml_snapshots_non_empty",
    "playtime_events",
]
TARGET_COLUMN = "churn"

TRAIN_TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.25
RANDOM_STATE = 42

CATBOOST_PARAMS = {
    "iterations": 120,
    "learning_rate": 0.03,
    "depth": 3,
    "l2_leaf_reg": 25,
    "min_data_in_leaf": 120,
    "random_strength": 4,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "early_stopping_rounds": 20,
    "random_seed": RANDOM_STATE,
    "verbose": 50,
}
