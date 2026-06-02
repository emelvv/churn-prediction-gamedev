from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "combined_users_features.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.cbm"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

FEATURES = [
    "events_total",
    "event_types_total",
    "active_days",
    "sessions_total",
    "events_per_session_avg",
    "session_duration_min_avg",
    "session_duration_min_median",
    "session_duration_min_max",
    "active_day_gap_avg",
    "active_day_gap_max",
    "days_since_install_min",
    "days_since_install_max",
    "last_event_to_month_end_days",
    "store_events",
    "gun_store_events",
    "ads_events",
    "heist_events",
    "lab_dungeon_events",
    "quest_events",
    "reward_events",
    "reward_money_events",
    "reward_itemtoken_events",
    "training_events",
    "playtime_events",
    "ml_snapshots_non_empty",
    "damage_lvl_delta",
    "health_lvl_delta",
    "regen_lvl_delta",
    "speed_lvl_delta",
    "crit_chance_lvl_delta",
    "crit_mult_lvl_delta",
    "player_dps_delta",
    "itemtoken_balance_delta",
    "hard_balance_last",
]
TARGET_COLUMN = "churn"

TRAIN_TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.25
RANDOM_STATE = 42

CATBOOST_PARAMS = {
    "iterations": 200,
    "learning_rate": 0.04,
    "depth": 4,
    "l2_leaf_reg": 20,
    "min_data_in_leaf": 120,
    "random_strength": 4,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "early_stopping_rounds": 30,
    "random_seed": RANDOM_STATE,
    "verbose": False,
}
