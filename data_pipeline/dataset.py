import pandas as pd

from config.settings import FEATURES, RAW_DATA_PATH, TARGET_COLUMN


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)


def validate_dataset(df: pd.DataFrame) -> None:
    missing_features = [feature for feature in FEATURES if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")


def build_feature_matrix(df: pd.DataFrame):
    validate_dataset(df)
    X = df[FEATURES].fillna(0)
    y = df[TARGET_COLUMN].astype(int)
    return X, y
