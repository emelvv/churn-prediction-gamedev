import json

from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from config.settings import (
    ARTIFACTS_DIR,
    CATBOOST_PARAMS,
    FEATURES,
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TRAIN_TEST_SPLIT,
    VALIDATION_SPLIT,
)
from data_pipeline.dataset import build_feature_matrix, load_dataset


def find_best_threshold(y_true, probabilities) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    best = {
        "threshold": 0.5,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    for p, r, threshold in zip(precision[:-1], recall[:-1], thresholds):
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(p),
                "recall": float(r),
            }

    return best


def train_and_save_model() -> dict:
    df = load_dataset()
    X, y = build_feature_matrix(df)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

    valid_probabilities = model.predict_proba(X_valid)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]
    threshold_info = find_best_threshold(y_valid, valid_probabilities)

    metadata = {
        "features": FEATURES,
        "decision_threshold": threshold_info["threshold"],
        "validation_f1": threshold_info["f1"],
        "validation_precision": threshold_info["precision"],
        "validation_recall": threshold_info["recall"],
        "validation_auc": float(roc_auc_score(y_valid, valid_probabilities)),
        "test_auc": float(roc_auc_score(y_test, test_probabilities)),
        "catboost_params": CATBOOST_PARAMS,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata
