import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.trainer import train_and_save_model
from config.settings import METADATA_PATH, MODEL_PATH


if __name__ == "__main__":
    metadata = train_and_save_model()
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(f"Decision threshold: {metadata['decision_threshold']:.4f}")
    print(f"Validation F1: {metadata['validation_f1']:.4f}")
    print(f"Validation AUC: {metadata['validation_auc']:.4f}")
    print(f"Test AUC: {metadata['test_auc']:.4f}")
    print(f"Test PR-AUC: {metadata['test_pr_auc']:.4f}")
    print(f"Test Recall: {metadata['test_recall']:.4f}")
