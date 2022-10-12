# evaluate.py
import joblib
import json
from pathlib import Path

from sklearn.metrics import accuracy_score

from train import load_data


def main(repo_path: Path):
    """Model evaluation pipeline"""
    test_csv_path = repo_path / "data/processed/test.csv"
    test_data, labels = load_data(test_csv_path)

    model_sgd = joblib.load(repo_path / "models/model_sgd.joblib")
    predictions = model_sgd.predict(test_data)

    accuracy = accuracy_score(labels, predictions)

    accuracy_path = repo_path / "metrics/accuracy.json"
    accuracy_path.write_text(json.dumps({"accuracy": accuracy}))


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
