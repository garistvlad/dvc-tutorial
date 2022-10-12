# preprocess.py
from pathlib import Path
from typing import List

import pandas as pd

FOLDERS_TO_LABELS = {
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def get_files_and_labels(source_path: Path):
    """Load all JPEG files recursively from Path provided"""
    images = []
    labels = []
    for image_path in source_path.rglob("*/*.JPEG"):
        filename = image_path.absolute()
        folder = image_path.parent.name

        if folder in FOLDERS_TO_LABELS:
            images.append(filename)
            labels.append(FOLDERS_TO_LABELS[folder])

    return images, labels


def save_as_csv(filenames: List, labels: List, csv_filepath: Path):
    """Save image names and labels to .csv"""
    df = pd.DataFrame({"filename": filenames, "label": labels})
    df.to_csv(csv_filepath)


def main(repo_path: Path):
    """preprocessing pipeline"""
    data_path = repo_path / "data"

    train_path = data_path / "raw/train"
    train_files, train_labels = get_files_and_labels(train_path)

    test_path = data_path / "raw/val"
    test_files, test_labels = get_files_and_labels(test_path)

    processed_dir = data_path / "processed"
    save_as_csv(train_files, train_labels, processed_dir / "train.csv")
    save_as_csv(test_files, test_labels, processed_dir / "test.csv")


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent  # . > src > dvc-tutorial
    main(repo_path)
