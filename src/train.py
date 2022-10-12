# train.py
from typing import List

import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread_collection
from skimage.transform import resize
from sklearn.linear_model import SGDClassifier


def load_images(df: pd.DataFrame, column_name: str) -> List:
    """Load images from DataFrame column"""
    files = df[column_name].to_list()
    images = imread_collection(files)
    return images


def load_labels(df: pd.DataFrame, column_name: str) -> List:
    """Load labels from DataFrame column"""
    labels = df[column_name].to_list()
    return labels


def preprocess(image):
    """Flatten an Image"""
    resized = resize(image, (100, 100, 3))
    reshaped = resized.reshape((1, 30_000))
    return reshaped


def load_data(data_path: Path):
    """Load images based on CSV file and preprocess them"""
    df = pd.read_csv(data_path)
    labels = load_labels(df, column_name="label")
    raw_images = load_images(df, column_name="filename")
    processed_images = [preprocess(image) for image in raw_images]
    data = np.concatenate(processed_images, axis=0)
    return data, labels


def main(repo_path: Path):
    """Train pipeline"""
    train_csv_path = repo_path / "data/processed/train.csv"
    train_data, labels = load_data(train_csv_path)
    model_sgd = SGDClassifier(max_iter=500)
    model_sgd.fit(train_data, labels)
    joblib.dump(model_sgd, repo_path / "models/model_sgd.joblib")


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
