"""data.py

File I/O utilities for text classification.
"""

import logging
import typing as t
from numpy.typing import ArrayLike
import pandas as pd
from pathlib import Path
from datetime import datetime


def load_csv_data(path: Path) -> t.Tuple[ArrayLike, ArrayLike]:
    """Load the data from a CSV file.

    Args:
        path (Path): path to the CSV file

    Returns:
        t.Tuple[np.ndarray, np.ndarray]: a tuple of (X, y), where X is a
            NumPy array of text and y is a NumPy array of labels
    """
    if not csv_path_valid(path):
        raise ValueError(f"Invalid path to CSV file: {path}")
    logging.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    X = df["text"]
    y = df["label"]
    return X, y


def csv_path_valid(path: Path) -> bool:
    """Check if a path to a CSV file is valid.

    Args:
        path (Path): path to the CSV file

    Returns:
        bool: True if the path is valid, False otherwise
    """
    return path.exists() and path.is_file() and path.suffix == ".csv"


def save_model_report(
        path: Path,
        model_name: str,
        vectorizer_name: str,
        train_results: t.Dict[str, t.Any],
        test_results: t.Dict[str, t.Any],
        model_params: t.Dict[str, t.Any],
        vectorizer_params: t.Dict[str, t.Any]) -> None:
    """Save the model report to a CSV file.

    Args:
        path (Path): path to the CSV file
        model_name (str): name of the model
        vectorizer_name (str): name of the vectorizer
        train_results (t.Dict[str, t.Any]): training results
        test_results (t.Dict[str, t.Any]): testing results
    """

    if not path.is_dir():
        raise ValueError("Path must be a directory")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}-{vectorizer_name}-{ts}.csv"

    df = pd.DataFrame({
        "model": [model_name],
        "timestamp": [pd.Timestamp.now()],
        "vectorizer": [vectorizer_name],
        "train_accuracy": [train_results["accuracy"]],
        "train_precision": [train_results["precision"]],
        "train_recall": [train_results["recall"]],
        "train_f1": [train_results["f1"]],
        "test_accuracy": [test_results["accuracy"]],
        "test_precision": [test_results["precision"]],
        "test_recall": [test_results["recall"]],
        "test_f1": [test_results["f1"]],
        "model_params": [model_params],
        "vectorizer_params": [vectorizer_params],
        "train_metrics_report": [train_results["report"]],
        "test_metrics_report": [test_results["report"]]
    })

    logging.info(f"Saving model report to {path}")
    df.to_csv(path / filename, index=False)
    logging.info(f"Model report saved to {path / filename}")
