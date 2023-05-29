"""classification_model.py

Base class for modelling and benchmarking text classification.
"""


import typing as t
import logging
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod
from joblib import dump, load
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from datetime import datetime
import time


if t.TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class ClassificationModel(ABC):
    """Base class for modelling and benchmarking text classification.

    Attributes:
        data (t.List[t.Tuple[str, str]]): List of (text, label) tuples.
        vectorizer (str): The vectorizer object.
        model_args (t.Dict[str, t.Any]): The model arguments.
        vectorizer_args (t.Dict[str, t.Any]): The vectorizer arguments.
        name (str): Name of the model.
        model (t.Type['BaseEstimator']): The model object.
    """

    _data: t.Union[t.Tuple[ArrayLike, ArrayLike], None]
    vectorizer: t.Union['CountVectorizer', 'TfidfVectorizer']

    def __init__(
                self,
                data: t.Optional[t.Tuple[ArrayLike, ArrayLike]] = None,
                vectorizer: t.Union[
                                    'CountVectorizer',
                                    'TfidfVectorizer',
                                    str] = "tfidf",
                model_args: t.Optional[t.Dict[str, t.Any]] = None,
                vectorizer_args: t.Optional[t.Dict[str, t.Any]] = None,
                name: t.Optional[str] = None,
                model: t.Union[
                        t.Type['BaseEstimator'],
                        'BaseEstimator',
                        None] = None):
        self.name = name

        self._data = data
        vectorizer_args = vectorizer_args or {
            "lowercase": True,
            "max_features": 1000,
            "ngram_range": (1, 2),
            "min_df": 0.02,
            "max_df": 0.98
        }
        if isinstance(vectorizer, str):
            if vectorizer == "count":
                self.vectorizer = CountVectorizer(**vectorizer_args)
            elif vectorizer == "tfidf":
                self.vectorizer = TfidfVectorizer(**vectorizer_args)
            else:
                raise ValueError(
                    "Invalid vectorizer, must be 'count' or 'tfidf'")
        elif isinstance(vectorizer, (CountVectorizer, TfidfVectorizer)):
            self.vectorizer = vectorizer
        else:
            raise ValueError("Invalid vectorizer, must be 'count' or 'tfidf'")
        model_args = model_args or {}
        if model is None:
            raise ValueError("Model must be specified")
        if isinstance(model, type):
            self.model = model(**model_args)
        else:
            self.model = model

    def set_data(self, data: t.Tuple[ArrayLike, ArrayLike]) -> None:
        """Set the data for the model.

        Args:
            data (t.Tuple[ArrayLike, ArrayLike]): The data.
        """
        self._data = data

    @property
    def data(self) -> t.Tuple[ArrayLike, ArrayLike]:
        """Get the data for the model.

        Returns:
            t.Tuple[ArrayLike, ArrayLike]: The data.
        """
        if self._data is None:
            raise ValueError("Data not set")
        return self._data

    @abstractmethod
    def train(self, x: ArrayLike, y: ArrayLike) -> None:
        """Train the model.

        Args:
            x (ArrayLike): List of text samples.
            y (ArrayLike): List of labels.
        """
        pass

    @abstractmethod
    def predict(self, x: ArrayLike) -> ArrayLike:
        """Predict labels for the given text samples.

        Args:
            x (ArrayLike): List of text samples.

        Returns:
            ArrayLike: List of predicted labels.
        """
        pass

    @abstractmethod
    def evaluate(self, x: ArrayLike, y: ArrayLike) -> t.Dict[str, float]:
        """Evaluate the model on the given data.

        Args:
            x (ArrayLike): List of text samples.
            y (ArrayLike): List of labels.

        Returns:
            t.Dict[str, float]: The evaluation scores.
        """
        pass

    def save(self, path: Path) -> None:
        """Save the model to the given path.

        Args:
            path (str): Path to save the model to.
                If the path is a directory, the model will be saved to a file
                with the name of the model, and the current date and time.
        """
        if path.is_dir():
            datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = path / f"{datestr}_{self.name}.joblib"
            dump(self, model_path)
            logging.info(f"Saved model to {model_path}")

            vec_path = path / f"{datestr}_{self.name}_vectorizer.joblib"
            dump(self.vectorizer, vec_path)
            logging.info(f"Saved vectorizer to {vec_path}")

        elif path.is_file() and path.suffix == ".joblib":
            dump(self, path)
            logging.info(f"Saved model to {path}")

            vec_path = path.with_name(path.stem + "_vectorizer.joblib")
            dump(self.vectorizer, vec_path)
            logging.info(f"Saved vectorizer to {vec_path}")
        else:
            raise ValueError("Invalid path {path}")

    @classmethod
    def load(cls, path: Path) -> 'ClassificationModel':
        """Load the model from the given path.

        Args:
            path (Path): Path to load the model from.

        Returns:
            ClassificationModel: the model instance with
                the classifier and vectorizer loaded.
        """
        if path.is_file() and path.suffix == ".joblib":
            logging.info(f"Loading model from {path}")
            model = load(path)
            if not isinstance(model, BaseEstimator):
                raise TypeError("Invalid model type")
            vec_path = path.with_name(path.stem + "_vectorizer.joblib")
            if vec_path.is_file():
                logging.info(f"Loading vectorizer from {vec_path}")
                vec = load(vec_path)
                if not isinstance(vec, (CountVectorizer, TfidfVectorizer)):
                    raise TypeError("Invalid vectorizer type")
            else:
                logging.info(
                    f"Vectorizer not found at {vec_path}, using default")
                vec = "count"
            clfm = cls(
                model=model,
                vectorizer=vec
            )
            return clfm

        raise ValueError("Invalid path {path}")

    def train_test_split(
                        self,
                        test_size: float = 0.2) -> t.Tuple[
                                                    ArrayLike,
                                                    ArrayLike,
                                                    ArrayLike,
                                                    ArrayLike]:
        """Split the data into training and testing sets.

        Args:
            test_size (float): The proportion of the data to use for testing.

        Returns:
            t.Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
                The training and testing data and labels.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            self.data[0],
            self.data[1],
            test_size=test_size)
        return x_train, x_test, y_train, y_test

    def run(self, test_size: float = 0.2) -> t.Tuple[
                                                t.Dict[str, float],
                                                t.Dict[str, float]]:
        """Run the model.

        Args:
            test_size (float): The proportion of the data to use for testing.

        Returns:
            t.Tuple[t.Dict[str, float], t.Dict[str, float]]:
                The training and testing scores.
        """
        logging.info(f"Running model {self.name}...")
        logging.info(f"Test size: {test_size:.2f}")
        logging.info(f"Model: {self.model_summary()}")
        logging.info(f"Vectorizer: {self.vectorizer_summary()}")
        x_train, x_test, y_train, y_test = self.train_test_split(test_size)
        start_time = time.time()

        logging.info("Training model, grab a coffee...")
        self.train(x_train, y_train)
        train_time = time.time() - start_time
        logging.info(f"Training time: {train_time:.2f}s")

        logging.info("Evaluating model...")
        start_time = time.time()
        train_score = self.evaluate(x_train, y_train)
        train_score["train_eval_time"] = time.time() - start_time
        train_score["train_time"] = train_time
        logging.info(f"Training score:\n{train_score['report']}")
        start_time = time.time()
        test_score = self.evaluate(x_test, y_test)
        test_score["test_eval_time"] = time.time() - start_time
        test_score["train_time"] = train_time
        logging.info(f"Testing score:\n{test_score['report']}")

        return train_score, test_score

    def model_summary(self) -> t.Dict[str, t.Any]:
        """Get a summary of the model parameters.

        Returns:
            t.Dict[str, t.Any]: The vectorizer parameters.
        """
        return {
            "model": self.model.__class__.__name__,
            "model_params": self.model.get_params()
        }

    def vectorizer_summary(self) -> t.Dict[str, t.Any]:
        """Get a summary of the vectorizer parameters.

        Returns:
            t.Dict[str, t.Any]: The vectorizer parameters.
        """
        return {
            "vectorizer": self.vectorizer.__class__.__name__,
            "vectorizer_params": self.vectorizer.get_params()
        }
