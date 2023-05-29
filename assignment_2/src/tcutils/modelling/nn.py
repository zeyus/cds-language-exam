"""nn.py

Neural network model.
"""


import typing as t
from numpy.typing import ArrayLike
from .classification_model import ClassificationModel
from sklearn.neural_network import MLPClassifier
from sklearn import metrics


class NN(ClassificationModel):
    """Neural network model.

    Attributes:
        data (t.List[t.Tuple[str, str]]): List of (text, label) tuples.
        vectorizer (str): The vectorizer object.
        model_args (t.Dict[str, t.Any]): The model arguments.
        vectorizer_args (t.Dict[str, t.Any]): The vectorizer arguments.
    """

    model: MLPClassifier

    def __init__(
            self,
            *args,
            **kwargs):

        model = MLPClassifier
        if "model" in kwargs:
            model_type = type(kwargs["model"])

            if model_type is None:
                kwargs.pop("model")
            elif model_type is MLPClassifier:
                model = kwargs["model"]
                kwargs.pop("model")
            else:
                raise ValueError("Model must be a MLPClassifier object")

        if "model_args" not in kwargs:
            kwargs["model_args"] = {
                "early_stopping": True,
                "max_iter": 1000,
                "hidden_layer_sizes": (100,),
            }

        super().__init__(
            *args,
            name="NN",
            model=model,
            **kwargs)

    def train(self, x: ArrayLike, y: ArrayLike) -> None:
        """Train the model.

        Args:
            x (ArrayLike): List of text samples.
            y (ArrayLike): List of labels.
        """
        x = self.vectorizer.fit_transform(x)
        self.model.fit(x, y)

    def predict(self, x: ArrayLike) -> ArrayLike:
        """Predict the labels for a list of texts.

        Args:
            x (ArrayLike): List of text samples.

        Returns:
            ArrayLike: List of predicted labels.
        """
        x = self.vectorizer.transform(x)  # type: ignore
        return self.model.predict(x)  # type: ignore

    def evaluate(self, x: ArrayLike, y: ArrayLike) -> t.Dict[str, float]:
        """Evaluate the model.

        Args:
            x (ArrayLike): List of text samples.
            y (ArrayLike): List of labels.

        Returns:
            t.Dict[str, float]: Dictionary of evaluation metrics.
        """
        x = self.vectorizer.transform(x)
        y_pred = self.model.predict(x)
        return {
            "report": metrics.classification_report(y, y_pred),  # type: ignore
            "accuracy": metrics.accuracy_score(y, y_pred),  # type: ignore
            "precision": metrics.precision_score(
                y, y_pred, average="weighted"),  # type: ignore
            "recall": metrics.recall_score(
                y, y_pred, average="weighted"),  # type: ignore
            "f1": metrics.f1_score(
                y, y_pred, average="weighted")  # type: ignore
        }
