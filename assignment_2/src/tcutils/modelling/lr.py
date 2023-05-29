"""lr.py

Linear regression model.
"""

import typing as t
from numpy.typing import ArrayLike
from .classification_model import ClassificationModel
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


class LR(ClassificationModel):
    """Linear regression model.

    Attributes:
        data (t.List[t.Tuple[str, str]]): List of (text, label) tuples.
        vectorizer (str): The vectorizer object.
        model_args (t.Dict[str, t.Any]): The model arguments.
        vectorizer_args (t.Dict[str, t.Any]): The vectorizer arguments.
    """

    model: LogisticRegression

    def __init__(
            self,
            *args,
            **kwargs):

        model = LogisticRegression
        if "model" in kwargs:
            model_type = type(kwargs["model"])

            if model_type is None:
                kwargs.pop("model")
            elif model_type is LogisticRegression:
                model = kwargs["model"]
                kwargs.pop("model")
            else:
                raise ValueError("Model must be a LogisticRegression object")

        super().__init__(
            *args,
            name="LR",
            model=model,
            **kwargs)

    def train(self, x: ArrayLike, y: ArrayLike) -> None:
        """Train the model.

        Args:
            x (ArrayLike): List of text samples.
            y (ArrayLike): List of labels.
        """
        super().train(x, y)
        x = self.vectorizer.fit_transform(x)
        self.model.fit(x, y)

    def predict(self, x: ArrayLike) -> ArrayLike:
        """Predict the labels for a list of texts.

        Args:
            x (ArrayLike): List of texts.

        Returns:
            ArrayLike: List of predicted labels.
        """
        x = self.vectorizer.transform(x)
        return self.model.predict(x)

    def evaluate(self, x: ArrayLike, y: ArrayLike) -> t.Dict[str, float]:
        """Evaluate the model.

        Args:
            x (ArrayLike): List of texts.
            y (ArrayLike): List of labels.

        Returns:
            t.Dict[str, float]: The evaluation metrics.
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
