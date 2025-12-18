import warnings

from loguru import logger
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from ..baseModel import BaseModel

warnings.filterwarnings("ignore")


class RandomForestTfIdf(BaseModel):
    """
    Sklearn implementation of BaseModel with integrated Grid Search.
    Builds a TF-IDF + RandomForest pipeline for multi-output text classification.
    """

    def __init__(self, language, path=None):
        """
        Initialize the RandomForestTfIdf model with configuration parameters.

        Args:
            language (str): Language for the model.
            path (str, optional): Path to load a pre-trained model. Defaults to None.
                                    If None, a new model is initialized.
        """

        self.params = {"stop_words": "english", "random_state": 42, "cv_folds": 5}

        self.grid_params = {
            "clf__estimator__n_estimators": [50, 100, 200],
            "clf__estimator__max_depth": [None, 10, 20],
            "tfidf__max_features": [3000, 5000, 8000],
        }

        super().__init__(language, path)

    def setup_model(self):
        """
        Initialize the scikit-learn pipeline with TF-IDF vectorizer and RandomForest classifier.
        """

        base_estimator = RandomForestClassifier(
            random_state=self.params["random_state"], n_jobs=-1
        )

        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=(1, 2), stop_words=self.params["stop_words"]),
                ),
                ("clf", MultiOutputClassifier(base_estimator, n_jobs=-1)),
            ]
        )

        self.model = self.pipeline
        logger.info("Scikit-learn pipeline initialized.")

    def train(self, X_train, y_train) -> dict[str, any]:
        """
        Train the model using Grid Search to find the best hyperparameters.

        Args:
            X_train: Input training data.
            y_train: True labels for training data.
        """

        if self.model is None:
            raise ValueError(
                "Model pipeline is not initialized. Call setup_model() before training."
            )

        logger.info(f"Starting training for: {self.language.upper()}")
        logger.info("Performing Grid Search for best hyperparameters...")
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid=self.grid_params,
            cv=self.params["cv_folds"],
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        logger.success(f"Best params found: {grid_search.best_params_}")

        parameters_to_log = {
            "max_features": grid_search.best_params_["tfidf__max_features"],
            "n_estimators": grid_search.best_params_["clf__estimator__n_estimators"],
            "max_depth": grid_search.best_params_["clf__estimator__max_depth"],
        }

        self.model = grid_search.best_estimator_
        logger.success(f"Training for {self.language.upper()} completed.")

        return parameters_to_log

    def evaluate(self, X_test, y_test) -> dict[str, any]:
        """
        Evaluate model on test data and return metrics.

        Args:
            X_test: Input test data.
            y_test: True labels for test data.
        """

        y_pred = self.predict(X_test)

        report = classification_report(y_test, y_pred, zero_division=0)
        print("\n" + "=" * 50)
        print("CLASSIFICATION REPORT")
        print(report)
        print("=" * 50 + "\n")

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
        }

        logger.info(
            f"Evaluation completed — Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}"
        )
        return metrics

    def predict(self, X) -> ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Input data for prediction.

        Returns:
            Predictions made by the model.
        """

        if self.model is None:
            raise ValueError("Model is not trained. Call train() or load() before prediction.")

        return self.model.predict(X)
