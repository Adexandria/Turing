from abc import ABC, abstractmethod
import os
import shutil

from loguru import logger
import mlflow
from numpy import ndarray


class BaseModel(ABC):
    """
    Abstract base class for training models.
    Subclasses should define the model and implement specific logic
    for training, evaluation, and model persistence.
    """

    def __init__(self, language, path=None):
        """
        Initialize the trainer.

        Args:
            language (str): Language for the model.
            path (str, optional): Path to load a pre-trained model. Defaults to None.
                                    If None, a new model is initialized.
        """

        self.language = language
        self.model = None
        if path:
            self.load(path)
        else:
            self.setup_model()

    @abstractmethod
    def setup_model(self):
        """
        Initialize or build the model.
        Called in __init__ of subclass.
        """
        pass

    @abstractmethod
    def train(self, X_train, y_train) -> dict[str,any]:
        """
        Main training logic for the model.

        Args:
            X_train: Input training data.
            y_train: True labels for training data.
        """
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test) -> dict[str,any]:
        """
        Evaluation logic for the model.

        Args:
            X_test: Input test data.
            y_test: True labels for test data.
        """
        pass

    @abstractmethod
    def predict(self, X) -> ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Input data for prediction.

        Returns:
            Predictions made by the model.
        """
        pass

    def save(self, path, model_name):
        """
        Save model and log to MLflow.

        Args:
            path (str): Path to save the model.
            model_name (str): Name to use when saving the model (without extension).
        """

        if self.model is None:
            raise ValueError("Model is not trained. Cannot save uninitialized model.")

        complete_path = os.path.join(path, f"{model_name}_{self.language}")
        if os.path.exists(complete_path) and os.path.isdir(complete_path):
            shutil.rmtree(complete_path)
        mlflow.sklearn.save_model(self.model, complete_path)

        try:
            mlflow.log_artifact(complete_path)
        except Exception as e:
            logger.error(f"Failed to log model to MLflow: {e}")

        logger.info(f"Model saved to: {complete_path}")

    def load(self, model_path):
        """
        Load model from specified local path or mlflow model URI.

        Args:
            model_path (str): Path to load the model from (local or mlflow URI).
        """

        self.model = mlflow.sklearn.load_model(model_path)
        logger.info(f"Model loaded from: {model_path}")
        
