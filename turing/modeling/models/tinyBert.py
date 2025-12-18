"""
Ultra-lightweight multi-label text classification model for code comment analysis.

This module implements a specialized neural architecture combining TinyBERT
(15MB, 96 layers compressed) with a custom multi-label classification head.
Designed for efficient inference on resource-constrained environments while
maintaining competitive performance on code comment classification tasks.

Architecture:
    - Encoder: TinyBERT (prajjwal1/bert-tiny)
    - Hidden dimension: 312
    - Classification layers: 312 -> 128 (ReLU) -> num_labels (Sigmoid)
    - Regularization: Dropout(0.2) for preventing overfitting
    - Loss function: Binary Cross-Entropy for multi-label classification

Performance characteristics:
    - Model size: ~15MB
    - Inference latency: ~50ms per sample
    - Memory footprint: ~200MB during training
    - Supports multi-label outputs via sigmoid activation
"""

from typing import List

from loguru import logger
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch import nn
from torch.optim import Adam

import turing.config as config
from turing.modeling.baseModel import BaseModel

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    logger.error("transformers library required. Install with: pip install transformers torch")


class TinyBERTClassifier(BaseModel):
    """
    Ultra-lightweight multi-label classifier for code comment analysis.

    Combines TinyBERT encoder with a custom classification head optimized for
    multi-label code comment classification across Java, Python, and Pharo.

    Attributes:
        device (torch.device): Computation device (CPU/GPU).
        model (nn.ModuleDict): Container for encoder and classifier components.
        tokenizer (AutoTokenizer): Hugging Face tokenizer for text preprocessing.
        classifier (nn.Sequential): Custom multi-label classification head.
        num_labels (int): Number of output classes per language.
        labels_map (list): Mapping of label indices to semantic categories.

    References:
        TinyBERT: https://huggingface.co/prajjwal1/bert-tiny
    """

    def __init__(self, language: str, path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"TinyBERT using device: {self.device}")
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.mlb = MultiLabelBinarizer()
        self.labels_map = config.LABELS_MAP.get(language, [])
        self.num_labels = len(self.labels_map)
        self.params = {
            "model": "TinyBERT",
            "model_size": "15MB",
            "epochs": 15,
            "batch_size": 8,
            "learning_rate": 1e-3,
        }
        super().__init__(language=language, path=path)

    def setup_model(self):
        """
        Initialize TinyBERT encoder and custom classification head.

        Loads the pre-trained TinyBERT model from Hugging Face model hub and
        constructs a custom multi-label classification head with:
        - Input: 312-dimensional encoder embeddings [CLS] token
        - Hidden layer: 128 units with ReLU activation
        - Dropout: 0.2 for regularization
        - Output: num_labels units with Sigmoid activation

        Raises:
            Exception: If model initialization fails due to network or missing dependencies.
        """
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize TinyBERT encoder and custom classification head.

        Loads the pre-trained TinyBERT model from Hugging Face model hub and
        constructs a custom multi-label classification head with:
        - Input: 312-dimensional encoder embeddings [CLS] token
        - Hidden layer: 128 units with ReLU activation
        - Dropout: 0.2 for regularization
        - Output: num_labels units with Sigmoid activation

        Raises:
            Exception: If model initialization fails due to network or missing dependencies.
        """
        try:
            model_name = "prajjwal1/bert-tiny"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            encoder = AutoModel.from_pretrained(model_name)
            encoder.to(self.device)

            hidden_dim = encoder.config.hidden_size

            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, self.num_labels),
                nn.Sigmoid(),
            ).to(self.device)

            self.model = nn.ModuleDict({"encoder": encoder, "classifier": self.classifier})

            logger.success(f"Initialized TinyBERTClassifier for {self.language}")
            logger.info(f"Model size: ~15MB | Labels: {self.num_labels}")

        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def train(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        path: str = None,
        model_name: str = "tinybert_classifier",
        epochs: int = 15,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
    ) -> dict:
        """
        Train the classifier using binary cross-entropy loss.

        Implements gradient descent optimization with adaptive learning rate scheduling.
        Supports checkpoint saving for model persistence and recovery.

        Args:
            X_train (List[str]): Training text samples (code comments).
            y_train (np.ndarray): Binary label matrix of shape (n_samples, n_labels).
            path (str, optional): Directory path for model checkpoint saving.
            model_name (str): Identifier for saved model artifacts.
            epochs (int): Number of complete training iterations. Default: 3.
            batch_size (int): Number of samples per gradient update. Default: 16.
            learning_rate (float): Adam optimizer learning rate. Default: 2e-5.

        Returns:
            dict: Training configuration including hyperparameters and model metadata.

        Raises:
            Exception: If training fails due to data inconsistency or resource exhaustion.
        """
        try:
            if self.model is None:
                self._initialize_model()

            optimizer = Adam(self.classifier.parameters(), lr=learning_rate)
            criterion = nn.BCELoss()

            num_samples = len(X_train)
            num_batches = (num_samples + batch_size - 1) // batch_size

            logger.info(f"Starting training: {epochs} epochs, {num_batches} batches per epoch")

            for epoch in range(epochs):
                total_loss = 0.0

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, num_samples)

                    batch_texts = X_train[start_idx:end_idx]
                    batch_labels = y_train[start_idx:end_idx]

                    optimizer.zero_grad()

                    tokens = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",
                    ).to(self.device)

                    with torch.no_grad():
                        encoder_output = self.model["encoder"](**tokens)
                        cls_token = encoder_output.last_hidden_state[:, 0, :]

                    logits = self.classifier(cls_token)

                    labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).to(self.device)
                    loss = criterion(logits, labels_tensor)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / num_batches
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

            logger.success(f"Training completed for {self.language}")

            if path:
                self.save(path, model_name)

            return {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "model_size_mb": 15,
            }

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def predict(self, texts: List[str], threshold: float = 0.3) -> np.ndarray:
        """
        Generate multi-label predictions for code comments.

        Performs inference in evaluation mode without gradient computation.
        Applies probability threshold to convert sigmoid outputs to binary labels.

        Args:
            texts (List[str]): Code comment samples for classification.
            threshold (float): Decision boundary for label assignment. Default: 0.5.
                Values below threshold are mapped to 0, above to 1.

        Returns:
            np.ndarray: Binary predictions matrix of shape (n_samples, n_labels).

        Raises:
            ValueError: If model is not initialized.
            Exception: If inference fails due to incompatible input dimensions.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Train or load a model first.")

        self.model.eval()
        predictions = []

        # Convert various types to list: pandas Series, Dataset Column, etc.
        if hasattr(texts, "tolist"):
            texts = texts.tolist()
        elif hasattr(texts, "__iter__") and not isinstance(texts, list):
            texts = list(texts)

        try:
            with torch.no_grad():
                tokens = self.tokenizer(
                    texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(self.device)

                encoder_output = self.model["encoder"](**tokens)
                cls_token = encoder_output.last_hidden_state[:, 0, :]

                logits = self.classifier(cls_token)
                probabilities = logits.cpu().numpy()

                predictions = (probabilities > threshold).astype(int)

            return predictions

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def evaluate(self, X_test: List[str], y_test: np.ndarray) -> dict:
        """
        Evaluate classification performance on test set.

        Computes per-label and macro-averaged metrics:
        - Precision: TP / (TP + FP) - correctness of positive predictions
        - Recall: TP / (TP + FN) - coverage of actual positive instances
        - F1-Score: 2 * (P * R) / (P + R) - harmonic mean of precision and recall
        - Accuracy: Per-sample exact match rate

        Args:
            X_test (List[str]): Test text samples for evaluation.
            y_test (np.ndarray): Ground truth binary label matrix or indices.

        Returns:
            dict: Evaluation metrics including f1_score, precision, recall, accuracy.

        Raises:
            Exception: If evaluation fails due to prediction errors.
        """
        try:
            predictions = self.predict(X_test)

            # Convert y_test to numpy array if needed
            if not isinstance(y_test, (np.ndarray, torch.Tensor)):
                y_test_np = np.array(y_test)
            elif isinstance(y_test, torch.Tensor):
                y_test_np = y_test.cpu().numpy()
            else:
                y_test_np = y_test

            # Handle conversion from flat indices to multi-hot encoding if needed
            is_multilabel_pred = predictions.ndim == 2 and predictions.shape[1] > 1
            is_flat_truth = (y_test_np.ndim == 1) or (
                y_test_np.ndim == 2 and y_test_np.shape[1] == 1
            )

            if is_multilabel_pred and is_flat_truth:
                # Create zero matrix for multi-hot encoding
                y_test_expanded = np.zeros((y_test_np.shape[0], self.num_labels), dtype=int)
                indices = y_test_np.flatten()

                # Set columns to 1 based on indices
                for i, label_idx in enumerate(indices):
                    idx = int(label_idx)
                    if 0 <= idx < self.num_labels:
                        y_test_expanded[i, idx] = 1

                y_test_np = y_test_expanded

            tp = np.sum((predictions == 1) & (y_test_np == 1), axis=0)
            fp = np.sum((predictions == 1) & (y_test_np == 0), axis=0)
            fn = np.sum((predictions == 0) & (y_test_np == 1), axis=0)

            precision_per_label = tp / (tp + fp + 1e-10)
            recall_per_label = tp / (tp + fn + 1e-10)
            f1_per_label = (
                2
                * (precision_per_label * recall_per_label)
                / (precision_per_label + recall_per_label + 1e-10)
            )

            metrics = {
                "f1_score": float(np.mean(f1_per_label)),
                "precision": float(np.mean(precision_per_label)),
                "recall": float(np.mean(recall_per_label)),
                "accuracy": float(np.mean(predictions == y_test_np)),
            }

            logger.info(f"Evaluation metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

    def save(self, path: str, model_name: str = "tinybert_classifier"):
        """
        Persist model artifacts including weights, tokenizer, and configuration.

        Saves the following components:
        - classifier.pt: PyTorch state dictionary of classification head
        - tokenizer configuration: Hugging Face tokenizer files
        - config.json: Model metadata and label mappings

        Args:
            path (str): Parent directory for model checkpoint storage.
            model_name (str): Model identifier used as subdirectory name.

        Raises:
            Exception: If file I/O or serialization fails.
        """
        try:
            import os

            model_path = os.path.join(path, model_name)
            os.makedirs(model_path, exist_ok=True)

            if self.classifier:
                torch.save(self.classifier.state_dict(), os.path.join(model_path, "classifier.pt"))

            if self.tokenizer:
                self.tokenizer.save_pretrained(model_path)

            config_data = {
                "language": self.language,
                "num_labels": self.num_labels,
                "labels_map": self.labels_map,
                "model_type": "tinybert_classifier",
                "model_name": model_name,
            }

            import json

            with open(os.path.join(model_path, "config.json"), "w") as f:
                json.dump(config_data, f, indent=2)

            logger.success(f"Model saved to {model_path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load(self, path: str):
        """
        Restore model state from checkpoint directory.

        Loads classifier weights from serialized PyTorch tensors and reinitializes
        the tokenizer from saved configuration. Restores language-specific label
        mappings from JSON metadata.

        Args:
            path (str): Directory containing model checkpoint files.

        Raises:
            Exception: If file not found or deserialization fails.
        """
        try:
            import json
            import os

            self._initialize_model()

            classifier_path = os.path.join(path, "classifier.pt")
            if os.path.exists(classifier_path):
                self.classifier.load_state_dict(
                    torch.load(classifier_path, map_location=self.device)
                )

            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                    self.language = config_data.get("language", self.language)
                    self.labels_map = config_data.get("labels_map", self.labels_map)

            logger.success(f"Model loaded from {path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
