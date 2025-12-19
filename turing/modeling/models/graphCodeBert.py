import os
import shutil
import warnings

from loguru import logger
import numpy as np
from numpy import ndarray
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from turing.config import MODELS_DIR

from ..baseModel import BaseModel

warnings.filterwarnings("ignore")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Sigmoid function to convert logits to probabilities
    probs = 1 / (1 + np.exp(-predictions)) 
    
    # Apply threshold of 0.5 (becomes 1 if > 0.5, otherwise 0)
    preds = (probs > 0.5).astype(int)
    
    # Calculate F1 score (macro average for multi-label)
    f1 = f1_score(labels, preds, average='macro') 
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }



class GraphCodeBERTDataset(Dataset):
    """
    Internal Dataset class for CodeBERTa.
    """
    
    def __init__(self, encodings, labels=None, num_labels=None):
        """
        Initialize the InternalDataset.
        Args:
            encodings (dict): Tokenized encodings.
            labels (list or np.ndarray, optional): Corresponding labels.
            num_labels (int, optional): Total number of classes. Required for auto-converting indices to one-hot.
        """

        self.encodings = {key: torch.tensor(val) for key, val in encodings.items()}
        
        if labels is not None:
            if not isinstance(labels, (np.ndarray, torch.Tensor)):
                labels = np.array(labels)
            
            # Case A: labels are indices (integers)
            if num_labels is not None and (len(labels.shape) == 1 or (len(labels.shape) == 2 and labels.shape[1] == 1)):
                labels_flat = labels.flatten()
                
                # Create one-hot encoded matrix
                one_hot = np.zeros((len(labels_flat), num_labels), dtype=np.float32)
                
                # Set the corresponding index to 1
                valid_indices = labels_flat < num_labels
                one_hot[valid_indices, labels_flat[valid_indices]] = 1.0
                
                self.labels = torch.tensor(one_hot, dtype=torch.float)

            # Case B: labels are already vectors (e.g., One-Hot or Multi-Hot)
            else:
                self.labels = torch.tensor(labels, dtype=torch.float)
        else:
            self.labels = None


    def __getitem__(self, idx):
        """
        Retrieve item at index idx.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels (if available).
        """

        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item


    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            int: Length of the dataset.
        """

        return len(self.encodings['input_ids'])



class GraphCodeBERTClassifier(BaseModel):
    """
    HuggingFace implementation of BaseModel for Code Comment Classification.
    Uses GraphCodeBERT for efficient inference.
    """

    def __init__(self, language, path=None):
        """
        Initialize the GraphCodeBERT model with configuration parameters.

        Args:
            language (str): Language for the model.
            path (str, optional): Path to load a pre-trained model. Defaults to None.
        """
        
        self.params = {
            "model_name_hf": "microsoft/GraphCodeBERT-base",
            "num_labels": 7 if language == "java" else 5 if language == "python" else 6,
            "max_length": 128,
            "epochs": 15,
            "batch_size_train": 16,
            "batch_size_eval": 64,
            "learning_rate": 1e-5,
            "weight_decay": 0.02,
            "train_size": 0.8,
            "early_stopping_patience": 3,
            "early_stopping_threshold": 0.005,
            "optimizer": "adamw_torch",
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        
        super().__init__(language, path)


    def setup_model(self):
        """
        Initialize the CodeBERTa tokenizer and model.
        """

        logger.info(f"Initializing {self.params['model_name_hf']} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.params["model_name_hf"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.params["model_name_hf"], 
            num_labels=self.params["num_labels"],
            problem_type="multi_label_classification",
            use_safetensors=True,
        ).to(self.device)
        logger.info("GraphCodeBERT model initialized.")


    def _tokenize(self, texts):
        """
        Helper to tokenize list of texts efficiently.

        Args:
            texts (list): List of text strings to tokenize.

        Returns:
            dict: Tokenized encodings.
        """
        
        safe_texts = []
        for t in texts:
            if t is None:
                safe_texts.append("")
            elif isinstance(t, (int, float)):
                if t != t: # NaN check
                    safe_texts.append("")
                else:
                    safe_texts.append(str(t))
            else:
                safe_texts.append(str(t))

        return self.tokenizer(
            safe_texts, 
            truncation=True, 
            padding=True, 
            max_length=self.params["max_length"]
        )


    def train(self, X_train, y_train) -> dict[str,any]:
        """
        Train the model using HF Trainer.

        Args:
            X_train (list): Training input texts.
            y_train (list or np.ndarray): Training labels.

        Returns:
            dict[str, any]: Dictionary of parameters used for training.
        """

        if self.model is None:
            raise ValueError("Model is not initialized. Call setup_model() before training.")

        # parameters of the model to log
        params = {k: v for k, v in self.params.items() if k != "model_name_hf" and k != "num_labels"}

        logger.info(f"Starting training for: {self.language.upper()}")
        
        # Prepare dataset (train/val split)
        train_encodings = self._tokenize(X_train)
        full_dataset = GraphCodeBERTDataset(train_encodings, y_train, num_labels=self.params["num_labels"])
        train_size = int(self.params["train_size"] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        # Temporary directory for checkpoints
        temp_ckpt_dir = os.path.join(MODELS_DIR, "temp_checkpoints")
        
        # Setup Trainer and training
        use_fp16 = torch.cuda.is_available()
        if not use_fp16:
            logger.info("Mixed Precision (fp16) disabled because CUDA is not available.")

        training_args = TrainingArguments(
            output_dir=temp_ckpt_dir,
            num_train_epochs=self.params["epochs"],
            per_device_train_batch_size=self.params["batch_size_train"],
            per_device_eval_batch_size=self.params["batch_size_eval"],
            learning_rate=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            logging_dir='./logs',
            logging_steps=50,
            fp16=use_fp16,
            optim=self.params["optimizer"],
            report_to="none",
            no_cuda=not torch.cuda.is_available() 
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.params["early_stopping_patience"],
                early_stopping_threshold=self.params["early_stopping_threshold"]
            )]
        )
        trainer.train()
        logger.info(f"Training for {self.language.upper()} completed.")
        
        # Remove temporary checkpoint directory
        if os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir)

        return params
    
    
    def evaluate(self, X_test, y_test) -> dict[str,any]:
        """
        Evaluate model on test data and return metrics.
        Handles automatic conversion of y_test to match multi-label prediction shape.

        Args:
            X_test (list): Input test data.
            y_test (list or np.ndarray): True labels for test data.

        Returns:
            dict[str, any]: Dictionary of evaluation metrics.
        """
        
        # Obtain predictions
        y_pred = self.predict(X_test)

        # Convert y_test to numpy array if needed
        if not isinstance(y_test, (np.ndarray, torch.Tensor)):
            y_test_np = np.array(y_test)
        elif isinstance(y_test, torch.Tensor):
            y_test_np = y_test.cpu().numpy()
        else:
            y_test_np = y_test

        num_labels = self.params["num_labels"]
        is_multilabel_pred = (y_pred.ndim == 2 and y_pred.shape[1] > 1)
        is_flat_truth = (y_test_np.ndim == 1) or (y_test_np.ndim == 2 and y_test_np.shape[1] == 1)

        if is_multilabel_pred and is_flat_truth:
            # Create a zero matrix
            y_test_expanded = np.zeros((y_test_np.shape[0], num_labels), dtype=int)
            
            # Flatten y_test for iteration
            indices = y_test_np.flatten()
            
            # Use indices to set the correct column to 1
            for i, label_idx in enumerate(indices):
                idx = int(label_idx)
                if 0 <= idx < num_labels:
                    y_test_expanded[i, idx] = 1
            
            y_test_np = y_test_expanded

        # Generate classification report
        report = classification_report(y_test_np, y_pred, zero_division=0)
        print("\n" + "=" * 50)
        print("CLASSIFICATION REPORT")
        print(report)
        print("=" * 50 + "\n")

        metrics = {
            "accuracy": accuracy_score(y_test_np, y_pred),
            "precision": precision_score(y_test_np, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_test_np, y_pred, average="macro", zero_division=0),
            "f1_score": f1_score(y_test_np, y_pred, average="macro"),
        }
        logger.info(
            f"Evaluation completed — Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}"
        )

        return metrics


    def predict(self, X) -> ndarray:
        """
        Make predictions for Multi-Label classification.
        Returns Binary Matrix (Multi-Hot) where multiple classes can be 1.

        Args:
            X (list): Input texts for prediction.

        Returns:
            np.ndarray: Multi-Hot Encoded predictions (e.g., [[0, 1, 1, 0], ...])
        """

        if self.model is None:
            raise ValueError("Model is not trained. Call train() or load() before prediction.")

        # Set model to evaluation mode
        self.model.eval()
        
        encodings = self._tokenize(X)
        # Pass None as labels because we are in inference
        dataset = GraphCodeBERTDataset(encodings, labels=None)
        
        use_fp16 = torch.cuda.is_available()
        
        training_args = TrainingArguments(
            output_dir="./pred_temp", 
            per_device_eval_batch_size=self.params["batch_size_eval"],
            fp16=use_fp16,
            report_to="none",
            no_cuda=not torch.cuda.is_available()
        )
        
        trainer = Trainer(model=self.model, args=training_args)
        output = trainer.predict(dataset)
        
        # Clean up temporary prediction directory
        if os.path.exists("./pred_temp"):
            shutil.rmtree("./pred_temp")
        
        # Convert logits to probabilities
        logits = output.predictions
        probs = 1 / (1 + np.exp(-logits))
        
        # Apply a threshold of 0.5 (if prob > 0.5, predict 1 else 0)
        preds_binary = (probs > 0.5).astype(int)
        
        return preds_binary
    

    def save(self, path, model_name):
        """
        Save model locally.

        Args:
            path (str): Directory path to save the model.
            model_name (str): Name for the saved model.
        """

        if self.model is None:
            raise ValueError("Model is not trained. Cannot save uninitialized model.")

        # Local Saving
        complete_path = os.path.join(path, f"{model_name}_{self.language}")
        
        # Remove existing directory if it exists
        if os.path.exists(complete_path) and os.path.isdir(complete_path):
            shutil.rmtree(complete_path)
        
        # Save model and tokenizer
        logger.info(f"Saving model to: {complete_path}")
        self.model.save_pretrained(complete_path)
        self.tokenizer.save_pretrained(complete_path)
        logger.info("Model saved locally.")


    def load(self, model_path):
        """
        Load model from a local path.

        Args:
            model_path (str): Local path to load the model from.
        """

        logger.info(f"Loading model from: {model_path}")

        # Loading from local path
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path not found: {model_path}")

            # Load tokenizer and model from local path
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            ).to(self.device)
            logger.info("Model loaded from local path successfully.")

        except Exception as e:
            logger.error(f"Failed to load model from local path: {e}")
            raise e

        # Set model to evaluation mode
        self.model.eval()