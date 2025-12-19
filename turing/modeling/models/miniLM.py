import os
import shutil
import random
import joblib
import mlflow
from peft import LoraConfig, TaskType, get_peft_model
from loguru import logger
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
import torch
import torch.nn as nn
import numpy as np
from numpy import ndarray
from sentence_transformers import SentenceTransformer 
from datasets import Dataset
from transformers import get_linear_schedule_with_warmup
from turing.modeling.models.MiniLMClassifierWrapper import MiniLMClassifierWrapper

from turing.modeling.baseModel import BaseModel

def drop_tokens(text, drop_prob=0.1):
    """
    Randomly drops tokens from the input text based on the specified drop probability.
    """

    tokens = text.split()
    if len(tokens) <= 3:
        return text
    
    return " ".join(
        t for t in tokens if random.random() > drop_prob
    )

def drop_tokens_batch(texts, drop_prob=0.1, apply_prob=0.3):
    """
    Apply token dropping augmentation to a batch of texts.
    """
    augmented = []
    for text in texts:
        if random.random() < apply_prob:
            augmented.append(drop_tokens(text, drop_prob))
        elif random.random() < 0.15:
            x = " ".join(text.split())
            augmented.append(x)
        else:
            augmented.append(text)
    return augmented


def finetune_miniLM(X_train, y_train, device,model_save_path="sentence-transformers/minilm.pt"):
    """
    Train MiniLM model with temporary classification head using java dataset only.

    Args:
        X_train: Input training data.
        y_train: True labels for training data.
    """
    encoder =  SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2').to(device)   
    peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    lora_alpha=16,
    bias="none",
    lora_dropout=0.1,
    ) 

    encoder[0].auto_model = get_peft_model(
    encoder[0].auto_model,
    peft_config
    )

    encoder[0].auto_model.print_trainable_parameters()

    y_train = np.array(y_train,dtype=np.float32)
    
    dataset = Dataset.from_dict({"text": X_train, "labels": y_train})

    split_set = dataset.train_test_split(test_size= 0.2, seed=42)
    train_set = split_set['train']
    eval_set = split_set['test']

    epoch = 10
    batch_size = 32
    total_steps = len(train_set) // batch_size * epoch  
    warm_up_steps = int(0.1 * total_steps)

    y_train = np.array(y_train,dtype=np.float32)
    
    classifier = nn.Sequential(
                nn.Linear(encoder.get_sentence_embedding_dimension(), 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, len(y_train[0]))
    ).to(device)

    logger.info(f"Training set size: {len(train_set)}, Evaluation set size: {len(eval_set)}")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(list(classifier.parameters()) + list(encoder.parameters()), lr=1e-4, weight_decay=0.01)  

    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps= warm_up_steps,
            num_training_steps=total_steps
    )

    logger.info("Starting training of MiniLM model with classification head...")

    low_loss = float('inf')

    patience_counter = 0
    for epoch in range(epoch):
            encoder.train()
            classifier.train()
            losses = []
            for i in range(0, len(train_set), batch_size):
               
                batch = train_set[i:i+batch_size]

                labels = torch.tensor(batch['labels']).to(device)

                texts = drop_tokens_batch(batch['text'])

                features = encoder.tokenize(texts)

                features = {k: v.to(device) for k, v in features.items()}

                embeddings = encoder(features)['sentence_embedding']
                embeddings = torch.tensor(embeddings).to(device)
                logits = classifier(embeddings)
               
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                if i % 100 == 0:
                    logger.info("Done {} out of {} batches".format(i, len(train_set)))
            
            encoder.eval()
            classifier.eval()
            with torch.no_grad():
                for i in range(0, len(eval_set), batch_size):
                    batch = eval_set[i:i+batch_size]

                    labels = torch.tensor(batch['labels']).to(device)

                    embeddings = encoder.encode(batch['text'])
                    embeddings = torch.tensor(embeddings).to(device)
                    logits = classifier(embeddings)

                    loss = criterion(logits, labels)

                    losses.append(loss.item()) 

            avg_loss = sum(losses) / len(losses)
            logger.info(f"Epoch {epoch+1} completed, Loss: {avg_loss:.4f}")

            if(avg_loss < low_loss):
                low_loss = avg_loss
                patience_counter = 0
                encoder.save(model_save_path)
                logger.info(f"encoder saved at {model_save_path}.")
            else:
                patience_counter += 1
                if(patience_counter >= 2):
                    logger.info("Early stopping triggered.")
                    break
    logger.info("MiniLM model trained with classification head.")
    return {
        "total_steps": total_steps,
        "warm_up_steps": warm_up_steps,
        "batch_size": batch_size,
        "epochs": epoch,
        "model_save_path": model_save_path
    }


class MiniLMModel(BaseModel):
    """
    MiniLM model implementation for efficient text embeddings.
    """

    def __init__(self, language, path=None):
        """
        Initialize the MiniLM model with configuration parameters.

        Args:
            language (str): Language for the model.
            path (str, optional): Path to load a pre-trained model. Defaults to None.
                                    If None, a new model is initialized.
        """
        self.number_of_estimators = 300
        self.learning_rate = 0.1
        self.max_depth = 4
        self.tree_method = 'hist'
        self.objective = 'binary:logistic'
        self.eval_metric = 'logloss'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.params = {
            "number_of_estimators": self.number_of_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "tree_method": self.tree_method,
            "objective": self.objective,
            "eval_metric": self.eval_metric
        }
        super().__init__(language, path)

    def setup_model(self):
        """
        Initialize the MiniLM SentenceTransformer model.
        """
        self.encoder = None
        self.model_path = "sentence-transformers/minilm.pt"
        xgb_classifier = XGBClassifier(n_estimators=self.number_of_estimators, 
                                        eval_metric=self.eval_metric,
                                        objective=self.objective,
                                        learning_rate=self.learning_rate,
                                        max_depth=self.max_depth,
                                        tree_method=self.tree_method)
        
        self.classifier = MultiOutputClassifier(xgb_classifier)
        logger.info("MiniLM model initialized.")
    
   

    def train(self, X_train, y_train):
        """
        Train the MiniLM model with a classification head.

        Args:
            X_train: Input training data.
            y_train: True labels for training data.
        """
        if self.encoder is None and self.language == "java":
            if os.path.exists(self.model_path):
                logger.info(f"Loading existing MiniLM model from {self.model_path} for fine-tuning...")
                self.encoder = SentenceTransformer(self.model_path).to(self.device)
            else:
                logger.info(f"Fine-tuning MiniLM encoder using {self.language} training data...")
                parameters = finetune_miniLM(X_train, y_train,device=self.device, model_save_path=self.model_path)
                self.params.update(parameters)
                self.encoder = SentenceTransformer(parameters["model_save_path"]).to(self.device)

        if self.encoder is None:
            self.encoder = SentenceTransformer(self.model_path).to(self.device)

        y_train = np.array(y_train,dtype=np.float32)

        train_embeddings = self.encoder.encode(X_train)

        logger.info("Starting training of MiniLM model with Xgboost...")

        self.classifier.fit(train_embeddings, y_train)

        return {
            "n_estimators": self.number_of_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "tree_method": self.tree_method,
            "objective": self.objective,
            "eval_metric": self.eval_metric
        }
    

    def evaluate(self, X_test, y_test) -> dict[str,any]:
        """
        Evaluate the MiniLM model on test data.

        Args:
            X_test: Input test data.
            y_test: True labels for test data.
        """
        y_test = np.array(y_test,dtype=np.float32)
        
        test_embeddings = self.encoder.encode(X_test)
        
        predictions = self.classifier.predict(test_embeddings)
            
        accuracy = accuracy_score(y_test, predictions)

        f1_micro = f1_score(y_test, predictions, average='micro')
        f1_macro = f1_score(y_test, predictions, average='macro')
        f1_weighted = f1_score(y_test, predictions, average='weighted')

        recall = recall_score(y_test, predictions, average='weighted')

        precision = precision_score(y_test, predictions, average='weighted')

        metrics = {
            "accuracy": accuracy,
            "f1_micro_score": f1_micro,
            "f1_macro_score": f1_macro,
            "f1_weighted_score": f1_weighted,
            "recall": recall,
            "precision": precision
        }

        return metrics
    
    def predict(self, X) -> ndarray:
        """
        Make predictions using the trained MiniLM model.

        Args:
            X: Input data for prediction.

        Returns:
            Predictions made by the model.
        """

        if self.encoder is None or self.classifier is None:
            raise ValueError("Model is not trained. Call train() or load() before prediction.")
       
        encodedText = self.encoder.encode(X)
    
        predictions = self.classifier.predict(encodedText)

        logger.info(f"Predictions: {predictions}.")

        return predictions
    
    def save(self, path, model_name):
        """
        Save model and log to MLflow.

        Args:
            path (str): Path to save the model.
            model_name (str): Name to use when saving the model (without extension).
        """

        if self.encoder is None and self.classifier is None:
            raise ValueError("Model is not trained. Cannot save uninitialized model.")
        
        complete_path = os.path.join(path, model_name)
        encoder_path = complete_path+f"_encoder_{self.language}"
        classifier_path = complete_path+f"_xgb_classifier_{self.language}.joblib"
        
        if os.path.exists(complete_path) and os.path.isdir(complete_path):
            shutil.rmtree(complete_path) 

        self.encoder.save(encoder_path)
        joblib.dump(self.classifier, classifier_path)   
        
        try:
            # Log to MLflow
            logger.info("Logging artifacts to MLflow...")
            mlflow.pyfunc.log_model(
                artifact_path=f"{model_name}_{self.language}",
                python_model=MiniLMClassifierWrapper(),
                artifacts={
                    "encoder_path": encoder_path,
                    "classifier_path": classifier_path
                },
                code_paths=["turing/modeling/models/MiniLMClassifierWrapper.py"]
            )
        except Exception as e:
            logger.error(f"Failed to log model artifacts to MLflow: {e}")


