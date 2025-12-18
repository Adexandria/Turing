# Adding a New Model

This guide explains how to integrate a new classification model into the Turing project.

## Prerequisites

Every new model must:
1. Inherit from the abstract class `BaseModel`
2. Implement all required abstract methods
3. Handle automatic conversion of input data
4. Support multi-label classification

## Base Class Structure

```python
from turing.modeling.baseModel import BaseModel

class BaseModel(ABC):
    def __init__(self, language, path=None):
        """Initialize the model"""
        self.language = language
        self.model = None
        if path:
            self.load(path)
        else:
            self.setup_model()
    
    @abstractmethod
    def setup_model(self):
        """Initialize the model architecture"""
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, path, model_name):
        """Train the model"""
        pass
    
    @abstractmethod
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    @abstractmethod
    def save(self, path, model_name):
        """Save the model"""
        pass
    
    @abstractmethod
    def load(self, model_path):
        """Load a saved model"""
        pass
```

## Implementing a New Model

### 1. Create the Class File

Create a new file in `turing/modeling/models/` with your model name:

```python
# turing/modeling/models/myModel.py

from typing import List
import numpy as np
from loguru import logger
import turing.config as config
from turing.modeling.baseModel import BaseModel

class MyModel(BaseModel):
    """
    Description of your model.
    """
    
    def __init__(self, language: str, path: str = None):
        self.device = None  # If using GPU/CPU
        self.model = None
        self.params = {
            "param1": value1,
            "param2": value2,
            # ... other parameters
        }
        super().__init__(language=language, path=path)
```

### 2. Implement `setup_model()`

This method must initialize the model architecture:

```python
def setup_model(self):
    """
    Initialize the model.
    
    Called automatically from __init__ if path=None.
    """
    try:
        # Load pre-trained weights or create from scratch
        # Configure layers, device, etc.
        self.model = ...
        logger.success(f"Model initialized for {self.language}")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise
```

### 3. Implement `train()`

The training method must:
- Accept X_train (list of strings) and y_train (numpy array)
- Return a dictionary with logged parameters
- Save the model if `path` is provided

```python
def train(
    self,
    X_train: List[str],
    y_train: np.ndarray,
    path: str = None,
    model_name: str = "my_model",
    **kwargs
) -> dict:
    """
    Train the model.
    
    Args:
        X_train: List of training texts
        y_train: Numpy array of shape (n_samples, n_labels)
        path: Directory to save the model
        model_name: Name of the saved model
        **kwargs: Additional hyperparameters
    
    Returns:
        dict: Training parameters to log in MLflow
    """
    try:
        if self.model is None:
            self.setup_model()
        
        # Training code
        logger.info(f"Training {self.language}...")
        
        # ... Training loop ...
        
        logger.success(f"Training completed for {self.language}")
        
        if path:
            self.save(path, model_name)
        
        return {
            "param1": value1,
            "param2": value2,
            # Actual parameters used
        }
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise
```

### 4. Implement `predict()`

Must automatically convert input data:

```python
def predict(self, X: List[str]) -> np.ndarray:
    """
    Make predictions on new data.
    
    Args:
        X: List of texts or pandas Series/Dataset Column
    
    Returns:
        np.ndarray: Prediction matrix (n_samples, n_labels)
                    with values 0 or 1 for multi-label
    """
    if self.model is None:
        raise ValueError("Model not initialized.")
    
    # Automatic conversion from Series/Column to list
    if hasattr(X, 'tolist'):
        X = X.tolist()
    elif hasattr(X, '__iter__') and not isinstance(X, list):
        X = list(X)
    
    try:
        # Prediction code
        predictions = ...
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise
```

### 5. Implement `evaluate()`

Must handle automatic conversion of y_test:

```python
def evaluate(self, X_test: List[str], y_test) -> dict:
    """
    Evaluate the model.
    
    Args:
        X_test: Test texts
        y_test: True labels (can be Series, Column, or array)
    
    Returns:
        dict: Metrics (f1_score, precision, recall, accuracy)
    """
    try:
        # Get predictions
        predictions = self.predict(X_test)
        
        # Convert y_test if necessary
        if not isinstance(y_test, np.ndarray):
            y_test = np.array(y_test)
        
        # Handle conversion from indices to multi-hot if necessary
        if (predictions.ndim == 2 and predictions.shape[1] > 1 and
            (y_test.ndim == 1 or (y_test.ndim == 2 and y_test.shape[1] == 1))):
            
            y_test_expanded = np.zeros((y_test.shape[0], self.num_labels), dtype=int)
            indices = y_test.flatten()
            
            for i, label_idx in enumerate(indices):
                idx = int(label_idx)
                if 0 <= idx < self.num_labels:
                    y_test_expanded[i, idx] = 1
            
            y_test = y_test_expanded
        
        # Calculate metrics
        metrics = {
            "f1_score": ...,
            "precision": ...,
            "recall": ...,
            "accuracy": ...
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise
```

### 6. Implement `save()` and `load()`

```python
def save(self, path: str, model_name: str = "my_model"):
    """Save the model to disk"""
    try:
        import os
        model_path = os.path.join(path, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Save weights, config, tokenizer, etc.
        # ...
        
        logger.success(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def load(self, model_path: str):
    """Load the model from disk"""
    try:
        # Load weights and config
        # ...
        
        logger.success(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
```

## Integration into the Training Pipeline

### 1. Update `train.py`

```python
from turing.modeling.models.myModel import MyModel

@app.command()
def main(model: str = typer.Option("codebert", help="Model to train: ...")):
    
    if model.lower() == "mymodel":
        model_name = "MyModel"
        exp_name = "fine-tuned-MyModel"
        model_class = MyModel
    # ... other models ...
```

### 2. Update `conftest.py` (for tests)

```python
# If your model is the default
MODEL_CLASS_TO_TEST = train.MODEL_CLASS
MODEL_EXPERIMENT_NAME = train.EXP_NAME
MODEL_NAME_BASE = train.MODEL_NAME

# Or load your model specifically
from turing.modeling.models.myModel import MyModel
```

## Checklist for a New Model

- [ ] Inherits from `BaseModel`
- [ ] Implements all 6 abstract methods
- [ ] `__init__` calls `super().__init__(language, path)`
- [ ] `setup_model()` is called in `__init__` if path=None
- [ ] `train()` returns dict of parameters
- [ ] `predict()` handles automatic conversion from Series/Column
- [ ] `evaluate()` handles automatic conversion of y_test
- [ ] All logging uses `logger` (from loguru)
- [ ] Error handling with try/except
- [ ] Complete documentation with docstrings
- [ ] Support for multi-label classification

## Special Attention Points

### 1. Data Handling

**Input X_train/X_test** can come as:
- List of strings 
- pandas Series
- Dataset Column (from Hugging Face)

**Input y_train/y_test** can come as:
- numpy array (multi-hot encoding, shape (n, n_labels))
- numpy array (flat indices, shape (n,))
- pandas Series
- Dataset Column

**Always convert internally in the `evaluate()` method!**

### 2. MLflow Logging

The `train.py` automatically logs:
```python
mlflow.log_params(model.params)  # From __init__
mlflow.log_params(parameters_to_log)  # From train()
mlflow.log_metrics(metrics)  # From evaluate()
mlflow.set_tag("dataset_name", dataset_name)
mlflow.set_tag("Language", lang)
mlflow.set_tag("model_name", f"{model_name}_{lang}")
```

Do not modify `train.py` for your class!

### 3. Dynamic Dataset Name

The dataset name is obtained automatically:
```python
dataset_name = dataset.get_dataset_name()
mlflow.set_tag("dataset_name", dataset_name)
```

Do not hardcode the dataset name in the model!

### 4. Device (GPU/CPU)

```python
# Good
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Don't use
self.device = torch.device("cuda")  # Fails without GPU
```

### 5. Number of Labels

```python
# Get dynamically from config
self.labels_map = config.LABELS_MAP.get(language, [])
self.num_labels = len(self.labels_map)

# Don't hardcode!
self.num_labels = 7  # Only for Java
```

## Complete Minimal Example

```python
class SimpleModel(BaseModel):
    def __init__(self, language: str, path: str = None):
        self.model = None
        self.params = {"model": "simple"}
        super().__init__(language=language, path=path)
    
    def setup_model(self):
        self.model = "initialized"
        logger.success(f"Model setup for {self.language}")
    
    def train(self, X_train, y_train, path=None, model_name="model", **kwargs):
        logger.info(f"Training on {len(X_train)} samples")
        if path:
            self.save(path, model_name)
        return {"samples": len(X_train)}
    
    def predict(self, X):
        if hasattr(X, 'tolist'):
            X = X.tolist()
        return np.random.randint(0, 2, (len(X), self.num_labels))
    
    def evaluate(self, X_test, y_test):
        pred = self.predict(X_test)
        y_test = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
        return {"f1_score": 0.5, "precision": 0.5, "recall": 0.5, "accuracy": 0.5}
    
    def save(self, path, model_name="model"):
        logger.info(f"Saving to {path}/{model_name}")
    
    def load(self, model_path):
        logger.info(f"Loading from {model_path}")
```

## Execution

After implementation:

```bash
# Train the model
python -m turing.modeling.train --model mymodel

# View results in MLflow
python scripts/tag_best_models.py show
```

## Support

For questions or issues:
1. Check existing models (CodeBERTa, RandomForest, TinyBERT)
2. Read BaseModel documentation
3. Verify data structure with print/logger
