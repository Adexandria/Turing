# Model Selection with MLflow Tags

System for tracking datasets and selecting best models per language using MLflow tags.

## What It Does

- Automatically logs which dataset was used to train each model
- Tracks best-performing models per language
- Enables reproducible model selection without hardcoding values
- Maintains clean separation between different training runs

## Quick Start

### Tag Best Models

After training, identify and tag the best model per language:

```bash
python scripts/tag_best_models.py tag
```

This:
- Searches all training runs in the experiment
- Finds the best model for each language based on f1_score
- Applies tags: `best_model=true`, `best_model_{language}=true`
- Records dataset name and selection timestamp
- Removes tags from other runs

### View Tagged Models

```bash
python scripts/tag_best_models.py show
```

Shows all tagged models with:
- Language and run information
- Dataset used for training
- Complete metrics

### Start API

```bash
uvicorn turing.api.app:app --reload
```

The API automatically loads and uses the tagged best models.

## How It Works

### Training Phase

During training, the system logs:

```python
dataset = DatasetManager()
dataset_name = dataset.get_dataset_name()

with mlflow.start_run(run_name=f"{model_name}_{lang}"):
    mlflow.set_tag("Language", lang)
    mlflow.set_tag("dataset_name", dataset_name)
    mlflow.log_params(model.params)
    # training...
```

Tags applied:
- `Language`: Language code
- `dataset_name`: Name of dataset folder
- `model_name`: Model identifier

### Selection Phase

The tagging script processes runs:

```python
def tag_best_models(experiment_name, metric="f1_score", languages=None):
    # For each language:
    # 1. Find all runs with that language tag
    # 2. Order by metric (highest first)
    # 3. Tag the best run
    # 4. Remove tags from others
```

### Dataset Name Resolution

Dataset names are extracted dynamically from the file system:

```python
def get_dataset_name(self) -> str:
    return self.base_interim_path.name
```

No hardcoded values - changes to the dataset path automatically propagate.

### Model Loading

The inference engine uses this selection strategy:

1. Query MLflow for models tagged `best_model=true`
2. Fall back to hardcoded registry if no tags found
3. Use metric-based selection as last resort

```python
from turing.modeling.predict import ModelInference

inference = ModelInference(use_best_model_tags=True)
response = inference.predict_payload(request)
```

## Configuration

Use tags for dynamic selection (default):

```python
inference = ModelInference(use_best_model_tags=True)
```

Or use hardcoded fallback:

```python
inference = ModelInference(use_best_model_tags=False)
```

## Advanced Usage

### Custom Metric

Select best models using a different metric:

```bash
python scripts/tag_best_models.py tag --metric precision
```

### Programmatic Tagging

```python
from scripts.tag_best_models import tag_best_models

tag_best_models(
    experiment_name="fine-tuned-CodeBERTa",
    metric="f1_score",
    languages=["java", "python", "pharo"]
)
```

## Tags Reference

### Applied During Training

| Tag | Purpose |
|-----|---------|
| `Language` | Language identifier |
| `dataset_name` | Dataset folder name |
| `model_name` | Model identifier |

### Applied During Selection

| Tag | Purpose |
|-----|---------|
| `best_model` | Marks the best model |
| `best_model_{lang}` | Language-specific best indicator |
| `selection_metric` | Metric used for selection |
| `selection_date` | When selection occurred |

## Data Flow

```
Training Runs (MLflow)
        ↓
  tag_best_models.py
        ↓
  Tagged Best Models
        ↓
  model_selector.py
        ↓
  ModelInference
        ↓
  API /predict
```

## Features

- **Dynamic Dataset Tracking**: Dataset name auto-detected from path
- **Per-Language Selection**: Separate best model for each language
- **Flexible Metrics**: Choose f1_score, precision, recall, or accuracy
- **Fallback Safety**: Works even if MLflow is unavailable
- **Clean Metadata**: All selection decisions recorded with timestamps

## Troubleshooting

### No tagged models found

```bash
python scripts/tag_best_models.py show
mlflow ui  # Check experiment directly
```

### Wrong model selected

Re-run tagging with correct metric:

```bash
python scripts/tag_best_models.py tag --metric f1_score
python scripts/tag_best_models.py show
```

### Dataset name not captured

Verify dataset name resolution:

```python
from turing.dataset import DatasetManager
print(DatasetManager().get_dataset_name())
```

## Files

- `turing/modeling/train.py` - Logs Language and dataset_name tags
- `turing/dataset.py` - Provides get_dataset_name() method
- `turing/modeling/model_selector.py` - Queries and selects models
- `turing/modeling/predict.py` - Loads selected models
- `turing/api/app.py` - Serves predictions
- `scripts/tag_best_models.py` - CLI for tagging operations
