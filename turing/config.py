from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Dataset
DATASET_HF_ID = "NLBSE/nlbse26-code-comment-classification"
LANGS = ["java", "python", "pharo"]
INPUT_COLUMN = "combo"
LABEL_COLUMN = "labels"

LABELS_MAP = {
    "java": ["summary", "Ownership", "Expand", "usage", "Pointer", "deprecation", "rational"],
    "python": ["Usage", "Parameters", "DevelopmentNotes", "Expand", "Summary"],
    "pharo": [
        "Keyimplementationpoints",
        "Example",
        "Responsibilities",
        "Intent",
        "Keymessages",
        "Collaborators",
    ],
}

TOTAL_CATEGORIES = sum(len(v) for v in LABELS_MAP.values())

# Score parameters
MAX_AVG_RUNTIME = 5.0  # seconds
MAX_AVG_FLOPS = 5000.0  # GFLOPS

# Training parameters
DEFAULT_BATCH_SIZE = 32

# Model configuration mapping
MODEL_CONFIG = {
    "codeberta": {
        "model_name": "fine-tuned-CodeBERTa",
        "exp_name": "fine-tuned-CodeBERTa",
        "model_class_module": "turing.modeling.models.codeBerta",
        "model_class_name": "CodeBERTa",
    },
    "graphcodebert": {
        "model_name": "GraphCodeBERT",
        "exp_name": "fine-tuned-GraphCodeBERT",
        "model_class_module": "turing.modeling.models.graphCodeBert",
        "model_class_name": "GraphCodeBERTClassifier",
    },
    "tinybert": {
        "model_name": "TinyBERT",
        "exp_name": "fine-tuned-TinyBERT",
        "model_class_module": "turing.modeling.models.tinyBert",
        "model_class_name": "TinyBERTClassifier",
    },
    "randomforest": {
        "model_name": "RandomForest-TfIdf",
        "exp_name": "RandomForest-TfIdf",
        "model_class_module": "turing.modeling.models.randomForestTfIdf",
        "model_class_name": "RandomForestTfIdf",
    },
    "minilm": {
        "model_name": "MiniLM",
        "exp_name": "fine-tuned-MiniLm",
        "model_class_module": "turing.modeling.models.miniLM",
        "model_class_name": "MiniLMModel",
    },
}
DEFAULT_NUM_ITERATIONS = 20

# Existing model modules
EXISTING_MODELS = [
    "randomForestTfIdf",
    "codeBerta",
]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except (ModuleNotFoundError, ValueError):
    pass
