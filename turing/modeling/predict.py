import importlib
import warnings

import dagshub
from loguru import logger
import mlflow
import numpy as np
import pandas as pd

from turing.config import INPUT_COLUMN, LABELS_MAP, LANGS, MODEL_CONFIG, MODELS_DIR
from turing.dataset import DatasetManager
from turing.modeling.model_selector import get_best_model_info
from turing.modeling.models.codeBerta import CodeBERTa


class ModelInference:
    # Model Configuration (Fallback Registry)
    FALLBACK_MODEL_REGISTRY = {
        "java": {
            "run_id": "446f4459780347da8c796e619129be37",
            "artifact": "fine-tuned-CodeBERTa_java",
            "model_id": "codeberta",
        },
        "python": {
            "run_id": "ef5fd8ebf33a412087dcf02afd9e3147",
            "artifact": "fine-tuned-CodeBERTa_python",
            "model_id": "codeberta",
        },
        "pharo": {
            "run_id": "97822c6d84fc40c5b2363c9201a39997",
            "artifact": "fine-tuned-CodeBERTa_pharo",
            "model_id": "codeberta",
        },
    }


    def __init__(self, repo_owner="se4ai2526-uniba", repo_name="Turing", use_best_model_tags=True):
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        warnings.filterwarnings("ignore")
        self.dataset_manager = DatasetManager()
        self.use_best_model_tags = use_best_model_tags

        # Initialize model registry based on configuration
        if use_best_model_tags:
            logger.info("Using MLflow tags to find best models")

            self.model_registry = {}
            for lang in LANGS:
                try:
                    model_info = get_best_model_info(
                        lang, fallback_registry=self.FALLBACK_MODEL_REGISTRY
                    )
                    self.model_registry[lang] = model_info
                    logger.info(f"Loaded model info for {lang}: {model_info}")

                    # raise error if any required info is missing
                    if not all(k in model_info for k in ("run_id", "artifact", "model_id")):
                        raise ValueError(f"Incomplete model info for {lang}: {model_info}")

                except Exception as e:
                    logger.warning(f"Could not load model info for {lang}: {e}")
                    if lang in self.FALLBACK_MODEL_REGISTRY:
                        self.model_registry[lang] = self.FALLBACK_MODEL_REGISTRY[lang]

                # Pre-cache models locally
                run_id = self.model_registry[lang]["run_id"]
                artifact = self.model_registry[lang]["artifact"]
                self._get_cached_model_path(run_id, artifact, lang)
        else:
            logger.info("Using hardcoded model registry")
            self.model_registry = self.FALLBACK_MODEL_REGISTRY

    def _decode_predictions(self, raw_predictions, language: str):
        """
        Converts the binary matrix from the model into human-readable labels.

        Args:
            raw_predictions: Numpy array or similar with binary predictions
            language: Programming language for label mapping
        """

        labels_map = LABELS_MAP.get(language, [])
        decoded_results = []

        # Ensure input is a numpy array for processing
        if isinstance(raw_predictions, list):
            raw_array = np.array(raw_predictions)
        elif isinstance(raw_predictions, pd.DataFrame):
            raw_array = raw_predictions.values
        else:
            raw_array = raw_predictions

        # Iterate over rows
        for row in raw_array:
            indices = np.where(row == 1)[0]
            # Map indices to labels safely
            row_labels = [labels_map[i] for i in indices if i < len(labels_map)]
            decoded_results.append(row_labels)

        return decoded_results

    def _get_cached_model_path(self, run_id: str, artifact_name: str, language: str) -> str:
        """Checks if model exists locally; if not, downloads it from MLflow."""
        # Define local path: models/mlflow_temp_models/language/artifact_name
        local_path = MODELS_DIR / "mlflow_temp_models" / language / artifact_name

        if local_path.exists():
            logger.info(f"Loading {language} model from local cache: {local_path}")
            return str(local_path)

        logger.info(
            f"Model not found locally. Downloading {language} model from MLflow (Run ID: {run_id})..."
        )

        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download artifacts to the parent directory (artifact_name folder will be created inside)
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_name, dst_path=str(local_path.parent)
        )
        logger.success(f"Model downloaded and cached at: {local_path}")

        return str(local_path)

    def predict_payload(self, texts: list[str], language: str):
        """
        API Prediction: Automatically fetches the correct model from the registry based on language.

        Args:
            texts: List of code comments to classify
            language: Programming language
        """

        # 1. Validate Language and Fetch Config
        if language not in self.model_registry:
            raise ValueError(
                f"Language '{language}' is not supported or the model is not configured."
            )

        model_config = self.model_registry[language]
        run_id = model_config["run_id"]
        artifact_name = model_config["artifact"]
        model_id = model_config["model_id"]

        # Dynamically import model class
        config_entry = MODEL_CONFIG[model_id]
        module_name = config_entry["model_class_module"]
        class_name = config_entry["model_class_name"]
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        # 2. Get Model Path (Local Cache or Download)
        model_path = self._get_cached_model_path(run_id, artifact_name, language)

        # Load Model
        model = model_class(language=language, path=model_path)

        # 3. Predict
        raw_predictions = model.predict(texts)

        # 4. Decode Labels
        decoded_labels = self._decode_predictions(raw_predictions, language)

        return raw_predictions, decoded_labels, run_id, artifact_name

    def predict_from_mlflow(
        self, mlflow_run_id: str, artifact_name: str, language: str, model_class=CodeBERTa
    ):
        """
        Legacy method for CML/CLI: Predicts on the test dataset stored on disk.
        """
        # Load Dataset
        try:
            full_dataset = self.dataset_manager.get_dataset()
            dataset_key = f"{language}_test"
            if dataset_key not in full_dataset:
                raise ValueError(f"Dataset key '{dataset_key}' not found.")
            test_ds = full_dataset[dataset_key]
            X_test = test_ds[INPUT_COLUMN]
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise e

        # Load Model (Local Cache or Download)
        model_path = self._get_cached_model_path(mlflow_run_id, artifact_name, language)
        model = model_class(language=language, path=model_path)

        raw_predictions = model.predict(X_test)

        # Decode output
        readable_predictions = self._decode_predictions(raw_predictions, language)

        logger.info("Dataset prediction completed.")
        return readable_predictions
