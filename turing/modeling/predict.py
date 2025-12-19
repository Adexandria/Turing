import importlib
import os
import warnings

from loguru import logger
import numpy as np
import pandas as pd
import typer

from turing.config import INPUT_COLUMN, LABELS_MAP, LANGS, MODEL_CONFIG, MODELS_DIR

app = typer.Typer()


class ModelInference:
    
    MODEL_REGISTRY = {
        "java": {
            "model_path": os.path.join(MODELS_DIR, "fine-tuned-GraphCodeBERT", "GraphCodeBERT_java"),
            "model_id": "graphcodebert",
        },
        "python": {
            "model_path": os.path.join(MODELS_DIR, "fine-tuned-GraphCodeBERT", "GraphCodeBERT_python"),
            "model_id": "graphcodebert",
        },
        "pharo": {
            "model_path": os.path.join(MODELS_DIR, "fine-tuned-GraphCodeBERT", "GraphCodeBERT_pharo"),
            "model_id": "graphcodebert",
        },
    }


    def __init__(self):
        warnings.filterwarnings("ignore")


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
    

    def predict_payload(self, texts: list[str], language: str):
        """
        API Prediction: Automatically fetches the correct model from the registry based on language.

        Args:
            texts: List of code comments to classify
            language: Programming language

        Returns:
            Tuple of (raw_predictions, decoded_labels)
        """

        # Validate Language and Fetch Config
        if language not in self.MODEL_REGISTRY:
            raise ValueError(f"Language '{language}' is not supported or the model is not configured.")

        model_config = self.MODEL_REGISTRY[language]
        model_id = model_config["model_id"]

        # Dynamically import model class
        config_entry = MODEL_CONFIG[model_id]
        module_name = config_entry["model_class_module"]
        class_name = config_entry["model_class_name"]
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        # Get Model Path and load the model
        model_path = self.MODEL_REGISTRY[language]["model_path"]
        model = model_class(language=language, path=model_path)

        # Predict and decode labels
        raw_predictions = model.predict(texts)
        decoded_labels = self._decode_predictions(raw_predictions, language)

        return raw_predictions, decoded_labels
    

@app.command()
def main():
    """
    Example CLI for testing ModelInference.
    """

    sample_texts = [
        "This function calculates the factorial of a number.",
        "Initialize the database connection and return the client object."
    ]

    inference_engine = ModelInference()
    language = "python"
    raw_preds, decoded = inference_engine.predict_payload(sample_texts, language)

    logger.info(f"Raw Predictions: {raw_preds}")
    logger.info(f"Decoded Predictions: {decoded}")


if __name__ == "__main__":
    app()