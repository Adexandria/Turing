from importlib import import_module
import os
import warnings
from loguru import logger
import numpy as np
import typer

import turing.config as config
from turing.dataset import DatasetManager
from turing.evaluate_model import evaluate_models

warnings.filterwarnings("ignore")


DEFAULT_MODEL = "codeberta"
DEFAULT_DATASET = None

app = typer.Typer()


@app.command()
def main(
    model: str = typer.Option(DEFAULT_MODEL, help="Model to train: codeberta, graphcodebert, tinybert, or randomforest"),
    dataset: str = typer.Option(DEFAULT_DATASET, help="Dataset to use for training")
):
    """
    Train and evaluate models.

    Args:
        model (str): Model to train.
        dataset (str): Dataset to use for training.
    """

    # Get model configuration from config
    model_key = model.lower()
    if model_key not in config.MODEL_CONFIG:
        logger.error(f"Unknown model: {model_key}. Available models: {list(config.MODEL_CONFIG.keys())}")
        return
    
    model_cfg = config.MODEL_CONFIG[model_key]
    model_name = model_cfg["model_name"]
    exp_name = model_cfg["exp_name"]
    
    # Dynamically import model class
    module = import_module(model_cfg["model_class_module"])
    model_class = getattr(module, model_cfg["model_class_name"])
    
    logger.info(f"Training model: {model_name}")

    # Load dataset
    if dataset is None:
        dataset_path = config.INTERIM_DATA_DIR / "base"
    else:
        dataset_path = config.INTERIM_DATA_DIR / "features" / dataset
    dataset_manager = DatasetManager(dataset_path=dataset_path)

    try:
        full_dataset = dataset_manager.get_dataset()
        dataset_name = dataset_manager.get_dataset_name()
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    logger.info(f"Dataset loaded successfully: {dataset_name}")

    # Train and evaluate models for each language
    models = {}
    for lang in config.LANGS:
        # Prepare training and testing data
        train_ds = full_dataset[f"{lang}_train"]
        test_ds = full_dataset[f"{lang}_test"]
        X_train = train_ds[config.INPUT_COLUMN]
        y_train = train_ds[config.LABEL_COLUMN]
        X_test = test_ds[config.INPUT_COLUMN]
        y_test = test_ds[config.LABEL_COLUMN]
        X_train = list(X_train)
        X_test = list(X_test)
        y_train = np.array(y_train)

        # Initialize model
        model = model_class(language=lang)

        # Train and evaluate model
        try:
            model.train(
                X_train,
                y_train
            )
            model.save(os.path.join(config.MODELS_DIR, exp_name), model_name=model_name)
            model.evaluate(X_test, y_test)  
        except Exception as e:
            logger.error(f"Error training/evaluating model for {lang}: {e}")
            return

        # Store trained model
        models[lang] = model
    logger.success(f"All {model_name} models trained and evaluated.")

    # Competition-style evaluation of trained models
    logger.info("Starting competition-style evaluation of trained models...")
    evaluate_models(models, full_dataset)
    logger.success("Evaluation completed.")


if __name__ == "__main__":
    app()
