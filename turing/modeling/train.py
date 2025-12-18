from importlib import import_module
import os
import warnings

import dagshub
from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import typer

import turing.config as config
from turing.dataset import DatasetManager
from turing.evaluate_model import evaluate_models

dagshub.init(repo_owner="se4ai2526-uniba", repo_name="Turing", mlflow=True)

warnings.filterwarnings("ignore")

DEFAULT_MODEL = "codeberta"
_default_cfg = config.MODEL_CONFIG[DEFAULT_MODEL]

MODEL_CLASS_MODULE = _default_cfg["model_class_module"]
MODEL_CLASS_NAME = _default_cfg["model_class_name"]
MODEL_CLASS = __import__(MODEL_CLASS_MODULE, fromlist=[MODEL_CLASS_NAME])
MODEL_CLASS = getattr(MODEL_CLASS, MODEL_CLASS_NAME)
EXP_NAME = _default_cfg["exp_name"]
MODEL_NAME = _default_cfg["model_name"]



app = typer.Typer()


def tag_best_models(
    metric: str = "f1_score"
):
    """
    Tag the best existing models in MLflow based on the specified metric.
    Remove previous best_model tags before tagging the new best models.
    
    Args:
        metric: Metric to use for determining the best model
    """

    dagshub.init(repo_owner="se4ai2526-uniba", repo_name="Turing", mlflow=True)
    client = MlflowClient()
    
    # Get all experiments from Mlflow
    experiments = client.search_experiments()
    if not experiments:
        logger.error("No experiments found in MLflow")
        return
    
    # Find the best run for each language
    experiments_ids = [exp.experiment_id for exp in experiments]
    for lang in config.LANGS:
        # Get all runs for the language
        runs = client.search_runs(
            experiment_ids=experiments_ids,
            filter_string=f"tags.Language = '{lang}'",
            order_by=[f"metrics.{metric} DESC"]
        )

        if not runs:
            logger.warning(f"No runs found for language {lang}")
            continue
        logger.info(f"Found {len(runs)} runs for {lang}")

        # Get the best run for the language
        best_run = runs[0]
        run_id = best_run.info.run_id

        # Remove previous best_model tags for this language
        for run in runs[1:]:
            try:
                client.delete_tag(run.info.run_id, "best_model")
            except Exception:
                pass

        # Tag the best model
        client.set_tag(run_id, "best_model", "true")


def show_tagged_models():
    """
    Show all models tagged as best_model.
    """

    dagshub.init(repo_owner="se4ai2526-uniba", repo_name="Turing", mlflow=True)
    client = MlflowClient()

    # Get all experiments from Mlflow
    experiments = client.search_experiments()
    if not experiments:
        logger.error("No experiments found in MLflow")
        return
    
    # Find all runs tagged as best_model
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id for exp in experiments],
        filter_string="tags.best_model = 'true'",
        order_by=["tags.Language ASC"]
    )
    logger.info(f"\nFound {len(runs)} best models in experiments:\n")
    
    # Display details of each tagged best model
    for run in runs:
        language = run.data.tags.get("Language", "unknown")
        exp_name = client.get_experiment(run.info.experiment_id).name
        run_id = run.info.run_id
        run_name = run.data.tags.get("mlflow.runName", "N/A")
        dataset_name = run.data.tags.get("dataset_name", "unknown")
        
        logger.info(f"Language: {language}")
        logger.info(f"  Run: {exp_name}/{run_name} ({run_id})")
        logger.info(f"  Dataset: {dataset_name}")
        
        if run.data.metrics:
            for metric in run.data.metrics:
                logger.info(f"  {metric}: {run.data.metrics[metric]:.4f}")
        
        logger.info("")


@app.command()
def main(model: str = typer.Option("codeberta", help="Model to train: codeberta, graphcodebert, tinybert, or randomforest"), dataset: str = typer.Option(None, help="Dataset to use for training")):
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
    mlflow.set_experiment(exp_name)
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

        # Train and evaluate model within an MLflow run
        try:
            with mlflow.start_run(run_name=f"{model_name}_{lang}"):
                mlflow.set_tag("Language", lang)
                mlflow.set_tag("dataset_name", dataset_name)
                mlflow.set_tag("model_id", model_key)
                mlflow.log_params(model.params)
                parameters_to_log = model.train(
                    X_train,
                    y_train
                )
                mlflow.log_params(parameters_to_log)
                model.save(os.path.join(config.MODELS_DIR, exp_name),model_name=model_name)
                metrics = model.evaluate(X_test, y_test)
                mlflow.log_metrics(metrics)
                
                # Log model name for later retrieval
                mlflow.set_tag("model_name", f"{model_name}_{lang}")
                
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

    logger.info("Tagging best models in MLflow...")
    tag_best_models()
    logger.info("Best models:")
    show_tagged_models()


if __name__ == "__main__":
    app()
