from typing import Optional

from loguru import logger
from mlflow.tracking import MlflowClient


def get_best_model_by_tag(
    language: str,
    tag_key: str = "best_model",
    metric: str = "f1_score"
) -> Optional[dict]:
    """
    Retrieve the best model for a specific language using MLflow tags.
    
    Args:
        language: Programming language (java, python, pharo)
        tag_key: Tag key to search for (default: "best_model")
        metric: Metric to use for ordering (default: "f1_score")
    
    Returns:
        Dict with run_id and artifact_name of the best model or None if not found
    """

    client = MlflowClient()
    experiments = client.search_experiments()
    if not experiments:
        logger.error("No experiments found in MLflow")
        return None
    
    try:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id for exp in experiments],
            filter_string=f"tags.{tag_key} = 'true' and tags.Language = '{language}'",
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        
        if not runs:
            logger.warning(f"No runs found with tag '{tag_key}' for language '{language}'")
            return None
        
        best_run = runs[0]
        run_id = best_run.info.run_id
        exp_name = client.get_experiment(best_run.info.experiment_id).name
        run_name = best_run.info.run_name
        artifact_name = best_run.data.tags.get("model_name")
        model_id = best_run.data.tags.get("model_id")
        logger.info(f"Found best model for {language}: {exp_name}/{run_name} ({run_id}), artifact={artifact_name}")
        
        return {
            "run_id": run_id,
            "artifact": artifact_name, 
            "model_id": model_id
        }
    
    except Exception as e:
        logger.error(f"Error searching for best model: {e}")
        return None


def get_best_model_info(
    language: str,
    fallback_registry: dict = None
) -> dict:
    """
    Retrieve the best model information for a language.
    First searches by tag, then falls back to hardcoded registry.
    
    Args:
        language: Programming language
        fallback_registry: Fallback registry with run_id and artifact
        
    Returns:
        Dict with run_id and artifact of the model
    """
    
    model_info = get_best_model_by_tag(language, "best_model")
    
    if model_info:
        logger.info(f"Using tagged best model for {language}")
        return model_info
    
    if fallback_registry and language in fallback_registry:
        logger.warning(f"No tagged model found for {language}, using fallback registry")
        return fallback_registry[language]
    
    model_info = get_best_model_by_metric(language)
    
    if model_info:
        logger.warning(f"Using best model by metric for {language}")
        return model_info
    
    raise ValueError(f"No model found for language {language}")


def get_best_model_by_metric(
    language: str,
    metric: str = "f1_score"
) -> Optional[dict]:
    """
    Find the model with the best metric for a language.
    
    Args:
        language: Programming language
        metric: Metric to use for ordering
        
    Returns:
        Dict with run_id and artifact of the model or None
    """

    client = MlflowClient()
    experiments = client.search_experiments()
    if not experiments:
        logger.error("No experiments found in MLflow")
        return None
    
    try:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id for exp in experiments],
            filter_string=f"tags.Language = '{language}'",
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        
        if not runs:
            logger.warning(f"No runs found for language '{language}'")
            return None
        
        best_run = runs[0]
        run_id = best_run.info.run_id
        exp_name = client.get_experiment(best_run.info.experiment_id).name
        run_name = best_run.info.run_name
        artifact_name = best_run.data.tags.get("model_name")
        model_id = best_run.data.tags.get("model_id")
        logger.info(f"Found best model for {language}: {exp_name}/{run_name} ({run_id}), artifact={artifact_name}")
        
        return {
            "run_id": run_id,
            "artifact": artifact_name,
            "model_id": model_id
        }
    
    except Exception as e:
        logger.error(f"Error finding best model by metric: {e}")
        return None
