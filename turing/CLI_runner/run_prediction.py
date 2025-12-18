from pathlib import Path
import sys

from loguru import logger
import typer

from turing.modeling.models.randomForestTfIdf import RandomForestTfIdf
from turing.modeling.predict import ModelInference

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

app = typer.Typer()


@app.command()
def main(
    mlflow_run_id: str = typer.Option(
        "af1fa5959dc14fa9a29a0a19c11f1b08", help="The MLflow Run ID"
    ),
    artifact_name: str = typer.Option(
        "RandomForestTfIdf_java", help="The name of the model artifact"
    ),
    language: str = typer.Option("java", help="The target programming language"),
):
    """
    Run inference using the dataset stored on disk (Standard CML/DVC workflow).
    """
    logger.info("Starting CLI inference process...")

    try:
        # Initialize inference engine
        inference_engine = ModelInference()

        # Run prediction on the test dataset
        results = inference_engine.predict_from_mlflow(
            mlflow_run_id=mlflow_run_id,
            artifact_name=artifact_name,
            language=language,
            model_class=RandomForestTfIdf,
        )

        # Output results
        print("\n--- Prediction Results ---")
        print(results)
        print("--------------------------")

    except Exception as e:
        logger.error(f"CLI Prediction failed: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
