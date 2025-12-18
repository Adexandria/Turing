import os
from pathlib import Path
import sys

from loguru import logger
import typer
from typing_extensions import Annotated

try:
    from turing.config import INTERIM_DATA_DIR, RAW_DATA_DIR
    from turing.dataset import DatasetManager
except ImportError:
    logger.error("Error: Could not import DatasetManager. Check sys.path configuration.")
    logger.error(f"Current sys.path: {sys.path}")
    sys.exit(1)


script_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(proj_root)

app = typer.Typer(help="CLI for dataset management (Download, Conversion, and Search).")


@app.command()
def download():
    """
    Loads the dataset from Hugging Face and saves it into the "raw" folder.
    """
    logger.info("Starting dataset download...")
    manager = DatasetManager()
    manager.download_dataset()
    logger.success("Download complete.")


@app.command(name="parquet-to-csv")
def parquet_to_csv():
    """
    Converts all parquet files in the raw data directory
    to CSV format in the interim data directory.
    """
    logger.info("Starting Parquet -> CSV conversion...")
    manager = DatasetManager()
    manager.parquet_to_csv()
    logger.success("Conversion complete.")


@app.command()
def search(
    filename: Annotated[
        str, typer.Argument(help="The exact filename to search for (e.g., 'java_train.parquet')")
    ],
    directory: Annotated[
        str,
        typer.Option(
            "--directory",
            "-d",
            help="Directory to search in. Keywords 'raw' or 'interim' can be used.",
        ),
    ] = "raw",
):
    """
    Searches for a file by name in the data directories.
    """
    logger.info(f"Initializing search for '{filename}'...")
    manager = DatasetManager()

    search_path = None
    if directory.lower() == "raw":
        search_path = RAW_DATA_DIR
        logger.info("Searching in 'raw' data directory.")
    elif directory.lower() == "interim":
        search_path = INTERIM_DATA_DIR
        logger.info("Searching in 'interim' data directory.")
    else:
        search_path = Path(directory)
        logger.info(f"Searching in custom path: {search_path}")

    results = manager.search_file(filename, search_directory=search_path)

    if results:
        logger.success(f"Found {len(results)} file(s):")
        for res in results:
            print(f"-> {res}")
    else:
        logger.warning(f"File '{filename}' not found in {search_path}.")


@app.command(name="show-raw-hf")
def show_raw_hf():
    """
    Loads and displays info about the raw dataset from Hugging Face.
    """
    logger.info("Loading raw dataset info from Hugging Face...")
    manager = DatasetManager()
    dataset = manager.get_raw_dataset_from_hf()
    if dataset:
        logger.info("Dataset info:")
        print(dataset)
    else:
        logger.error("Could not retrieve dataset.")


if __name__ == "__main__":
    app()
