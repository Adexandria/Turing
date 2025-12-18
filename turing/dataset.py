import ast
import os
from pathlib import Path

from datasets import DatasetDict, load_dataset
from loguru import logger

import turing.config as config


class DatasetManager:
    """
    Manages the loading, transformation, and access of project datasets.
    """

    def __init__(self, dataset_path: Path = None):
        self.hf_id = config.DATASET_HF_ID
        self.raw_data_dir = config.RAW_DATA_DIR
        self.interim_data_dir = config.INTERIM_DATA_DIR
        self.base_interim_path = self.interim_data_dir / "base"
        
        if dataset_path:
            self.dataset_path = dataset_path
        else:
            self.dataset_path = self.base_interim_path

    def _format_labels_for_csv(self, example: dict) -> dict:
        """
        Formats the labels list as a string for CSV storage.
        (Private class method)

        Args:
            example (dict): A single example from the dataset.

        Returns:
            dict: The example with labels converted to string.
        """
        labels = example.get("labels")
        if isinstance(labels, list):
            example["labels"] = str(labels)
        return example

    def download_dataset(self):
        """
        Loads the dataset from Hugging Face and saves it into the "raw" folder.
        """
        logger.info(f"Loading dataset: {self.hf_id}")
        try:
            ds = load_dataset(self.hf_id)
            logger.success("Dataset loaded successfully.")
            logger.info(f"Dataset splits: {ds}")

            self.raw_data_dir.mkdir(parents=True, exist_ok=True)

            for split_name, dataset_split in ds.items():
                output_path = os.path.join(
                    self.raw_data_dir, f"{split_name.replace('-', '_')}.parquet"
                )
                dataset_split.to_parquet(output_path)

            logger.success(f"Dataset saved to {self.raw_data_dir}.")
        except Exception as e:
            logger.warning(f"Error during loading: {e}.")

    def parquet_to_csv(self):
        """
        Converts all parquet files in the raw data directory
        to CSV format in the interim data directory.
        """
        logger.info("Starting Parquet to CSV conversion...")
        self.base_interim_path.mkdir(parents=True, exist_ok=True)

        for file_name in os.listdir(self.raw_data_dir):
            if file_name.endswith(".parquet"):
                part_name = file_name.replace(".parquet", "").replace("-", "_")

                # Load the parquet file
                dataset = load_dataset(
                    "parquet", data_files={part_name: str(self.raw_data_dir / file_name)}
                )

                # Map and format labels
                dataset[part_name] = dataset[part_name].map(self._format_labels_for_csv)

                # Save to CSV
                csv_output_path = os.path.join(self.base_interim_path, f"{part_name}.csv")
                dataset[part_name].to_csv(csv_output_path)

                logger.info(f"Converted {file_name} to {csv_output_path}")

        logger.success("Parquet -> CSV conversion complete.")

    def get_dataset_name(self) -> str:
        """
        Returns the name of the current dataset being used.
        
        Returns:
            str: The name of the dataset (e.g., 'clean-aug-soft-k5000').
        """
        return self.dataset_path.name

    def get_dataset(self) -> DatasetDict:
        """
        Returns the processed dataset from the interim data directory
        as a DatasetDict (loaded from CSVs).

        Returns:
            DatasetDict: The complete dataset with train and test splits for each language.
        """

        dataset_path = self.dataset_path

        # Define the base filenames
        data_files = {
            "java_train": str(dataset_path / "java_train.csv"),
            "java_test": str(dataset_path / "java_test.csv"),
            "python_train": str(dataset_path / "python_train.csv"),
            "python_test": str(dataset_path / "python_test.csv"),
            "pharo_train": str(dataset_path / "pharo_train.csv"),
            "pharo_test": str(dataset_path / "pharo_test.csv"),
        }

        # Verify file existence before loading
        logger.info("Loading CSV dataset from splits...")
        existing_data_files = {}
        for key, path in data_files.items():
            if not os.path.exists(path):
                found = False
                if os.path.exists(dataset_path):
                    for f in os.listdir(dataset_path):
                        if f.startswith(key) and f.endswith(".csv"):
                            existing_data_files[key] = str(dataset_path / f)
                            found = True
                            break
                if not found:
                    logger.warning(f"File not found for split '{key}': {path}")
            else:
                existing_data_files[key] = path

        if not existing_data_files:
            logger.error("No dataset CSV files found. Run 'parquet-to-csv' first.")
            raise FileNotFoundError("Dataset CSV files not found.")

        logger.info(f"Found files: {list(existing_data_files.keys())}")

        full_dataset = load_dataset("csv", data_files=existing_data_files)

        logger.info("Formatting labels (from string back to list)...")
        for split in full_dataset:
            full_dataset[split] = full_dataset[split].map(
                lambda x: {
                    "labels": ast.literal_eval(x["labels"])
                    if isinstance(x["labels"], str)
                    else x["labels"]
                }
            )

        logger.success("Dataset is ready for use.")
        return full_dataset

    def get_raw_dataset_from_hf(self) -> DatasetDict:
        """
        Loads the raw dataset directly from Hugging Face without saving.

        Returns:
            DatasetDict: The raw dataset from Hugging Face.
        """
        logger.info(f"Loading raw dataset '{self.hf_id}' from Hugging Face...")
        try:
            ds = load_dataset(self.hf_id)
            logger.success(f"Successfully loaded '{self.hf_id}'.")
            return ds
        except Exception as e:
            logger.error(f"Failed to load dataset from Hugging Face: {e}")
            return None

    def search_file(self, file_name: str, search_directory: Path = None) -> list:
        """
        Recursively searches for a file by name within a specified data directory.

        Args:
            file_name (str): The name of the file to search for (e.g., "java_train.csv").
            search_directory (Path, optional): The directory to search in.
                                              Defaults to self.raw_data_dir.

        Returns:
            list: A list of Path objects for all found files.
        """
        if search_directory is None:
            search_directory = self.raw_data_dir
            logger.info(f"Defaulting search to raw data directory: {search_directory}")

        if not search_directory.is_dir():
            logger.error(f"Search directory not found: {search_directory}")
            return []

        logger.info(f"Searching for '{file_name}' in '{search_directory}'...")

        found_files = []
        for root, dirs, files in os.walk(search_directory):
            for file in files:
                if file == file_name:
                    found_files.append(Path(root) / file)

        if not found_files:
            logger.warning(f"No files named '{file_name}' found in '{search_directory}'.")
        else:
            logger.success(f"Found {len(found_files)} matching file(s).")

        return found_files
