from pathlib import Path

import pytest

# Project modules are importable thanks to conftest.py
import turing.config as config
from turing.dataset import DatasetManager


@pytest.mark.data_loader
class TestDatasetManager:
    """
    Unit tests for the DatasetManager class.
    This test suite validates initialization, data transformation logic,
    and data loading mechanisms, including error handling.
    """

    def test_initialization_paths_are_correct(self, manager: DatasetManager):
        """
        Verifies that the DatasetManager initializes with the correct
        Hugging Face ID and constructs its paths as expected.
        """
        assert manager.hf_id == "NLBSE/nlbse26-code-comment-classification"
        assert "data/raw" in str(manager.raw_data_dir)
        # base_interim_path should contain either 'base' or 'features'
        path_str = str(manager.base_interim_path)
        assert "data/interim" in path_str and ("base" in path_str or "features" in path_str)

    @pytest.mark.parametrize(
        "input_labels, expected_output",
        [
            ([1, 0, 1], "[1, 0, 1]"),  # Case: Standard list
            ("[1, 0, 1]", "[1, 0, 1]"),  # Case: Already a string
            ([], "[]"),  # Case: Empty list
            (None, None),  # Case: None value
        ],
    )
    def test_format_labels_for_csv(self, manager: DatasetManager, input_labels, expected_output):
        """
        Tests the internal _format_labels_for_csv method to ensure
        it correctly serializes label lists (or handles other inputs) to strings.
        """
        # Arrange
        example = {"labels": input_labels}

        # Act
        formatted_example = manager._format_labels_for_csv(example)

        # Assert
        assert formatted_example["labels"] == expected_output

    def test_get_dataset_raises_file_not_found(self, monkeypatch):
        """
        Ensures that get_dataset() raises a FileNotFoundError when
        the target interim CSV files do not exist.
        """
        # Arrange
        # Patch the config to point to a non-existent directory
        fake_dir = Path("/path/that/is/totally/fake")
        monkeypatch.setattr(config, "INTERIM_DATA_DIR", fake_dir)

        # Manager must be initialized *after* patching config
        manager_with_fake_path = DatasetManager()

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="Dataset CSV files not found."):
            manager_with_fake_path.get_dataset()

    def test_get_dataset_success_and_label_parsing(self, fake_csv_data_dir: Path, monkeypatch):
        """
        Verifies that get_dataset() successfully loads data from mock CSVs
        and correctly parses the string-formatted labels back into lists.
        """
        # Arrange
        # Point the config at our temporary fixture directory
        monkeypatch.setattr(config, "INTERIM_DATA_DIR", fake_csv_data_dir)
        manager = DatasetManager()

        # Act
        dataset = manager.get_dataset()

        # Assert
        # Check that the correct splits were loaded
        assert "java_train" in dataset
        assert "java_test" in dataset
        assert "python_train" not in dataset  # Confirms only found files are loaded

        # Check content integrity
        assert len(dataset["java_train"]) == 2
        assert dataset["java_train"][0]["combo"] == "java code text"

        # Ccheck that the string '[1, 0, ...]' was parsed back to a list
        expected_labels = [1, 0, 0, 0, 0, 0, 0]
        assert dataset["java_train"][0]["labels"] == expected_labels
        assert isinstance(dataset["java_train"][0]["labels"], list)
