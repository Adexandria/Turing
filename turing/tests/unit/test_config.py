import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the module to be tested
import turing.config as config


@pytest.mark.config
class TestConfig:
    """
    Test suite for validating the project's configuration module (config.py).

    These tests verify that paths are structured correctly, critical constants
    are of the expected type and value, and module-level logic
    (like calculations and .env loading) executes as intended.
    """

    def test_proj_root_is_correctly_identified(self):
        """
        Validates that PROJ_ROOT is a Path object and points to the
        actual project root directory (which should contain 'pyproject.toml').
        """
        assert isinstance(config.PROJ_ROOT, Path)
        assert config.PROJ_ROOT.is_dir()

        # A common "sanity check" is to look for a known file at the root
        expected_file = config.PROJ_ROOT / "pyproject.toml"
        assert expected_file.is_file(), (
            f"PROJ_ROOT ({config.PROJ_ROOT}) does not seem to be the project root. "
            f"Could not find {expected_file}"
        )

    def test_directory_paths_are_correctly_structured(self):
        """
        Ensures all key directory variables are Path objects
        and are correctly parented under PROJ_ROOT.
        """
        # List of all directory variables defined in config.py
        path_vars = [
            config.DATA_DIR,
            config.RAW_DATA_DIR,
            config.INTERIM_DATA_DIR,
            config.PROCESSED_DATA_DIR,
            config.EXTERNAL_DATA_DIR,
            config.MODELS_DIR,
            config.REPORTS_DIR,
            config.FIGURES_DIR,
        ]

        for path_var in path_vars:
            assert isinstance(path_var, Path)
            # Check that PROJ_ROOT is an ancestor of this path
            assert config.PROJ_ROOT in path_var.parents

        # Spot-check a few for correct relative paths
        assert config.DATA_DIR == config.PROJ_ROOT / "data"
        assert config.RAW_DATA_DIR == config.PROJ_ROOT / "data" / "raw"
        assert config.FIGURES_DIR == config.PROJ_ROOT / "reports" / "figures"

    def test_dataset_constants_are_valid(self):
        """
        Validates that critical dataset constants are non-empty and of
        the correct type.
        """
        assert isinstance(config.DATASET_HF_ID, str)
        assert config.DATASET_HF_ID == "NLBSE/nlbse26-code-comment-classification"

        assert isinstance(config.LANGS, list)
        assert len(config.LANGS) == 3
        assert "java" in config.LANGS

        assert isinstance(config.INPUT_COLUMN, str) and config.INPUT_COLUMN
        assert isinstance(config.LABEL_COLUMN, str) and config.LABEL_COLUMN

    def test_labels_map_and_total_categories_are_correct(self):
        """
        Validates the LABELS_MAP structure and ensures TOTAL_CATEGORIES
        is correctly calculated from it.
        """
        assert isinstance(config.LABELS_MAP, dict)

        # Ensure all languages in LANGS are keys in LABELS_MAP
        for lang in config.LANGS:
            assert lang in config.LABELS_MAP
            assert isinstance(config.LABELS_MAP[lang], list)
            assert len(config.LABELS_MAP[lang]) > 0

        # Validate the derived calculation
        expected_total = (
            len(config.LABELS_MAP["java"])
            + len(config.LABELS_MAP["python"])
            + len(config.LABELS_MAP["pharo"])
        )
        assert config.TOTAL_CATEGORIES == expected_total
        assert config.TOTAL_CATEGORIES == 18  # 7 + 5 + 6

    def test_numeric_parameters_are_positive(self):
        """
        Ensures that numeric scoring and training parameters are positive
        and of the correct type.
        """
        numeric_params = {
            "MAX_AVG_RUNTIME": config.MAX_AVG_RUNTIME,
            "MAX_AVG_FLOPS": config.MAX_AVG_FLOPS,
            "DEFAULT_BATCH_SIZE": config.DEFAULT_BATCH_SIZE,
            "DEFAULT_NUM_ITERATIONS": config.DEFAULT_NUM_ITERATIONS,
        }

        for name, value in numeric_params.items():
            assert isinstance(value, (int, float)), f"{name} is not numeric"
            assert value > 0, f"{name} must be positive"

    @patch("dotenv.load_dotenv")
    def test_load_dotenv_is_called_on_module_load(self, mock_load_dotenv):
        """
        Tests that the load_dotenv() function is executed when the
        config.py module is loaded.

        This requires reloading the module, as it's likely already been
        imported by pytest or conftest.
        """
        # Arrange (Patch is active)

        # Act
        # Reload the config module to trigger its top-level statements
        importlib.reload(config)

        # Assert
        # Check that the patched load_dotenv was called
        mock_load_dotenv.assert_called_once()
