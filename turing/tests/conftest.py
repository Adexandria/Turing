import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

import turing.config as config
from turing.dataset import DatasetManager
from turing.reporting import TestReportGenerator

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(proj_root)

train_dir = os.path.join(proj_root, "turing", "modeling")
sys.path.insert(1, train_dir)


try:
    # Import train.py
    import turing.modeling.train as train
except ImportError as e:
    pytest.skip(
        f"Could not import 'train.py'. Check sys.path. Error: {e}", allow_module_level=True
    )

# --- Reporting Setup ---
execution_results = []
active_categories = set()


def clean_test_name(nodeid):
    """Pulisce il nome del test rimuovendo parametri lunghi."""
    parts = nodeid.split("::")
    test_name = parts[-1]
    if len(test_name) > 50:
        test_name = test_name[:47] + "..."
    return test_name


def format_error_message(long_repr):
    """Estrae solo l'errore principale."""
    if not long_repr:
        return ""
    lines = str(long_repr).split("\n")
    last_line = lines[-1]
    clean_msg = last_line.replace("|", "-").strip()
    if len(clean_msg) > 60:
        clean_msg = clean_msg[:57] + "..."
    return clean_msg


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == "call":
        path_str = str(item.fspath)
        category = "GENERAL"

        if "unit" in path_str:
            category = "UNIT"
        elif "behavioral" in path_str:
            category = "BEHAVIORAL"
        elif "modeling" in path_str:
            category = "MODELING"

        active_categories.add(category)

        # Simplified status mapping
        status_map = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}
        status_str = status_map.get(report.outcome, report.outcome.upper())

        execution_results.append(
            {
                "Category": category,
                "Module": item.fspath.basename,
                "Test Case": clean_test_name(item.nodeid),
                "Result": status_str,
                "Time": f"{report.duration:.2f}s",
                "Message": format_error_message(report.longrepr) if report.failed else "",
            }
        )


def pytest_sessionfinish(session, exitstatus):
    """Generate enhanced test report at session end."""
    if not execution_results:
        return

    report_type = (
        f"{list(active_categories)[0].lower()}_tests"
        if len(active_categories) == 1
        else "unit_and_behavioral_tests"
    )

    try:
        manager = TestReportGenerator(context_name="turing", report_category=report_type)
        
        # Main title
        manager.add_header("Turing Test Execution Report")
        manager.add_divider("section")
        
        # Environment info
        manager.add_environment_metadata()
        manager.add_divider("thin")

        df = pd.DataFrame(execution_results)

        # Sommario
        total = len(df)
        passed = len(df[df["Result"] == "[ PASS ]"])
        failed = len(df[df["Result"] == "[ FAILED ]"])
        summary = pd.DataFrame(
            [
                {
                    "Total": total,
                    "Passed": passed,
                    "Failed": failed,
                    "Success Rate": f"{(passed / total) * 100:.1f}%",
                }
            ]
        )
        manager.add_dataframe(summary, title="Executive Summary")

        # Detailed breakdown by category
        cols = ["Module", "Test Case", "Result", "Time", "Message"]
        
        if len(active_categories) > 1:
            manager.add_header("Detailed Test Results by Category", level=2)
            manager.add_divider("thin")
            
            for cat in sorted(active_categories):
                subset = df[df["Category"] == cat][cols]
                manager.add_dataframe(subset, title=f"{cat} Tests")
        else:
            manager.add_alert_box(
                "All tests passed successfully!",
                box_type="success"
            )

        manager.save("report.md")
    except Exception as e:
        print(f"\nError generating report: {e}")


# --- Fixtures ---


@pytest.fixture(scope="function")
def manager() -> DatasetManager:
    """
    Provides a instance of DatasetManager for each test.
    """
    return DatasetManager()


@pytest.fixture(scope="function")
def fake_csv_data_dir(tmp_path: Path) -> Path:
    """
    Creates a temporary directory structure mocking 'data/interim/features/clean-aug-soft-k5000'
    and populates it with minimal, valid CSV files for testing.

    Returns:
        Path: The path to the *parent* of 'features' (e.g., the mocked INTERIM_DATA_DIR).
    """
    interim_dir = tmp_path / "interim_test"
    features_dir = interim_dir / "features" / "clean-aug-soft-k5000"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Define minimal valid CSV content
    csv_content = (
        "combo,labels\n"
        '"java code text","[1, 0, 0, 0, 0, 0, 0]"\n'
        '"other java code","[0, 1, 0, 0, 0, 0, 0]"\n'
    )

    # Write mock files
    (features_dir / "java_train.csv").write_text(csv_content)
    (features_dir / "java_test.csv").write_text(csv_content)

    # Return the root of the mocked interim directory
    return interim_dir


@pytest.fixture(scope="session")
def mock_data():
    """
    Provides a minimal, consistent, session-scoped dataset for model testing.
    This simulates the (X, y) data structure used for training and evaluation.
    """
    X = [
        "this is java code for summary",
        "python is great for parameters",
        "a java example for usage",
        "running python script for development notes",
        "pharo is a language for intent",
        "another java rational example",
    ]

    # Mock labels for a 'java' model (7 categories)
    # Shape (6 samples, 7 features)
    y = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 1],
        ]
    )
    return {"X": X, "y": y}


@pytest.fixture(scope="module")
def trained_rf_model(mock_data, tmp_path_factory):
    """
    Provides a fully-trained RandomForestTfIdf model instance.
    """
    # Import locally to ensure proj_root is set
    from modeling.models.randomForestTfIdf import RandomForestTfIdf

    # Arrange
    model = RandomForestTfIdf(language="java")

    # Monkeypatch grid search parameters for maximum speed
    model.grid_params = {
        "tfidf__max_features": [10, 20],  # Use minimal features
        "clf__estimator__n_estimators": [2, 5],  # Use minimal trees
    }
    model.params["cv_folds"] = 2  # Use minimal CV folds

    # Create a persistent temp dir for this module's run
    model_path = tmp_path_factory.mktemp("trained_rf_model")

    # Act: Train the model
    model.train(mock_data["X"], mock_data["y"], path=str(model_path), model_name="test_model")

    # Yield the trained model and its save path
    yield model, model_path


MODEL_CLASS_TO_TEST = train.MODEL_CLASS
MODEL_EXPERIMENT_NAME = train.EXP_NAME
MODEL_NAME_BASE = train.MODEL_NAME


@pytest.fixture(scope="session")
def get_predicted_labels():
    def _helper(model, comment_sentence: str, lang: str) -> set:
        if config.INPUT_COLUMN == "combo":
            combo_input = f"DummyClass.{lang} | {comment_sentence}"
            input_data = [combo_input]
        else:
            input_data = [comment_sentence]

        prediction_array = model.predict(input_data)[0]
        labels_map = config.LABELS_MAP[lang]
        predicted_labels = {labels_map[i] for i, val in enumerate(prediction_array) if val == 1}
        return predicted_labels

    return _helper


@pytest.fixture(scope="module")
def java_model():
    """Loads the Java model from the config path"""
    model_path = os.path.join(config.MODELS_DIR, MODEL_EXPERIMENT_NAME, f"{MODEL_NAME_BASE}_java")
    if not os.path.exists(model_path):
        pytest.skip(
            "Production model not found. Skipping behavioral tests for Java.",
            allow_module_level=True,
        )
    return MODEL_CLASS_TO_TEST(language="java", path=model_path)


@pytest.fixture(scope="module")
def python_model():
    """Loads the Python model from the config path"""
    model_path = os.path.join(
        config.MODELS_DIR, MODEL_EXPERIMENT_NAME, f"{MODEL_NAME_BASE}_python"
    )
    if not os.path.exists(model_path):
        pytest.skip(
            "Production model not found. Skipping behavioral tests for Python.",
            allow_module_level=True,
        )
    return MODEL_CLASS_TO_TEST(language="python", path=model_path)


@pytest.fixture(scope="module")
def pharo_model():
    """Loads the Pharo model from the config path"""
    model_path = os.path.join(config.MODELS_DIR, MODEL_EXPERIMENT_NAME, f"{MODEL_NAME_BASE}_pharo")
    if not os.path.exists(model_path):
        pytest.skip(
            "Production model not found. Skipping behavioral tests for Pharo.",
            allow_module_level=True,
        )
    return MODEL_CLASS_TO_TEST(language="pharo", path=model_path)
