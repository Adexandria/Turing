import inspect

import numpy as np
import pytest

from turing.config import EXISTING_MODELS
import turing.modeling.models as my_models


@pytest.fixture
def get_model(request: str):
    """Fixture that returns a list of existing model names."""
    model_name = request.param

    module = getattr(my_models, model_name, None)

    classes = [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
    ]

    cls = classes[0]

    from turing.config import LANGS

    lang = LANGS[0]
    return cls(language=lang)


@pytest.mark.parametrize("get_model", EXISTING_MODELS, indirect=True)
def test_model_initialization(get_model):
    """
    Test that each model class can be initialized without errors.
    """
    model = get_model
    assert model is not None
    from turing.modeling.baseModel import BaseModel

    assert isinstance(model, BaseModel)


@pytest.mark.parametrize("get_model", EXISTING_MODELS, indirect=True)
def test_model_setup(get_model):
    """
    Test that each model class sets up its internal model correctly.
    """
    model = get_model
    model.setup_model()
    assert model.model is not None


@pytest.mark.parametrize("get_model", EXISTING_MODELS, indirect=True)
def test_model_train(tmp_path, get_model):
    """
    Test that each model class can run the train method without errors.
    """
    model = get_model
    model.setup_model()

    # Using mock data for training
    X_train = ["sample text data"] * 10

    y_train = [0, 1] * 5

    y_train = np.array(y_train).reshape(-1, 1)

    # fake directory and model name
    fake_path = tmp_path / "out"
    fake_path.mkdir()

    parameters = model.train(X_train, y_train)

    assert isinstance(parameters, dict)
    assert model.model is not None


@pytest.mark.parametrize("get_model", EXISTING_MODELS, indirect=True)
def test_model_evaluate(tmp_path, get_model):
    """
    Test that each model class can run the evaluate method without errors.
    """
    model = get_model
    model.setup_model()

    # Using mock data for training
    X_train = ["sample text data"] * 10

    y_train = [0, 1] * 5

    y_train = np.array(y_train).reshape(-1, 1)

    # fake directory and model name
    fake_path = tmp_path / "out"
    fake_path.mkdir()

    _ = model.train(X_train, y_train)

    # Using mock data for evaluation
    X_test = ["sample text data"] * 10
    y_test = [0, 1] * 5
    metrics = model.evaluate(X_test, y_test)

    assert isinstance(metrics, dict)
    assert metrics and "accuracy" in metrics
    assert "f1_score" in metrics or "f1_score_micro" in metrics


@pytest.mark.parametrize("get_model", EXISTING_MODELS, indirect=True)
def test_model_predict(tmp_path, get_model):
    """
    Test that each model class can run the predict method without errors.
    """
    model = get_model
    model.setup_model()

    # Using mock data for training
    X_train = ["sample text data"] * 10

    y_train = [0, 1] * 5

    y_train = np.array(y_train).reshape(-1, 1)

    # fake directory and model name
    fake_path = tmp_path / "out"
    fake_path.mkdir()

    _ = model.train(X_train, y_train)

    # Using mock data for prediction
    X_input = ["sample text data"] * 3
    predictions = model.predict(X_input)

    assert predictions is not None
    assert len(predictions) == len(X_input)
