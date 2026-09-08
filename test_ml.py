import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)


def test_train_model():
    """Tests that train_model returns a fitted Random Forest model."""
    X_train = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "classes_")


def test_inference():
    """Tests that inference returns the expected prediction type and size."""
    X_train = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)
    predictions = inference(model, X_train)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == y_train.shape
    assert set(predictions).issubset({0, 1})


def test_compute_model_metrics():
    """Tests that the metric function returns the expected values."""
    y = np.array([1, 1, 0, 0])
    predictions = np.array([1, 0, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y, predictions)

    assert precision == 0.5
    assert recall == 0.5
    assert fbeta == 0.5


def test_save_and_load_model(tmp_path):
    """Tests that a saved model can be loaded and used for prediction."""
    X_train = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)
    model_path = tmp_path / "test_model.pkl"

    save_model(model, model_path)
    loaded_model = load_model(model_path)

    original_predictions = inference(model, X_train)
    loaded_predictions = inference(loaded_model, X_train)

    assert isinstance(loaded_model, RandomForestClassifier)
    assert np.array_equal(original_predictions, loaded_predictions)