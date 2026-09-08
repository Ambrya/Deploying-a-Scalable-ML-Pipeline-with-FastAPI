import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.

    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)

    return precision, recall, fbeta


def inference(model, X):
    """
    Runs model inference and returns the predictions.

    Inputs
    ------
    model
        Trained machine learning model.
    X : np.array
        Data used for prediction.

    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model, path):
    """
    Serializes a model or encoder to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path where the pickle file will be saved.
    """
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path):
    """
    Loads a pickle file from the specified path and returns it.
    """
    with open(path, "rb") as file:
        return pickle.load(file)


def performance_on_categorical_slice(
    data,
    column_name,
    slice_value,
    categorical_features,
    label,
    encoder,
    lb,
    model,
):
    """
    Computes model performance on a categorical data slice.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label.
    column_name : str
        Name of the feature used to create the slice.
    slice_value : str, int, or float
        Value used to select the slice.
    categorical_features : list
        Names of the categorical features.
    label : str
        Name of the label column.
    encoder
        Trained OneHotEncoder.
    lb
        Trained LabelBinarizer.
    model
        Trained machine learning model.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    slice_data = data[data[column_name] == slice_value]

    X_slice, y_slice, _, _ = process_data(
        slice_data,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)

    return precision, recall, fbeta