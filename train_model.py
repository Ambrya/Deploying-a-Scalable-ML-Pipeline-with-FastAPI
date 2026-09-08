import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)


# Establish paths relative to this project.
project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "data", "census.csv")

# Load the Census data and remove spaces after commas.
data = pd.read_csv(data_path, skipinitialspace=True)

# Split the data into training and testing datasets.
train, test = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["salary"],
)

# Categorical features.
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the training data.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Process the test data using the fitted encoder and label binarizer.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model.
model = train_model(X_train, y_train)

# Save the model, encoder, and label binarizer.
model_path = os.path.join(project_path, "model", "model.pkl")
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
lb_path = os.path.join(project_path, "model", "lb.pkl")

save_model(model, model_path)
save_model(encoder, encoder_path)
save_model(lb, lb_path)

print(f"Model saved to {model_path}")
print(f"Encoder saved to {encoder_path}")
print(f"Label binarizer saved to {lb_path}")

# Load the saved model.
model = load_model(model_path)

# Run inference on the test data.
preds = inference(model, X_test)

# Calculate and print overall performance.
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Clear the previous slice output before writing new results.
slice_output_path = os.path.join(project_path, "slice_output.txt")

with open(slice_output_path, "w") as output_file:
    for col in cat_features:
        for slice_value in sorted(test[col].unique()):
            count = test[test[col] == slice_value].shape[0]

            p, r, fb = performance_on_categorical_slice(
                data=test,
                column_name=col,
                slice_value=slice_value,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model,
            )

            print(
                f"{col}: {slice_value}, Count: {count:,}",
                file=output_file,
            )
            print(
                f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}",
                file=output_file,
            )

print(f"Slice performance saved to {slice_output_path}")