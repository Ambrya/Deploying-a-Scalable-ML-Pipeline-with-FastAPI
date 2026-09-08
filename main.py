import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(
        ..., example="United-States", alias="native-country"
    )


# Load the saved model and encoder.
project_path = os.path.dirname(os.path.abspath(__file__))
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
model_path = os.path.join(project_path, "model", "model.pkl")

encoder = load_model(encoder_path)
model = load_model(model_path)


# Create the FastAPI application.
app = FastAPI()


@app.get("/")
async def get_root():
    """Returns a welcome message."""
    return {"message": "Hello from the API!"}


@app.post("/data/")
async def post_inference(data: Data):
    """Returns an income prediction for the submitted Census data."""
    data_dict = data.dict()

    # Convert Python-compatible names back to the Census column names.
    data_for_prediction = {
        key.replace("_", "-"): [value]
        for key, value in data_dict.items()
    }
    data_frame = pd.DataFrame.from_dict(data_for_prediction)

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

    data_processed, _, _, _ = process_data(
        data_frame,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
    )

    prediction = inference(model, data_processed)

    return {"result": apply_label(prediction)}