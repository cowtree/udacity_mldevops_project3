'''FastAPI implementation of the API for model deployment on HeroKu'''

# Put the code for your API here.

# Add the necessary imports for the starter code.
import pickle
from pydantic import BaseModel, Field
from fastapi import FastAPI
import pandas as pd

from ml.model import inference
from ml.data import process_data


class Employee(BaseModel):
    """Employee data model"""

    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num : int = Field(None, alias="education-num")
    marital_status : str = Field(None,  alias="marital-status")
    occupation : str
    relationship : str
    race : str
    sex : str
    capital_gain : int = Field(None, alias="capital-gain")
    capital_loss : int =  Field(None, alias="capital-loss")
    hours_per_week : int = Field(None, alias="hours-per-week")
    native_country  : str = Field(None, alias="native-country")

class Prediction(BaseModel):
    """Prediction data model"""

    prediction: str


app = FastAPI()

@app.on_event("startup")
async def startup_event(): 
    """Load the model and encoder at startup"""
    global model, encoder, binarizer
    model = pickle.load(open("./model/model.pkl", "rb"))
    encoder = pickle.load(open("./model/encoder.pkl", "rb"))
    binarizer = pickle.load(open("./model/lb.pkl", "rb"))

@app.get("/")
def read_root():
    """welcome message

    Returns:
        str: Welcome message
    """ 
    return "Hello world"

# POST model inference endpoint
@app.post("/predict", response_model=Prediction, status_code=200)
async def predict(data: Employee):
    """Performs an inference on a trained model with input data.

    Args:
        data (pd.DataFrame): Input data to the model.

    Returns:
        dict: The model predictions.
    """
    # Convert the input data into a dataframe.
    data_df = pd.DataFrame([data.dict()])

    cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

    # Preprocess the input data.
    X, _, _, _ = process_data(data_df, encoder=encoder, categorical_features=cat_features, lb=binarizer, training=False)

    # Get the model's predictions.
    prediction = inference(model, X)

    if prediction == 0:
        prediction = "<=50K"
    else:
        prediction = ">50K"

    # Return the predictions.
    return {"prediction": prediction}

