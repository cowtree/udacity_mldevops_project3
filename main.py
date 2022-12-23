'''FastAPI implementation of the API for model deployment on HeroKu'''

# Put the code for your API here.

# Add the necessary imports for the starter code.
import pickle
import pydantic
from fastapi import FastAPI, Request

from ml.model import inference
from ml.data import process_data


class employee(pydantic.BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str

app = FastAPI()

@app.get("/")
def read_root():
    """welcome message

    Returns:
        str: Welcome message
    """    
    return "Hello world"

# POST model inference endpoint
@app.post("/predict")
def predict(data: employee):
    """Performs an inference on a trained model with input data.

    Args:
        data (pd.DataFrame): Input data to the model.

    Returns:
        list: The model predictions.
    """
    # Load the model.
    model = pickle.load(open("model.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    lb = pickle.load(open("lb.pkl", "rb"))

    # Preprocess the input data.
    X, _, _, _ = process_data(data, encoder=encoder, lb=lb, training=False)

    # Get the model's predictions.
    preds = inference(model, X)

    # Return the predictions.
    return preds



