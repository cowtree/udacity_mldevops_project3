# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pickle
import logging

# Add the necessary imports for the starter code.
import pandas as pd
from model import train_model, compute_model_metrics, inference
from data import process_data

# init loggingfile
logging.basicConfig(filename='train_model.log', level=logging.INFO)


# Add code to load in the data.
data = pd.read_csv("/Users/moscao/Projects/udacity_mlops/project3/udacity_mldevops_project3/data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
logging.info('Splitting data into train and test')
train, test = train_test_split(data, test_size=0.20, random_state=42)

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

# Proces the test data with the process_data function.
logging.info('Processing data')
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
logging.info('Training model')
model = train_model(X_train, y_train)
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(encoder, open('encoder.pkl', 'wb'))
pickle.dump(lb, open('lb.pkl', 'wb'))
