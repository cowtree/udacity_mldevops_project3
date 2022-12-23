''' Testing performance of the model on test dataset'''

import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from data import process_data
from model import compute_model_metrics


# init loggingfile
logging.basicConfig(filename='model_performance.log', level=logging.INFO, format='%(asctime)s %(message)s')

# load data
logging.info('Loading data')
raw_data = pd.read_csv("./data/census.csv")

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

# load model and encoder
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
lb = pickle.load(open('lb.pkl', 'rb'))

# split data into train and test
# note: random_state is set to 42 to ensure that the same split is used for all slices
logging.info('Splitting data into train and test')
_, test_data = train_test_split(raw_data, test_size=0.20, random_state=42)

x_test, y_test, _, _ = process_data(
                test_data,
                cat_features,
                label= "salary", encoder=encoder, lb=lb,  training=False)


# Evaluate the model and save the metrics.
logging.info('Evaluating model')
preds = model.predict(x_test)
metrics = compute_model_metrics(y_test, preds)
logging.info('Saving metrics')
logging.info('-----------------------')
logging.info('Fbeta: %s', metrics[0])
logging.info('Precision: %s', metrics[1])
logging.info('Recall: %s', metrics[2])
