''' Testing performance of the model on slices of the data'''

import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from data import process_data
from model import compute_model_metrics, inference

# init loggingfile
logging.basicConfig(filename='slice_performance.log', level=logging.INFO, filemode='w')
logging.info('------ Starting testing slice performance ----')

# load data
logging.info('Loading data')
raw_data = pd.read_csv("./data/census.csv")

# load model and encoder
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
lb = pickle.load(open('lb.pkl', 'rb'))

# split data into train and test
# note: random_state is set to 42 to ensure that the same split is used for all slices
logging.info('Splitting data into train and test')
_ , test_data = train_test_split(raw_data, test_size=0.20, random_state=42)

# define slices
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

for category in cat_features:

    for cls in test_data[category].unique():
       
       

        df_test = test_data[test_data[category] == cls]
        catx_test, caty_test, _, _ = process_data(
                        df_test,
                        cat_features,
                        label= "salary", encoder=encoder, lb=lb, training=False)

    
        preds = inference(model, catx_test)
        metrics = compute_model_metrics(caty_test, preds)

        
        # Evaluate the model on each slice and save the metrics.
        logging.info('Evaluating model for category and class: %s - %s', category,cls)
        logging.info('-----------------------')
        logging.info('Fbeta: %s', metrics[0])
        logging.info('Precision: %s', metrics[1])
        logging.info('Recall: %s', metrics[2])
    

