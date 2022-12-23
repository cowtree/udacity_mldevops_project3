'''Pytest fixtures for testing.'''
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data

# define pytest fixtures
@pytest.fixture(scope="session")
def prep_train_data():
    """Prepare data for machine learning model."""
    data = pd.read_csv("/Users/moscao/Projects/udacity_mlops/project3/udacity_mldevops_project3/data/census.csv")
    train, _ = train_test_split(data, test_size=0.20)

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

    x_train, y_train, _, _ = process_data(
    train, categorical_features=cat_features, label="salary", training=True
        )

    return x_train, y_train
