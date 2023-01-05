'''Test API endpoints.'''

import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_read_root():
    '''Test the root endpoint.'''
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello world"

def test_predict_lowsalary(prep_api_data_lowsalary):
    '''Test the predict endpoint for label low salary'''
    data = json.dumps(prep_api_data_lowsalary)
    print(data)
    response = client.post("/predict", data = data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}

def test_predict_highsalary(prep_api_data_highsalary):
    '''Test the predict endpoint for label high salary'''
    data = json.dumps(prep_api_data_highsalary)
    print(data)
    response = client.post("/predict", data = data)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}