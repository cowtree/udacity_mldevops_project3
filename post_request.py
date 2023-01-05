import requests
import json

data_dict = {
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num" : 13,
    "marital-status" : "Never-married",
    "occupation" : "Adm-clerical",
    "relationship" : "Not-in-family",
    "race" : "White",
    "sex" : "Male",
    "capital-gain" : 2174,
    "capital-loss" : 0,
    "hours-per-week" : 40,
    "native-country"  : "United-States"

}

response = requests.post(url="https://mldevops-project3-prod.herokuapp.com/predict", json=data_dict)
print(response.status_code)
print(response.json())
