# udacity_mldevops_project3
Deploying a Machine Learning Model on Heroku with FastAPI in order to predict (binrary classification) the salary of an employee based on some metrics

##
[Github Link](https://github.com/cowtree/udacity_mldevops_project3)

## Content in this project
- Data version control using DVC
- Implementation of CI/CD pipeline 
- Machine Learning Model training and inference
- FASTAPI application for model inference
- Deployment on Heroku with automatic deployment after successful CI

## Folder structure
```
├── README.md
├── __pycache__
│   └── main.cpython-38.pyc
├── data
│   ├── census.csv
│   └── census.csv.bak
├── data.dvc
├── eda_cencus.ipynb
├── encoder.pkl
├── lb.pkl
├── main.py
├── ml
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   ├── data.cpython-38.pyc
│   │   └── model.cpython-38.pyc
│   ├── data.py
│   ├── model.py
│   ├── model_performance.py
│   ├── slice_performance.py
│   └── train_model.py
├── model
├── model.pkl
├── model_card.md
├── model_performance.log
├── post_request.py
├── sanitycheck.py
├── screenshots
├── slice_performance.log
├── tests
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   ├── conftest.cpython-38-pytest-7.2.0.pyc
│   │   ├── test_api.cpython-38-pytest-7.2.0.pyc
│   │   ├── test_api.cpython-38.pyc
│   │   └── test_model.cpython-38-pytest-7.2.0.pyc
│   ├── conftest.py
│   ├── test_api.py
│   └── test_model.py
└── train_model.log
```

## Data exploration
 - ```eda_census.ipynb``` [notebook](eda_census.ipynb) contains some in-depth look into the data 
 - Whitespaces in the datasheet was removed using:
    ```sed -i.bak 's/[[:space:]]*,[[:space:]]*/,/g' data/census.csv```
 - The dataset is a table of employees with different attributes which are used to predict the salary in a binary fashion (<=50k || >50k):

```python
 
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num : int 
    marital_status : str 
    occupation : str
    relationship : str
    race : str
    sex : str
    capital_gain : int 
    capital_loss : int 
    hours_per_week : int 
    native_country  : str 
    salary: str
```


### Model
- In this example here, a random forest model was trained on the given dataset ([census dataset](/data/census.csv)) and 
- For more details on model performance, please have a look at the [model card](model_card.md)


 ### API
 - The API was impelemented using fastAPI framework
 - It consists of two functions:
    - **GET**: Display a welcome message at the root path
    - **POST**: Given input dataset of an employee predict the salary of the employee
 - Example of parameters of the [Example POST](example.png)   

### Tests
- Tets can be found in ```tests/``` folder
    - ```conftest.py``` contains data used for unit testing
    - ```test_api.py``` contains the tests of the **GET** and **POST** operations of the API
    - ```test_model.py``` contains unit tests for model folder


 