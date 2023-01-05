import requests

response = requests.get(url="https://mldevops-project3-prod.herokuapp.com/")
print(response.status_code)
print(response.json())