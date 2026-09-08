import requests


url = "http://127.0.0.1:8000"

# Send a GET request.
response = requests.get(url)

print(f"Status Code: {response.status_code}")
print(f"Result: {response.json()['message']}")


data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST request.
response = requests.post(f"{url}/data/", json=data)

print(f"Status Code: {response.status_code}")
print(f"Result: {response.json()['result']}")