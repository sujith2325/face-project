import requests

url = "http://127.0.0.1:5000/register"
files = {'image': open('img1.jpeg', 'rb')}
data = {'name': 'test_person'}

response = requests.post(url, files=files, data=data)
print(response.status_code)
print(response.text)
