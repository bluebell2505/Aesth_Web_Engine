import requests

# if running locally:
url = "http://localhost:8000/api/extract-aesthetic"
files = {'file': open("tests/sample1.jpg","rb")}
r = requests.post(url, files=files, timeout=20)
print(r.status_code, r.json())
