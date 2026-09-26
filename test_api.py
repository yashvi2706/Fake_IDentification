import requests
import json

url = "http://localhost:8000/api/analyze"

data = {
    "document_type": "passport"
}

files = {
    "document": ("dummy_passport.jpg", open("backend/dummy_passport.jpg", "rb"), "image/jpeg"),
    "face_image": ("dummy_selfie.jpg", open("backend/dummy_selfie.jpg", "rb"), "image/jpeg")
}

response = requests.post(url, data=data, files=files)
print(f"Status Code: {response.status_code}")
print(json.dumps(response.json(), indent=2))
