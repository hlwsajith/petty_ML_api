import requests

# Specify the URL of your FastAPI server
api_url = "http://localhost:7000"  # Replace with the actual URL of your server

# Prepare the image file (replace 'dog.jpg' with your image file)
files = {'file': ('dog.jpg', open('dog.jpg', 'rb'))}

# Send the POST request
response = requests.post(f"{api_url}/predict/", files=files)

# Check the response
if response.status_code == 200:
    print("Prediction Result:", response.json())
else:
    print("Error:", response.text)
