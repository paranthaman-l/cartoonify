import requests
import base64
from io import BytesIO
from PIL import Image

# Load and encode your image
with open('input/before.jpg', 'rb') as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

# Prepare the data to send in the POST request
data = {
    "image": f"data:image/png;base64,{encoded_image}",
    "model_id": 0,  # Change as needed
    "load_size": 800  # Change as needed
}

# Send the POST request to the local server
response = requests.post("http://127.0.0.1:2004/transform", json=data)

if response.status_code == 200:
    result = response.json()
    image_data = result["output"].split(",")[1]
    image_bytes = base64.b64decode(image_data)

    # Create an image from the bytes
    image = Image.open(BytesIO(image_bytes))
    image.show()  # This will open the default image viewer
else:
    print("Error:", response.json())