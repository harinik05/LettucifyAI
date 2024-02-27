import pytest
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

# Define a function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image

# Test function to check if the AKS endpoint is working correctly
def test_aks_endpoint():
    # Load an image for testing
    image_path = "test_image.jpg"
    image = preprocess_image(image_path)

    # Convert the PyTorch tensor to JSON format
    data = image.numpy().tolist()

    # Define the AKS endpoint URL
    endpoint = "YOUR_AKS_ENDPOINT_URL"

    # Send a POST request to the AKS endpoint
    response = requests.post(endpoint, json={"data": data})

    # Check if the request was successful (HTTP status code 200)
    assert response.status_code == 200

    # Check if the response contains the predicted class (you may need to adjust this based on your model output)
    assert "prediction" in response.json()

    # Print the predicted class
    print("Predicted class:", response.json()["prediction"])

if __name__ == "__main__":
    pytest.main()
