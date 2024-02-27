from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

# Define the CNNModel class
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 128 * 128, 128),
            nn.ReLU(),
            nn.Linear(128, 16)  # Output size 16 for 16 classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Load the saved model
model = CNNModel()
model.load_state_dict(torch.load('CNN2.pth'))
model.eval()

# Define a function to preprocess the image
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize image to match input size of the model
        transforms.ToTensor(),           # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # Open the image and convert to RGB
    image = transform(image)  # Apply the transformation
    return image.unsqueeze(0)  # Add batch dimension

# Initialize Flask app
app = Flask(__name__)

# Define route to handle POST requests
@app.route('/your-endpoint', methods=['POST'])
def predict_image():
    # Check if request contains file data
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Get the file from the request
    file = request.files['file']
    
    try:
        # Read image file
        image_bytes = file.read()

        # Preprocess the image
        image = preprocess_image(image_bytes)

        # Perform inference
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        # Return the predicted class
        return jsonify({'predicted_class': predicted.item()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
