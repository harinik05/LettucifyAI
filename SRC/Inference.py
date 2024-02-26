%%writefile {train_src_dir}/Inference.py
import torch
import torchvision.transforms as transforms
from PIL import Image
from Model import CNNModel  # Import the CNNModel from the model module

class inference_script_class:
    def run_inference(image_path):
        # Load the trained model
        model = CNNModel()
        model.load_state_dict(torch.load("trained_model.pth"))
        model.eval()

        # Define transformations for input images
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        # Load and preprocess the input image
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(image)
            predicted_label = torch.argmax(output, dim=1).item()

        return predicted_label
