# Use the official PyTorch image as the base image
FROM pytorch/pytorch:latest

# Install Flask and Pillow (PIL) for image processing
RUN pip install torch torchvision Pillow

# Create a directory for the application
WORKDIR /K8S

# Copy the Python script and the model file
COPY hello.py .

# Expose port 80 for the Flask application
EXPOSE 80

# Command to run the Flask application
CMD ["python", "hello.py"]
