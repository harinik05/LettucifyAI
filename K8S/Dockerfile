FROM pytorch/pytorch:latest

# Copy your model file
COPY your_model.pth /app/model.pth

# Copy inference script
COPY inference_script.py /app/inference_script.py

# Install any additional dependencies
RUN pip install numpy

# Set the working directory
WORKDIR /app

# Command to run inference script
CMD ["python", "inference_script.py"]
