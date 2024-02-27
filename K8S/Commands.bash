# Build Docker image
docker build -t my-pytorch-model .

# Tag the Docker image for ACR
docker tag my-pytorch-model myacr.azurecr.io/my-pytorch-model:v1

# Log in to ACR
az acr login --name myacr

# Push Docker image to ACR
docker push myacr.azurecr.io/my-pytorch-model:v1

# Deploy the Docker image to AKS
kubectl apply -f deployment.yaml

# Expose the deployed model as a service
kubectl expose deployment your-deployment-name --type=LoadBalancer --port=80 --target-port=80

# Use curl to send a request to the AKS endpoint
pytest test_aks_endpoint.py
