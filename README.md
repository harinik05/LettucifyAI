# LettucifyAI


https://github.com/harinik05/LettucifyAI/assets/63025647/dc4534b5-e4e6-4969-87f1-a7e62533865c


LettucifyAI is your best AI friend, when it comes to managing food items in your own household refridgerator. This project employs the use of Azure Machine Learning Studio and pipelines facilitated via Azure DevOps to build an end-to-end MLOps solution for this image classification scenario. Various libraries part of Python such as Seaborn and Matplotlib were used for plotting the necessary confusion matrixes and ROC curves. Using the PyTorch environment, a modularized and fine-tuned CNN was built to perform classification using a distributed GPU training mechanism in NVIDIA Tesla GPU. This split the workload amongst the same GPU through multiple nodes (VMs) within Azure, improving on the overall performance and scalability of systems. 

## ⛰️ Functionalities
- **Supervised Machine Learning**: Image classification for various food items will be done via a trained CNN. This employs the use of labels in the training set for completing the ML job.
- **Distributed GPU Training**: To perform distributed GPU training, you specify a distributed training configuration in your Azure ML experiment. This configuration includes details such as the number of nodes, the type of GPU VMs to use, and the distribution strategy (e.g., data parallelism or model parallelism).
- **Logging and Metrics**: During training, Azure ML provides monitoring and logging capabilities to track metrics such as loss, accuracy, and training time. You can view these metrics in real-time using Azure ML's interface or programmatically access them for further analysis
- **Model Registration and Deployment**: After training is complete, you can register the trained model in your Azure ML workspace and deploy it as a web service for inference.

## 🎡 Process
![image](https://github.com/harinik05/LettucifyAI/assets/63025647/1225921c-c26a-4ad8-8500-ec9e10dc22c0)

### Configuring GPU Clusters
The accelerated GPU training for this model is done through multi-node, multi-gpu pytorch job. The distributed training job splits the workload uniformly, allowing the AzureML client to connect to the GPU clusters made in Azure VM Scale Set. The minimum number of nodes are set to 0, and the max is 5 in the `STANDARD_NC6s_v3` family. 
### Training via Convolutional Neural Network (CNN)
There is a use of OOP (Inheritance) from nn.Module, which is the base parent class for all Convolutional Neural Networks (CNN). The convolutional layer is a set of sequential stacks for convolution and activations in the neural network. This gets flattened out to the fully-connected layer and feeds to the output. The dimensionality starts with 16 feature maps, reducing each time with max pooling layers. The output is 16 feature maps from the fully-connected layers, which corresponds to the number of classes. 
### 🛸 Deployment
The model is then registered locally via AzureML studio. Then, the training job is set to run all the tasks in ML client. To complete the inference and do the testing, an online endpoint is created. Then, a validation set is made to batch-test the provided data showing a remarkable 85% accuracy of the model that we trained!
- **Online Endpoint**: A blue-green deployment was performed on the batch. Then, there is an inference ran to test if the image classification is done correctly. It will provide the predicted label in the response JSON, making it easy to test validation set
- **Kubernetes (Container Orchestration)**: Deployment file is set, then kubectl commands are used along with an Application Load Balancer (ALB) to efficiently expose the Kubernetes clusters and pods to the outside world. For the sake of testing, only one pod was used. Then, pytest was run to hit this endpoint and get inference for the corresponding values. 

### 🎱 Future Steps
1. Improve upon container orchestration (Kubernetes), by adding a greater count of pods
2. Produce graphs and results for validation set

