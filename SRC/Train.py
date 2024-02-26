import os
import argparse
import pandas as pd
import mlflow
import mlflow.pytorch
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from Model import CNNModel  # Import the CNNModel from the model module

def main():
    """Main function of the script."""
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--num_workers", type=int, help="number of workers")
    parser.add_argument("--prefetch_factor", type=int, help="prefetch factor")
    parser.add_argument("--model_arch", type=str, help="model architecture")
    parser.add_argument("--model_arch_pretrained", type=bool, help="whether to use pretrained model architecture")
    parser.add_argument("--num_epochs", type=int, help="number of epochs")
    parser.add_argument("--learning_rate", type=float, help="learning rate")
    parser.add_argument("--momentum", type=float, help="momentum")
    parser.add_argument("--register_model_as", type=str, help="model registration name")
    parser.add_argument("--enable_profiling", type=bool, help="whether to enable profiling")
    args = parser.parse_args()
   
    # Initialize distributed backend
    torch.distributed.init_process_group(backend='nccl')

    # Start Logging
    mlflow.start_run()

    ###################
    #<prepare the data>
    ###################
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)
    
    # Assuming you have a train dataset in args.data directory
    train_dataset = torchvision.datasets.ImageFolder(
        root=args.data,
        transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Resize((512, 512))
        ])
    )

    # Use DistributedSampler for distributed training
    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                              num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    ####################
    #</prepare the data>
    ####################

    ##################
    #<train the model>
    ##################
    model = CNNModel()
    
    # Wrap model with DistributedDataParallel
    model = DistributedDataParallel(model)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    # Lists to store true labels and predictions
    all_labels = []
    all_predictions = []

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Collect true labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(torch.argmax(outputs, axis=1).cpu().numpy())

    ###################
    #</train the model>
    ###################

    # Save the trained model
    torch.save(model.state_dict(), "trained_model.pth")

    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
