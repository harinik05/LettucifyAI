###################################
# Name: Harini Karthik
# File_Name: Model_Training_Sequence
# Parent_File: Training.py
# Date: Feb. 23, 2024
# Reference: AzureML Sample Directory
###################################

# Import the required libraries for proj
import os 
import time 
import json 
import logging 
import pickle
import argparse 
import mlflow
from tqdm import tqdm 
from distutils.util import strobool 

# Import the torch libraries 
import torch 
import torch.nn as nn 
import torch.optim as optim
import torchvision 
from torch.optim import lr_scheduler 
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.profiler import record_function 

# Internal imports
from model import load_model, MODEL_ARCH_LIST
from image_io import build_image_datasets
from profiling import PyTorchProfilerHandler 

# Define a class for the Model_Training_Sequence
class Model_Training_Sequence:

    # Constructor 
    def __init__(self):
        # Intialize the logger object 
        self.logger = logging.getLogger(__name__)

        # Data
        self.training_data_sampler = None
        self.training_data_loader = None
        self.validation_data_loader = None

        # Model 
        self.model = None
        self.labels = []
        self.model_signature = None 

        # Distributed GPU Training Configuration 
        self.world_size = 1
        self.world_rank = 0
        self.local_world_size = 1
        self.local_rank = 0
        self.multinode_available = False 
        self.cpu_count = os.cpu_count()
        self.device = None
        self.self_is_main_node = True #flag to tell if ur in first node 

        # Training Configs 
        self.dataloading_config = None 
        self.training_config = None 

        # Profiler 
        self.profiler = None 
        self.profiler_output_tmp_dir = None 

    # Setup Configuration Function 
    def setup_config(self, args):
        

    


