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
    '''
    Number of workers: Refers to the number of worker processes that can be running for 
    data loading and training config job. If more than 0, then this will run paralleling 
    using multiple CPU cores to load data concurrently. 
    '''
    def setup_configuration(self, args):
        self.dataloading_config = args
        self.training_config = args 

        # Verify the parameter number of workers 
        if self.dataloading_config.num_workers is None:
            self.dataloading_config.num_workers = 0
        if self.dataloading_config.num_workers < 0:
            self.dataloading_config.num_workers = os.cpu_count()
        if self.dataloading_config.num_workers == 0:
            self.logger.warning("The number of workers is 0")
            self.dataloading_config.num_workers = None

        self.dataloading_config.pin_memory = bool(self.dataloading_config.pin_memory)
        self.dataloading_config.non_blocking = bool(self.dataloading_config.non_blocking)


        # Distributed: detect multinode config depending on Azure ML distribution type for DistributedDataParallel 
        self.distributed_backend = args.distributed_backend
        '''
        NCCL = NVIDIA Collective Communication Library for training across 
        multiple GPUs across a single node or across multiple nodes with GPUs
        from NVIDIA 

        MPI = Message Passing Interface for distributed computing
        '''
        if self.distributed_backend == "nccl":
            self.world_size = int(os.environ.get("WORLD_SIZE","1"))
            self.world_rank  =int(os.environ.get("RANK","0"))
            self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
            self.local_rank = int(os.environ.get("LOCAL_RANK","0"))
            self.multinode_available = self.world_size>1
            self.self_is_main_node = self.world_rank == 0
        
        elif self.distributed_backend == "mpi":
            self.world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE","1"))
            self.world_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK","0"))
            self.local_world_size = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE","1"))
            self.local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
            self.multinode_available = self.world_size>1
            self.self_is_main_node = self.world_rank==0

        else:
            raise NotImplementedError(f"the distributed backend {self.distributed_backend} is not implemented")
        



    


