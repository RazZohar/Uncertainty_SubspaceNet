#!/usr/bin/env python3
"""
Dataset creation script - wrapper around multi_model_dataset.py
This script handles dataset creation separately from the module definitions.
"""
import sys
import os

# Add project root to path for standalone execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.multi_model_dataset import SensorSourceGraphDataset
from src.utils import set_unified_seed
import torch

def create_dataset():
    """Create and save the dataset"""
    # Parameters
    domain = ((0, 10), (0, 10))  # (x range, y range)
    n_sensors = 2
    n_sources = 1
    d_sensor_sensor = 1.0
    d_source_source = 1.5
    d_sensor_source = 2.5

    print("Setting unified seed...")
    set_unified_seed()

    print("Creating dataset...")
    dataset = SensorSourceGraphDataset(
        D=50000,
        n_sensors=n_sensors,
        n_sources=n_sources,
        d_sensor_sensor=d_sensor_sensor,
        d_source_source=d_source_source,
        d_sensor_source=d_sensor_source,
        domain=domain,
        configuration_file='/home/alonhel/MBDL_MultiSubArrays/configuration/multi_model_data_config.json'
    )

    print("Saving dataset...")
    dataset.save_to_file('/home/alonhel/MBDL_MultiSubArrays/data/MultiSubArrays/SensorSourceGraphDataset.pkl')
    
    print("Loading dataset to verify...")
    dataset_load = torch.load('/home/alonhel/MBDL_MultiSubArrays/data/MultiSubArrays/SensorSourceGraphDataset.pkl', weights_only=False)
    
    print("Dataset created successfully!")
    print(f"Dataset length: {len(dataset_load)}")
    print(f"Number of sensors: {n_sensors}")
    print(f"Number of sources: {n_sources}")
    print(dataset_load)

if __name__ == '__main__':
    create_dataset() 