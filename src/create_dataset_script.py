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
    domain = ((0, 100), (0, 100))  # (x range, y range)
    n_sensors = 3
    n_sources = 2
    d_sensor_sensor = 1.0
    d_source_source = 6.0
    d_sensor_source = 3.5

    print("Setting unified seed...")
    set_unified_seed()

    DATASET_TRAIN_SIZE = 50000
    #DATASET_TRAIN_SIZE = 50
    DATASET_TEST_SIZE = int(DATASET_TRAIN_SIZE * 0.1)


    print("Creating dataset...")
    dataset = SensorSourceGraphDataset(
        D=DATASET_TRAIN_SIZE,
        n_sensors=n_sensors,
        n_sources=n_sources,
        d_sensor_sensor=d_sensor_sensor,
        d_source_source=d_source_source,
        d_sensor_source=d_sensor_source,
        domain=domain,
        configuration_file='/Users/razzohar/PycharmProjects/MBDL_MultiSubArrays/configuration/multi_model_configuration.json'
    )

    print("Saving dataset...")
    dataset.save_to_file('/Users/razzohar/PycharmProjects/MBDL_MultiSubArrays/data/MultiSubArrays/SensorSourceGraphDataset.pkl')

    print("Loading dataset to verify..   .")
    dataset_load = torch.load('/Users/razzohar/PycharmProjects/MBDL_MultiSubArrays/data/MultiSubArrays/SensorSourceGraphDataset.pkl', weights_only=False)

    print("Creating test dataset...")
    dataset = SensorSourceGraphDataset(
        D=DATASET_TEST_SIZE,
        n_sensors=n_sensors,
        n_sources=n_sources,
        d_sensor_sensor=d_sensor_sensor,
        d_source_source=d_source_source,
        d_sensor_source=d_sensor_source,
        domain=domain,
        configuration_file='/Users/razzohar/PycharmProjects/MBDL_MultiSubArrays/configuration/multi_model_configuration.json',
        sensor_positions=dataset_load._sensor_position
    )

    print("Saving dataset...")
    dataset.save_to_file(
        '/Users/razzohar/PycharmProjects/MBDL_MultiSubArrays/data/MultiSubArrays/SensorSourceGraphDataset_test.pkl')

    print("Dataset created successfully!")
    #print(f"Dataset length: {len(dataset_load)}")
    print(f"Number of sensors: {n_sensors}")
    print(f"Number of sources: {n_sources}")
    #print(dataset_load)
    #print(dataset_load.__getitem__(0))
    """
    dataset_load.use_graph_features = True
    for idx, item in enumerate(dataset_load):
        if item[1][0][0] > 8:
            print(item[1])
            print(idx)
            break
#        if idx > 200:
#            break
    """
if __name__ == '__main__':
    create_dataset() 