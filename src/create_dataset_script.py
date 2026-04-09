#!/usr/bin/env python3
"""
Dataset creation script - wrapper around multi_model_dataset.py
This script handles dataset creation separately from the module definitions.
"""
import sys
import os
import argparse
import torch
import json

# Add project root to path for standalone execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.multi_model_dataset import SensorSourceGraphDataset
from src.utils import set_unified_seed


def create_dataset(config_path, output_dir):
    """Create and save the dataset dynamically based on sweep parameters"""

    # --- NEW: Read parameters dynamically from JSON ---
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Extract L (Number of subarrays/sensors)
    n_sensors = config_data.get("L", 3)

    # Extract M (Number of sources/targets) from the first subarray config
    try:
        n_sources = config_data["subarray_config"][0]["system_model"]["M"]
    except (KeyError, IndexError):
        n_sources = 2  # Fallback just in case

    # Static Parameters
    domain = ((0, 100), (0, 100))  # (x range, y range)
    d_sensor_sensor = 1.0
    d_source_source = 6.0
    d_sensor_source = 3.5

    print("Setting unified seed...")
    set_unified_seed()

    DATASET_TRAIN_SIZE = 50000
    DATASET_TEST_SIZE = int(DATASET_TRAIN_SIZE * 0.1)

    print(f"\nCreating main dataset (Size: {DATASET_TRAIN_SIZE})...")
    print(f" -> Sensors (L): {n_sensors} | Sources (M): {n_sources}")

    dataset = SensorSourceGraphDataset(
        D=DATASET_TRAIN_SIZE,
        n_sensors=n_sensors,
        n_sources=n_sources,
        d_sensor_sensor=d_sensor_sensor,
        d_source_source=d_source_source,
        d_sensor_source=d_sensor_source,
        domain=domain,
        configuration_file=config_path
    )

    # Save Main Dataset
    main_dataset_path = os.path.join(output_dir, 'dataset.pt')
    print(f"Saving main dataset to: {main_dataset_path}")
    dataset.save_to_file(main_dataset_path)

    print(f"\nCreating test dataset (Size: {DATASET_TEST_SIZE})...")
    test_dataset = SensorSourceGraphDataset(
        D=DATASET_TEST_SIZE,
        n_sensors=n_sensors,
        n_sources=n_sources,
        d_sensor_sensor=d_sensor_sensor,
        d_source_source=d_source_source,
        d_sensor_source=d_sensor_source,
        domain=domain,
        configuration_file=config_path,
        sensor_positions=dataset._sensor_position  # Ensure test sensors exactly match train sensors
    )

    # Save Test Dataset
    test_dataset_path = os.path.join(output_dir, 'test_dataset.pt')
    print(f"Saving test dataset to: {test_dataset_path}")
    test_dataset.save_to_file(test_dataset_path)

    print("\n✅ Datasets created successfully!")


def main():
    parser = argparse.ArgumentParser(description="Create datasets for DoA estimation sweeps")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated datasets")
    args = parser.parse_args()

    # Ensure the output directory exists before generating anything
    os.makedirs(args.output_dir, exist_ok=True)

    # Run the generator
    create_dataset(args.config, args.output_dir)


if __name__ == "__main__":
    main()