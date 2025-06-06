import json

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data import Dataset
import torch
import os.path

from src.utils import set_unified_seed
from system_model import SystemModelParams
from data_handler import create_samples
from signal_creation import Samples

def in_min_distance(pt, pts, d):
    return all(np.linalg.norm(pt - p) >= d for p in pts)

def generate_points_with_gap(n_points, d_self, domain, other_pts=None, d_other=None, max_attempts=10000):
    """Generate points with minimum distance between them and optionally from another set."""
    points = []
    attempts = 0
    while len(points) < n_points and attempts < max_attempts:
        x = np.random.uniform(*domain[0])
        y = np.random.uniform(*domain[1])
        pt = np.array([x, y])

        if in_min_distance(pt, points, d_self):
            if other_pts is None or in_min_distance(pt, other_pts, d_other):
                points.append(pt)

        attempts += 1

    return np.array(points)


def compute_relative_angles(sensors, sources):
    """
    Compute angles from each sensor to each source using atan2.
    Returns a (n_sensors, n_sources) array of angles in radians.
    """
    n_sensors = sensors.shape[0]
    n_sources = sources.shape[0]
    angles = np.zeros((n_sensors, n_sources))

    for i in range(n_sensors):
        for j in range(n_sources):
            dx = sources[j, 0] - sensors[i, 0]
            dy = sources[j, 1] - sensors[i, 1]
            angles[i, j] = np.arctan2(dy, dx)

    return angles


class Sensor:
    def __init__(self, position):
        self.position = tuple(position)


class Source:
    def __init__(self, position, source_strength):
        self.position = tuple(position)
        self.source_strength = source_strength  # custom property



def create_sensor_source_graph(sensors, sources, angles):
    G = nx.Graph()

    #TODO: Load json file and read configuration

    # Add sensor nodes
    for i, pos in enumerate(sensors):
        sensor = Sensor(pos)
        G.add_node(f'sensor_{i}', obj=sensor, type='sensor')

    # Add source nodes
    for j, pos in enumerate(sources):
        source_strength = np.random.uniform(0.5, 2.0)
        source = Source(pos, source_strength)
        G.add_node(f'source_{j}', obj=source, type='source')

    # Add edges with angles
    for i in range(len(sensors)):
        for j in range(len(sources)):
            G.add_edge(f'sensor_{i}', f'source_{j}', angle=angles[i, j])

    return G


def draw_graph_with_precomputed_angles(G):
    pos = {n: data['obj'].position for n, data in G.nodes(data=True)}

    node_types = nx.get_node_attributes(G, 'type')

    sensor_nodes = [n for n in G.nodes if node_types[n] == 'sensor']
    source_nodes = [n for n in G.nodes if node_types[n] == 'source']

    edge_labels = {
        (u, v): f"{d['angle']:.1f}°"
        for u, v, d in G.edges(data=True)
        if 'angle' in d
    }

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, nodelist=sensor_nodes, node_color='blue', node_size=100, label='Sensors')
    nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, node_color='red', node_size=100, label='Sources')
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green', font_size=8)
    plt.title("Sensor-Source Graph with Precomputed Angle Labels")
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.show()


def visualize_localization_scene(sensor_positions, source_positions):
    plt.figure(figsize=(8, 8))
    for sensor in sensor_positions:
        for source in source_positions:
            direction = source - sensor
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue  # skip overlapping
            unit_dir = direction / norm
            arrow_dx, arrow_dy = unit_dir * 0.8
            plt.arrow(sensor[0], sensor[1],
                      arrow_dx, arrow_dy,
                      head_width=0.15, head_length=0.2,
                      fc='gray', ec='gray', alpha=0.4)
    plt.scatter(sensor_positions[:, 0], sensor_positions[:, 1], c='blue', label='Sensors')
    plt.scatter(source_positions[:, 0], source_positions[:, 1], c='red', label='Sources')
    plt.legend()
    plt.title("Sensor-Source Network Placement")
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.show()


def create_single_graph_data(d_sensor_sensor, d_sensor_source, d_source_source, domain, n_sensors, n_sources):
    # Step 1: Generate sensors
    sensor_positions = generate_points_with_gap(
        n_points=n_sensors,
        d_self=d_sensor_sensor,
        domain=domain
    )
    # Step 2: Generate sources with respect to sensors
    source_positions = generate_points_with_gap(
        n_points=n_sources,
        d_self=d_source_source,
        domain=domain,
        other_pts=sensor_positions,
        d_other=d_sensor_source
    )
    # Compute relative angles
    relative_angles = compute_relative_angles(sensor_positions, source_positions)

    model_graph = create_sensor_source_graph(sensor_positions, source_positions, np.degrees(relative_angles))
    return model_graph, sensor_positions, source_positions, relative_angles


class SensorSourceGraphDataset(Dataset):
    def __init__(self, D, n_sensors, n_sources,
                 d_sensor_sensor, d_source_source, d_sensor_source,
                 domain, configuration_file):
        self.samples_graphs = []
        self.localization_scene = []
        with open(configuration_file, 'r') as f:
            subarray_configuration = json.load(f)


        self.__system_model_params = (
            SystemModelParams()
        ).set_params_from_json(subarray_configuration)

        self.__SAMPLE_SIZE_PER_SUBARRAY = 20
        self.__samples_model = Samples(self.__system_model_params)


        for index in range(D):
            #TODO: pass config file for the sensors
            model_graph, sensor_positions, source_positions, relative_angles = create_single_graph_data(d_sensor_sensor, d_sensor_source,
                                                                                       d_source_source, domain,
                                                                                       n_sensors,
                                                                                       n_sources)


            draw_graph_with_precomputed_angles(model_graph)

            samples_graphs = []
            scene_model_dataset = []
            scene_generic_dataset = []
            # Generate I-Q signals due to the sample model
            for index in range(n_sensors):

                #this is the dataset by (X, Theta) in this manner X is the signals as observed by the i'th sensor
                subarray_model_dataset, subarray_generic_dataset = create_samples(model_type="MultiRSSN", phase=None, samples_model=self.__samples_model, samples_size=self.__SAMPLE_SIZE_PER_SUBARRAY, tau=None,
                                                            true_doa=relative_angles[index])

                scene_model_dataset.append(subarray_model_dataset)
                scene_generic_dataset.append(subarray_generic_dataset)



                samples_graphs.append((scene_model_dataset, scene_generic_dataset))

            self.localization_scene.append((model_graph, samples_graphs))

    def __len__(self):
        return len(self.localization_scene)

    def __getitem__(self, idx):
        return self.localization_scene[idx]

    def save_to_file(self, filename):
        path = os.path.dirname(filename)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created path: {path}")
        torch.save(self, filename)



def test_data_creation():
    # Parameters
    domain = ((0, 10), (0, 10))  # (x range, y range)
    n_sensors = 4
    n_sources = 3
    d_sensor_sensor = 1.0
    d_source_source = 1.5
    d_sensor_source = 1.2

    set_unified_seed()
    #model_graph, sensor_positions, source_positions, relative_angles = create_single_graph_data(d_sensor_sensor, d_sensor_source, d_source_source, domain, n_sensors, n_sources)

    #visualize_localization_scene(sensor_positions, source_positions)
    #draw_graph_with_precomputed_angles(model_graph)

    #print(relative_angles)

    dataset = SensorSourceGraphDataset(
        D=5,
        n_sensors=n_sensors,
        n_sources=n_sources,
        d_sensor_sensor=d_sensor_sensor,
        d_source_source=d_source_source,
        d_sensor_source=d_sensor_source,
        domain=domain,
        configuration_file='../configuration/SignalsSubspaceNet_M=3_T=100_SNR_10_tau=None_NarrowBand_diff_method=None_non-coherent_eta=0_bias=0.0_sv_noise=0.json')

    dataset.save_to_file('../data/MultiSubArrays/SensorSourceGraphDataset.pkl')

    dataset_load = torch.load('../data/MultiSubArrays/SensorSourceGraphDataset.pkl')

    print(dataset_load)

if __name__ == '__main__':
    test_data_creation()