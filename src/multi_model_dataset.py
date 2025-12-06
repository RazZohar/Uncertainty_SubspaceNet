import copy
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data import Dataset
import torch
import os.path

# Use relative imports since this file is now only used as a module
from .utils import set_unified_seed
from .system_model import SystemModelParams
from .data_handler import create_samples
from .signal_creation import Samples

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

    # TODO: limit by the domain
    plt.xlim((0,100))
    plt.ylim((-0.1, 100))

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

def check_angle_separation(relative_angles,
                           min_angle_sep_rad,
                           sensor_indices=None):
    """
    relative_angles: (n_sensors, n_sources) in radians
    min_angle_sep_rad: minimal allowed separation (radians)
    sensor_indices: iterable of sensor indices to enforce on,
                    or None for all sensors
    """
    n_sensors, n_sources = relative_angles.shape

    if sensor_indices is None:
        sensor_indices = range(n_sensors)

    for i in sensor_indices:
        for j in range(n_sources):
            for k in range(j + 1, n_sources):
                a1 = relative_angles[i, j]
                a2 = relative_angles[i, k]
                # minimal angular difference in [0, pi]
                diff = np.abs(np.arctan2(np.sin(a1 - a2), np.cos(a1 - a2)))
                if diff < min_angle_sep_rad:
                    return False
    return True

def create_single_graph_data(
    d_sensor_sensor,
    d_sensor_source,
    d_source_source,
    domain,
    n_sensors,
    n_sources,
    sensor_positions=None,
    min_angle_sep_deg=None,      # NEW: minimal separation in degrees
    sep_sensor_indices=None,     # NEW: which sensors to enforce on (None = all)
    max_angle_attempts=5000      # NEW: max re-sampling attempts
):
    if sensor_positions is None:
        # Step 1: Generate sensors
        sensor_positions = generate_points_with_gap(
            n_points=n_sensors,
            d_self=d_sensor_sensor,
            domain=domain
        )

    # Step 2: Generate sources with respect to sensors,
    #         but enforce angular separation if requested
    attempts = 0
    while True:
        source_positions = generate_points_with_gap(
            n_points=n_sources,
            d_self=d_source_source,
            domain=domain,
            other_pts=sensor_positions,
            d_other=d_sensor_source
        )

        # Always compute relative angles; we might need them for the constraint
        relative_angles = compute_relative_angles(sensor_positions, source_positions)

        if min_angle_sep_deg is None:
            # No angle constraint -> accept immediately
            break

        min_angle_sep_rad = 0.2#np.deg2rad(min_angle_sep_deg)
        if check_angle_separation(relative_angles,
                                  min_angle_sep_rad,
                                  sensor_indices=sep_sensor_indices):
            # Configuration satisfies angle constraint
            break

        attempts += 1
        if attempts >= max_angle_attempts:
            raise RuntimeError(
                f"Could not sample sources satisfying angular separation "
                f">= {min_angle_sep_deg}° after {max_angle_attempts} attempts."
            )

    # At this point, source_positions + relative_angles satisfy the constraints
    model_graph = create_sensor_source_graph(
        sensor_positions,
        source_positions,
        np.degrees(relative_angles)
    )
    return model_graph, sensor_positions, source_positions, relative_angles



class SensorSourceGraphDataset(Dataset):
    def __init__(self, D, n_sensors, n_sources,
                 d_sensor_sensor, d_source_source, d_sensor_source,
                 domain, configuration_file):
        self.samples_graphs = []
        self.localization_scene = []
        self.use_graph_features = False  # Default to using original graphs
        with open(configuration_file, 'r') as f:
            subarray_configuration = json.load(f)


        self.__system_model_params = (
            SystemModelParams()
        ).set_params_from_json(subarray_configuration)

        self.__SAMPLE_SIZE_PER_SUBARRAY = 1
        self.__samples_model = Samples(self.__system_model_params)

        #TODO: we limit the source to be on X axis
        #source_domain = ((domain[0][0], domain[0][1]), (domain[1][0], domain[1][1]))
        source_domain = ((domain[0][0], domain[0][1]), (0.0, 0.0))
        self._sensor_position = generate_points_with_gap(
            n_points=n_sensors,
            d_self=d_sensor_sensor,
            domain=source_domain
            )



        # Put the sources in front of the sensors
        y_max = max(self._sensor_position, key=lambda x:x[1])[1]

        # Put the sources at RHS of the sensor in order to allow -pi/2,pi/2
        x_min = max(self._sensor_position, key=lambda x: x[0])[0]

        # Fix the domain of sources locations
        #domain = (domain[0][0], domain[0][1]), (y_max, domain[1][1])
        #domain = (x_min + 2, domain[0][1]), (5.0, domain[1][1])
        domain = (40, 100), (5.0, domain[1][1])

        for dataset_index in range(D):

            #TODO: pass config file for the sensors
            model_graph, sensor_positions, source_positions, relative_angles = create_single_graph_data(d_sensor_sensor, d_sensor_source,
                                                                                       d_source_source, domain,
                                                                                       n_sensors,
                                                                                       n_sources, sensor_positions=self._sensor_position,
                                                                                                        min_angle_sep_deg=15.0)

            print(f'{sensor_positions=}, {source_positions=}, {relative_angles=}')
            #draw_graph_with_precomputed_angles(model_graph)

            samples_graphs = []
            scene_model_dataset = []
            scene_generic_dataset = []
            relative_angles = np.degrees(relative_angles) - 90.0
            # Generate I-Q signals due to the sample model
            for index in range(n_sensors):

                #this is the dataset by (X, Theta) in this manner X is the signals as observed by the i'th sensor
                subarray_model_dataset, subarray_generic_dataset = create_samples(model_type="MultiRSSN", phase=None, samples_model=self.__samples_model, samples_size=self.__SAMPLE_SIZE_PER_SUBARRAY, tau=None,
                                                            true_doa=relative_angles[index])

                # Convert to tensor during data generation
                scene_model_dataset.insert(index, copy.deepcopy(subarray_model_dataset))
                #scene_generic_dataset.append(subarray_generic_dataset)



            # samples_graphs.insert(dataset_index, copy.deepcopy(scene_model_dataset))

            #TODO: add the source positions?
            # self.localization_scene.insert(dataset_index, (model_graph, copy.deepcopy(samples_graphs)))

            scene_model_dataset = self.collapse_samples(scene_model_dataset)

            # Pre-compute graph features for fast batching
            sensor_positions = torch.tensor([
                model_graph.nodes[node]['obj'].position
                for node in model_graph.nodes
                if model_graph.nodes[node]['type'] == 'sensor'
            ], dtype=torch.float32)

            source_positions = torch.tensor([
                model_graph.nodes[node]['obj'].position
                for node in model_graph.nodes
                if model_graph.nodes[node]['type'] == 'source'
            ], dtype=torch.float32)

            # Store both original graph and pre-computed features
            self.localization_scene.insert(dataset_index, (
                model_graph,
                copy.deepcopy(scene_model_dataset),
                sensor_positions,
                source_positions
            ))

    def __len__(self):
        return len(self.localization_scene)

    def __getitem__(self, idx):
        """
        Returns
        -------
        If use_graph_features=False (default):
            graph    : networkx.Graph
            signals  : FloatTensor (n_arrays, N, T, n_samples)
            doas     : FloatTensor (n_arrays, M)

        If use_graph_features=True:
            sensor_positions : FloatTensor (n_sensors, 2)
            source_positions : FloatTensor (n_sources, 2)
            signals  : FloatTensor (n_arrays, N, T, n_samples)
            doas     : FloatTensor (n_arrays, M)
        """
        graph, sensor_list, sensor_positions, source_positions = self.localization_scene[idx]

        # sensor_list[j] = (X_j, Y_j) with
        #   X_j: (N, T, n_samples)
        #   Y_j: (M, 1)

        # 1) stack all X_j → (n_arrays, N, T, n_samples)
        IQ_signals_stack = torch.stack([x for x, _ in sensor_list], dim=0)

        # 2) stack all Y_j, then squeeze → (n_arrays, M)
        doa_stack = torch.stack([y.squeeze(-1) for _, y in sensor_list], dim=0)

        # Return based on the flag
        if self.use_graph_features:
            return sensor_positions, source_positions, IQ_signals_stack, doa_stack
        else:
            return graph, IQ_signals_stack, doa_stack

    def save_to_file(self, filename):
        path = os.path.dirname(filename)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created path: {path}")
        torch.save(self, filename)

    def get_sensor_potision(self):
        return self._sensor_position

    def set_use_graph_features(self, use_features=True):
        """
        Set whether to return pre-computed graph features or original graphs

        Args:
            use_features (bool): If True, return tensor features for batching.
                               If False, return original NetworkX graphs.
        """
        self.use_graph_features = use_features

    def collapse_samples(self, samples):
        """
        Convert the current structure

            samples[sensor][sample] = (X, Y)

        into

            samples[sensor] = (X_stack, Y0)

        where
            X_stack : (N, T, n_samples)
            Y0      : (M, 1)

        Args
        ----
        samples : list  # len = n_sensors
            Each item is a list of length n_samples with (X, Y) tuples.

        Returns
        -------
        new_samples : list            # len = n_sensors
            Each item is a tuple (X_stack, Y0) as described above.
        """
        new_samples = []

        for sensor_idx, sensor_list in enumerate(samples):
            # 1) Split the tuples
            X_list, Y_list = zip(*sensor_list)      # tuples of tensors

            # 2) Stack X along a **new third axis**
            #    Each X_list[k] is (N, T) → unsqueeze(-1) → (N, T, 1)
            X_pc = torch.cat(
                [x.unsqueeze(-1) for x in X_list],  # cat on last dim
                dim=-1                              # (N, T, n_samples)
            )

            X_stack = X_pc.permute(2, 0, 1).contiguous()

            # 3) Ensure all Y are identical, keep the first
            Y0 = Y_list[0]
            if not all(torch.equal(y, Y0) for y in Y_list[1:]):
                raise ValueError(
                    f"Sensor {sensor_idx}: Y labels differ across snapshots."
                )

            new_samples.append((X_stack, Y0))

        return new_samples

if __name__ == "__main__":
    pass