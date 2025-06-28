import torch.nn as nn
import torch
import copy
import json
import numpy as np

from system_model import SystemModelParams
from models import SignalsSubspaceNetEsprit
from multi_model_dataset import SensorSourceGraphDataset, Sensor, Source
from learned_agg_layer import LearnedAgg

from localization_block import RayIntersection

POSITION_2D = 2


def get_location_from_model_graph(model_graph, type_req='sensor'):
    """
    Return the position of the sensors
    :param model_graph:
    :return:
    """
    return [model_graph.nodes[node]['obj'].position for node in model_graph.nodes if type_req in node]


class MultiSubarraysModel(nn.Module):
    def __init__(self, sensors_positions, multi_model_configuration):
        super(MultiSubarraysModel, self).__init__()


        self.sensors_graph = sensors_positions

        subarrays_config, self.number_of_sensors = self._load_multi_model_configuration(multi_model_configuration)

        self.subarray_models = nn.ModuleList()
        self.learned_attentaion = nn.ModuleList()

        self.create_model(subarrays_config)

        #self.attentaion_list = nn.ModuleList()
        #self.create_attentaion_by_position(sensors_positions)

        # Doa Assosication block
        self.rays_intersection = RayIntersection()





    def _load_multi_model_configuration(self, multi_model_configuration_filename):
        multi_model_configuration = json.load(open(multi_model_configuration_filename))
        return multi_model_configuration["subarray_config"], multi_model_configuration["L"]

    def create_model(self, subarray_configuration):
        for subarray_index in range(self.number_of_sensors):
            subarray_model = self._create_subarray_model_by_configuration(subarray_configuration[subarray_index])
            self.subarray_models.insert(subarray_index, subarray_model)

            # Add the learned attentaion layer
            self.learned_attentaion.insert(subarray_index, LearnedAgg(self.number_of_sensors, POSITION_2D))


    def _create_subarray_model_by_configuration(self, subarray_configuration):
        system_model_params = SystemModelParams()
        system_model_params.set_params_from_json(subarray_configuration)
        return SignalsSubspaceNetEsprit(N=system_model_params.N, T=system_model_params.T, tau=8, M=system_model_params.M, codebook_size=system_model_params.codebook_size)


    def forward(self, localization_scene):
        model_graph, samples = localization_scene
        sensor_location = get_location_from_model_graph(model_graph)

        q_i = []
        for subarray_index in range(self.number_of_sensors):
            iq_signals, doa = samples[0][subarray_index][0][0], samples[0][subarray_index][0][1]
            vq_loss, q = self.subarray_models[subarray_index].sense_device_forward(iq_signals.unsqueeze(0))
            q_i.insert(subarray_index, q)

        stacked_words = torch.stack([i.squeeze(0) for i in q_i], dim=0)

        #TODO: "Attentaion" for the fusion between subarrrays
        #z_i = []
        #for subarray_index in range(self.number_of_sensors):
        #    z, phi = self.learned_attentaion[subarray_index].forward(stacked_words.unsqueeze(0), torch.Tensor(sensor_location).unsqueeze(0))
        #    z_i.insert(subarray_index, z)

        bearings = []
        for subarray_index in range(self.number_of_sensors):
            R, estimated_angles = self.subarray_models[subarray_index].inference_device_forward(q_i[subarray_index])
            bearings.insert(subarray_index, estimated_angles)

        #TODO: Assosicate angles

        # Intersect rays

        source_estimated_position, dop = self.rays_intersection.forward(np.array(sensor_location), torch.Tensor(bearings))
        return source_estimated_position


if __name__ == '__main__':
    dataset_load = torch.load('../data/MultiSubArrays/SensorSourceGraphDataset.pkl')
    multi_arrays_model = MultiSubarraysModel(sensors_positions=[(0, 1), (1.5, 0)], multi_model_configuration="../configuration/multi_model_configuration.json")

    for scene in dataset_load:
        x_hat = multi_arrays_model(scene)
        source_true_location = torch.Tensor(get_location_from_model_graph(scene[0], type_req='source')).float()
        print(f'{x_hat=}, {source_true_location=}, {torch.norm(x_hat-source_true_location)=}')
    #print(multi_arrays_model)
