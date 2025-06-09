import torch.nn as nn
import torch
import copy
import json

from system_model import SystemModelParams

from models import SignalsSubspaceNetEsprit
from multi_model_dataset import SensorSourceGraphDataset, Sensor, Source


from localization_block import RayIntersection


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


    def _create_subarray_model_by_configuration(self, subarray_configuration):
        system_model_params = SystemModelParams()
        system_model_params.set_params_from_json(subarray_configuration)
        return SignalsSubspaceNetEsprit(N=system_model_params.N, T=system_model_params.T, tau=8, M=system_model_params.M, codebook_size=system_model_params.codebook_size)


    def forward(self, model_graph, IQ_signals_stack, doa_stack):

        sensor_location = get_location_from_model_graph(model_graph)

        bearings = []
        for subarray_index in range(self.number_of_sensors):
            # Original
            # iq_signals, doa = samples[0][subarray_index][0][0], samples[0][subarray_index][0][1]
            
            # With new dataset
            # iq_signals is now a tensor with dim of (__SAMPLE_SIZE_PER_SUBARRAY,N,T)
            # if you want only one sample for sub array (as in the code before) just use iq_signals[0]
            iq_signals = IQ_signals_stack[subarray_index]
            
            # doa is now a tensor of dimension [M - Number of sources]. Each entry is the direction
            # of source i from sensor array [subarray_index]
            doa = doa_stack[subarray_index]
            
            #Suggestion:
            # rand_idx = torch.randint(0, len(samples[subarray_index]), (1,)).item()
            # iq_signal, doa = samples[subarray_index][rand_idx]
            
            estimated_angles = self.subarray_models[subarray_index].forward(iq_signals)
            bearings.insert(subarray_index, copy.deepcopy(estimated_angles))

        #TODO: "Attentaion" for the fusion between subarrrays

        #TODO: Assosicate angles

        # Intersect rays
        source_estimated_position, dop = self.rays_intersection.forward(sensor_location, bearings)
        return source_estimated_position


if __name__ == '__main__':
    dataset_load = torch.load('/home/alonhel/MBDL_MultiSubArrays/data/MultiSubArrays/SensorSourceGraphDataset.pkl', weights_only=False)
    multi_arrays_model = MultiSubarraysModel(sensors_positions=[(0, 1), (1.5, 0), (2.5, 6)], multi_model_configuration="/home/alonhel/MBDL_MultiSubArrays/configuration/multi_model_configuration.json")

    model_graph, IQ_signals_stack, doa_stack = dataset_load.__getitem__(0)
    multi_arrays_model(model_graph, IQ_signals_stack, doa_stack)
    print(multi_arrays_model)