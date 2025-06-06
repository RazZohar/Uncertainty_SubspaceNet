import torch.nn as nn
import torch

from system_model import SystemModelParams

from models import SignalsSubspaceNetEsprit
import json

class MultiSubarraysModel(nn.Module):
    def __init__(self, sensors_positions, multi_model_configuration):
        super(MultiSubarraysModel, self).__init__()


        self.sensors_graph = sensors_positions

        subarrays_config, self.number_of_sensors = self._load_multi_model_configuration(multi_model_configuration)

        self.subarray_models = nn.ModuleList()

        self.create_model(subarrays_config)

        #self.attentaion_list = nn.ModuleList()
        #self.create_attentaion_by_position(sensors_positions)



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


if __name__ == '__main__':
    multi_arrays_model = MultiSubarraysModel(sensors_positions=[(0, 1), (1.5, 0), (2.5, 6)], multi_model_configuration="../configuration/multi_model_configuration.json")
    print(multi_arrays_model)