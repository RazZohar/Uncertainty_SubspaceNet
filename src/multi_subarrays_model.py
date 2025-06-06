import torch.nn as nn
import torch

from system_model import SystemModelParams

from models import SignalsSubspaceNetEsprit
import json

class MultiSubarraysModel(nn.Module):
    def __init__(self, number_of_sensors, sensors_graph, multi_model_configuration):
        super(MultiSubarraysModel, self).__init__()

        # Termed L at our convention
        self.number_of_sensors = number_of_sensors

        self.sensors_graph = sensors_graph

        subarrays_config = self._load_multi_model_configuration(multi_model_configuration)

        self.subarray_models = nn.ModuleList()

        self.create_subarray_models(subarrays_config)



    def _load_multi_model_configuration(self, multi_model_configuration_filename):
        multi_model_configuration = json.load(open(multi_model_configuration_filename))
        return multi_model_configuration["subarray_config"]

    def create_model(self, subarray_configuration):
        for subarray_index in range(self.number_of_sensors):
            subarray_model = self._create_subarray_model_by_configuration(subarray_configuration[subarray_index])
            self.subarray_models[subarray_index] = subarray_model

    def _create_subarray_model_by_configuration(self, subarray_configuration):
        system_model_params = SystemModelParams()
        system_model_params.set_params_from_json(subarray_configuration)
        return SignalsSubspaceNetEsprit(N=system_model_params.N, T=system_model_params.T, tau=self.tau, M=system_model_params.M, codebook_size=system_model_params.codebook_size)
