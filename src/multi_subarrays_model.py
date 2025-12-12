import torch.nn as nn
import torch
import copy
import json
import numpy as np

from .system_model import SystemModelParams
from .models import SignalsSubspaceNetEsprit
from .multi_model_dataset import SensorSourceGraphDataset, Sensor, Source
from .localization_block import RayIntersection, triangulation_with_soft_area_batched, position_errors
from .learned_agg_layer import LearnedAgg
from .uncertainty_block import UncertaintyEstimation

def get_location_from_model_graph(model_graph, type_req='sensor'):
    """
    Return the position of the sensors
    :param model_graph:
    :return:
    """
    return [model_graph.nodes[node]['obj'].position for node in model_graph.nodes if type_req in node]


class MultiSubarraysModel(nn.Module):
    def __init__(self, sensors_positions, multi_model_configuration,args):
        super(MultiSubarraysModel, self).__init__()


        self.sensors_graph = sensors_positions
        self.args = args
        subarrays_config, self.number_of_sensors = self._load_multi_model_configuration(multi_model_configuration)

        self.subarray_models = nn.ModuleList()
        self.learned_attentaion = nn.ModuleList()

        # Init the uncertainty predication block
        self.uncertainty_pred = nn.ModuleList()

        self.create_model(subarrays_config)

        #self.attentaion_list = nn.ModuleList()
        #self.create_attentaion_by_position(sensors_positions)

        # Doa Assosication block
        self.rays_intersection = RayIntersection()

        # Change Those flags by train/inference iterations
        self.estimate_uncertainty = False
        self.fuse_sensors = False
        self.estimate_position = False





    def _load_multi_model_configuration(self, multi_model_configuration_filename):
        multi_model_configuration = json.load(open(multi_model_configuration_filename))
        return multi_model_configuration["subarray_config"], multi_model_configuration["L"]

    def create_model(self, subarray_configuration):
        for subarray_index in range(self.number_of_sensors):
            subarray_model = self._create_subarray_model_by_configuration(subarray_configuration[subarray_index])
            # subarray_model.set_batch_size(self.args.batch_size)
            self.subarray_models.insert(subarray_index, subarray_model)

            # Add the learned attentaion layer
            self.learned_attentaion.insert(subarray_index, LearnedAgg(self.number_of_sensors))

            self.uncertainty_pred.insert(subarray_index, UncertaintyEstimation(self.number_of_sensors))


    def _create_subarray_model_by_configuration(self, subarray_configuration):
        system_model_params = SystemModelParams()
        system_model_params.set_params_from_json(subarray_configuration)
        return SignalsSubspaceNetEsprit(N=system_model_params.N,
                                        T=system_model_params.T,
                                        tau=8,
                                        M=system_model_params.M,
                                        codebook_size=system_model_params.codebook_size,
                                        quantize_source=False)


    def forward(self, sensor_location, IQ_signals_stack, gt_pos):

        # sensor_location = get_location_from_model_graph(model_graph)

        bearings = []
        q_i = []
        for subarray_index in range(self.number_of_sensors):
            # Original
            # iq_signals, doa = samples[0][subarray_index][0][0], samples[0][subarray_index][0][1]

            # With new dataset
            # iq_signals is now a tensor with dim of (B, n_arrays, __SAMPLE_SIZE_PER_SUBARRAY,N,T)
            # if you want only one sample for sub array (as in the code before) just use iq_signals[0]
            iq_signals = IQ_signals_stack[:,subarray_index,0,:,:] # (B, N, T)

            # FIXME: gt_pos shouldn't be here, this is the ground truth position of the source,
            # should be in the training loop only
            # doa is now a tensor of dimension [B, M - Number of sources]. Each entry is the direction
            # of source i from sensor array [subarray_index]
            #doa = gt_pos[:,subarray_index]

            #Suggestion:
            # rand_idx = torch.randint(0, len(samples[subarray_index]), (1,)).item()
            # iq_signal, doa = samples[subarray_index][rand_idx]

            vq_loss , q_quantized = self.subarray_models[subarray_index].sense_device_forward(iq_signals)
            q_i.append(q_quantized)

        q_i_stack = torch.stack(q_i, dim=1)
        # TODO: Later add option to work in stages with arguments
        if self.fuse_sensors is True:
            with torch.no_grad():
                # TODO: add attention between subarrays
                z_i = []
                phi_i = []
                for subarray_index in range(self.number_of_sensors):
                    z, phi = self.learned_attentaion[subarray_index].forward(q_i_stack, sensor_location.squeeze(0))
                    z_i.insert(subarray_index, z)
                    phi_i.insert(subarray_index, phi)

            z_i_stack = torch.stack(z_i, dim=1)
        sigma_i = []
        for subarray_index in range(self.number_of_sensors):
            R, doa_pred = self.subarray_models[subarray_index].inference_device_forward(q_i_stack[:,subarray_index,:,:])

            bearings.append(doa_pred)
            if self.estimate_uncertainty is True:
                with torch.no_grad():
                    sigma = self.uncertainty_pred[subarray_index].forward(doa_pred.rad2deg(), R)
                    sigma_i.insert(subarray_index, sigma)


        bearings = torch.stack(bearings, dim=1)
        if self.estimate_uncertainty is True:
            sigma_i_stack = torch.stack(sigma_i, dim=1)


        #TODO: Assosicate angles
        with torch.no_grad():
            #TODO: Foward both WLS and LS to compare
            if self.estimate_position is True:
                source_estimated_position, dop = self.rays_intersection.forward(sensor_location, bearings.squeeze(-1))
                if self.estimate_uncertainty is True:
                    source_estimated_position_wls, dop_wls = self.rays_intersection.forward(sensor_location, bearings.squeeze(-1), torch.deg2rad(sigma_i_stack.sqrt()))
                    position_metrics = position_errors(source_estimated_position, source_estimated_position_wls, gt_pos)
                #centroid, area_soft, Sigma_s = triangulation_with_soft_area_batched(sensor_location, bearings, torch.deg2rad(sigma_i_stack.sqrt()))

        if self.args.train_doa_only:
            requested_values = {}
            requested_values["bearings"] = bearings


            if self.fuse_sensors is True:
                requested_values["phi_i"] = phi_i

            if self.estimate_uncertainty is True:
                requested_values["sigma_i"] = torch.deg2rad(sigma_i_stack.sqrt())



            if self.estimate_uncertainty is True and self.estimate_position is True:
                requested_values["position_metrics"] = position_metrics
                requested_values["source_estimated_position"] = source_estimated_position
                requested_values["dop"] = dop
                requested_values["source_estimated_position_wls"] = source_estimated_position_wls
                requested_values["dop_wls"] = dop_wls

            """
            requested_values["area"] = area_soft
            requested_values["centroid"] = centroid
            requested_values["Sigma_s"] = Sigma_s
            """
            return requested_values

        #TODO: Assosicate angles

        # Intersect rays
        #return bearings, source_estimated_position, dop
        #return source_estimated_position

    def enable_fuse_sensors(self):
        self.fuse_sensors = True

    def enable_uncetainty_estimation(self):
        self.estimate_uncertainty = True


if __name__ == '__main__':
    dataset_load = torch.load('../data/MultiSubArrays/SensorSourceGraphDataset.pkl')
    multi_arrays_model = MultiSubarraysModel(sensors_positions=[(0, 1), (1.5, 0)], multi_model_configuration="../configuration/multi_model_configuration.json")

    for scene in dataset_load:
        x_hat = multi_arrays_model(scene)
        source_true_location = torch.Tensor(get_location_from_model_graph(scene[0], type_req='source')).float()
        print(f'{x_hat=}, {source_true_location=}, {torch.norm(x_hat-source_true_location)=}')
    #print(multi_arrays_model)
