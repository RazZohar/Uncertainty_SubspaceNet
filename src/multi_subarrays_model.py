import torch.nn as nn
import torch
import copy
import json
import numpy as np

from .system_model import SystemModelParams
from .models import SignalsSubspaceNetEsprit
from .multi_model_dataset import SensorSourceGraphDataset, Sensor, Source
from .uncertainty_block import UncertaintyEstimation, plot_sigma_vs_doa


def get_location_from_model_graph(model_graph, type_req='sensor'):
    """
    Return the position of the sensors
    :param model_graph:
    :return:
    """
    return [model_graph.nodes[node]['obj'].position for node in model_graph.nodes if type_req in node]


class MultiSubarraysModel(nn.Module):
    def __init__(self, sensors_positions, multi_model_configuration, args):
        super(MultiSubarraysModel, self).__init__()

        self.sensors_graph = sensors_positions
        self.args = args
        subarrays_config, self.number_of_sensors = self._load_multi_model_configuration(multi_model_configuration)

        self.subarray_models = nn.ModuleList()

        # Init the uncertainty prediction block
        self.uncertainty_pred = nn.ModuleList()

        self.create_model(subarrays_config)

        # Change those flags by train/inference iterations
        self.estimate_uncertainty = False
        # Full covariance is always computed whenever uncertainty is enabled.
        # The training loss still uses only sigma_i / diagonal variance.
        self.estimate_full_covariance = True

    def _load_multi_model_configuration(self, multi_model_configuration_filename):
        multi_model_configuration = json.load(open(multi_model_configuration_filename))
        return multi_model_configuration["subarray_config"], multi_model_configuration["L"]

    def create_model(self, subarray_configuration):
        for subarray_index in range(self.number_of_sensors):
            subarray_model = self._create_subarray_model_by_configuration(subarray_configuration[subarray_index])
            self.subarray_models.insert(subarray_index, subarray_model)


            self.uncertainty_pred.insert(
                subarray_index,
                UncertaintyEstimation(subarray_configuration[subarray_index]['system_model']['T'])
            )

    def _create_subarray_model_by_configuration(self, subarray_configuration):
        system_model_params = SystemModelParams()
        system_model_params.set_params_from_json(subarray_configuration)
        return SignalsSubspaceNetEsprit(
            N=system_model_params.N,
            T=system_model_params.T,
            tau=8,
            M=system_model_params.M,
            codebook_size=system_model_params.codebook_size,
            quantize_source=False)


    def forward(self, sensor_location, IQ_signals_stack, gt_pos):
        bearings = []
        q_i = []

        for subarray_index in range(self.number_of_sensors):
            # IQ_signals_stack: [B, n_arrays, samples_per_subarray, N, T]
            iq_signals = IQ_signals_stack[:, subarray_index, 0, :, :]  # [B, N, T]
            vq_loss, q_quantized = self.subarray_models[subarray_index].sense_device_forward(iq_signals)
            q_i.append(q_quantized)

        q_i_stack = torch.stack(q_i, dim=1)

        sigma_i = []
        cov_i = []

        for subarray_index in range(self.number_of_sensors):
            if self.fuse_sensors is False:
                R, doa_pred = self.subarray_models[subarray_index].inference_device_forward(
                    q_i_stack[:, subarray_index, :, :]
                )

            bearings.append(doa_pred)

            if self.estimate_uncertainty is True:
                # Torch analytic uncertainty block. We keep it under no_grad because
                # it is an analytic evaluator/calibrator. Full covariance is always
                # computed for ANEES/APEC/EEC, while the training loss still uses
                # only sigma_i / diagonal variance.
                with torch.no_grad():
                    sigma, cov = self.uncertainty_pred[subarray_index].forward(
                        doa_pred.rad2deg(),
                        R,
                        return_covariance=True,
                    )
                    sigma_i.insert(subarray_index, sigma)       # [B,P], deg
                    cov_i.insert(subarray_index, cov)           # [B,P,P], deg^2

        bearings = torch.stack(bearings, dim=1)  # [B, S, P]

        # Sort bearings by target dimension.
        bearings, bearings_order_index = torch.sort(bearings, dim=2)

        if self.estimate_uncertainty is True:
            sigma_i_stack = torch.stack(sigma_i, dim=1)  # [B, S, P]
            sigma_i_stack = torch.gather(
                sigma_i_stack,
                dim=2,
                index=bearings_order_index,
            )

            cov_i_stack = torch.stack(cov_i, dim=1)  # [B, S, P, P], deg^2
            P = cov_i_stack.shape[-1]

            # Reorder covariance rows according to the same target ordering.
            row_idx = bearings_order_index.unsqueeze(-1).expand(-1, -1, -1, P)
            cov_i_stack = torch.gather(cov_i_stack, dim=2, index=row_idx)

            # Reorder covariance columns according to the same target ordering.
            col_idx = bearings_order_index.unsqueeze(-2).expand(-1, -1, P, -1)
            cov_i_stack = torch.gather(cov_i_stack, dim=3, index=col_idx)

            # Common metric key in radians^2 for ANEES/APEC/EEC.
            covariance_i_stack = cov_i_stack * ((torch.pi / 180.0) ** 2)

        with torch.no_grad():
                if self.estimate_uncertainty is True:
                    source_estimated_position_wls, dop_wls = self.rays_intersection.forward(
                        sensor_location,
                        bearings.squeeze(-1),
                        torch.deg2rad(sigma_i_stack),
                    )

        if self.args.train_doa_only:
            requested_values = {"bearings": bearings}

            if self.estimate_uncertainty is True:
                requested_values["sigma_i"] = sigma_i_stack  # degrees; loss uses this only
                requested_values["covariance_i"] = covariance_i_stack  # radians^2; metrics use this
                requested_values["cov_i"] = cov_i_stack  # degrees^2, full covariance for debugging/backward compatibility


            return requested_values

        # TODO: return localization output for non-DOA training path if needed.

    def enable_uncetainty_estimation(self):
        # Keep the original misspelled method name for backward compatibility.
        self.estimate_uncertainty = True

    def enable_uncertainty_estimation(self):
        self.enable_uncetainty_estimation()

    def enable_full_covariance_estimation(self):
        # Kept for backward compatibility. Full covariance is always enabled.
        self.estimate_uncertainty = True
        self.estimate_full_covariance = True
        for block in self.uncertainty_pred:
            block.enable_full_covariance(True)

    def disable_full_covariance_estimation(self):
        # Kept for backward compatibility, but intentionally does not disable
        # covariance computation because covariance is now always estimated.
        self.enable_full_covariance_estimation()


if __name__ == '__main__':
    dataset_load = torch.load('../data/MultiSubArrays/SensorSourceGraphDataset.pkl')
    multi_arrays_model = MultiSubarraysModel(
        sensors_positions=[(0, 1), (1.5, 0)],
        multi_model_configuration="../configuration/multi_model_configuration.json",
    )

    for scene in dataset_load:
        x_hat = multi_arrays_model(scene)
        source_true_location = torch.Tensor(get_location_from_model_graph(scene[0], type_req='source')).float()
        print(f'{x_hat=}, {source_true_location=}, {torch.norm(x_hat-source_true_location)=}')
