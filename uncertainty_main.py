import json
import argparse
from pathlib import Path
import torch

from src.utils import set_unified_seed
from src.system_model import SystemModelParams
from src.data_handler import create_dataset

from src.models import esprit, root_music, doa_covariance_from_eig, empirical_error_cov

def create_dataset_from_config(configuration_file, dataset_path, samples_size, phase, snr=None):
    # Load the dataset or create

    with open(configuration_file, 'r') as f:
        subarray_configuration = json.load(f)

    system_model_params = (
        SystemModelParams()
    ).set_params_from_json(subarray_configuration)

    if snr != None:
        system_model_params = system_model_params.set_parameter("snr", snr)

    train_dataset, generic_train_dataset, samples_model = create_dataset(
        system_model_params=system_model_params,
        samples_size=samples_size,
        model_type="SubspaceNet",
        tau=min(8, system_model_params.T - 1), # default TAU
        save_datasets=True,
        datasets_path=dataset_path,
        true_doa=None,
        phase=phase,
    )

    return train_dataset, generic_train_dataset, samples_model

def calculate_cov_batch(X):
    # X shape (M, N_snap)
    Xc = X - X.mean(dim=-1, keepdim=True)
    N_snap = Xc.shape[-1]
    return (Xc @ Xc.conj().transpose(-1, -2)) / N_snap, N_snap


def check_coveriance_with_snr(snr):
    #train_dataset, generic_train_dataset, samples_model = create_dataset_from_config(configuration_file,
    #                                                                                 dataset_path=datasets_path,
    #                                                                                 samples_size=10, phase="train")
    test_dataset, generic_test_dataset, samples_model_test = create_dataset_from_config(configuration_file,
                                                                                        dataset_path=datasets_path,
                                                                                        samples_size=1000, phase="test", snr=snr)

    model_test_dataset = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, drop_last=False
    )
    generic_test_dataset = torch.utils.data.DataLoader(
        generic_test_dataset, batch_size=1, shuffle=False, drop_last=False
    )
    sigma_list = []
    var_sigma_list = []
    emprical_sigma_list = []
    var_emprical_sigma_list = []
    predicated_doa_list = []
    true_doa = []
    for i, data in enumerate(generic_test_dataset):
        X, doa = data
        X = X[0]
        true_doa.append(torch.Tensor(doa).squeeze(dim=0))
        doa = doa.cpu().detach().numpy()  # DoA is in Radians

        Rx, N_snap = calculate_cov_batch(X)
        predicated_doa, subspace = esprit(torch.Tensor(Rx).unsqueeze(dim=0), 3, 1)
        eigenvalues, eigenvectors = subspace[0]
        idx = eigenvalues.real.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        r_true = 3

        sigma = \
        doa_covariance_from_eig(torch.Tensor(eigenvalues).unsqueeze(dim=0), torch.Tensor(eigenvectors).unsqueeze(dim=0),
                                torch.Tensor(predicated_doa[0]), N_snap)[0]

        #print("σ² estimate: ", eigenvalues[r_true:].real.mean())
        #print("trace(Dᴴ P⊥ D)⁻¹ :", torch.real(torch.linalg.inv(
        #    (eigenvectors[:, r_true:]  # Un
        #     @ eigenvectors[:, r_true:].conj().T)
        #)).trace())

        sigma_list.append(sigma)
        var_sigma_list.append(sigma.trace())

        predicated_doa_list.append(torch.Tensor(predicated_doa).squeeze(dim=0))
    true_doa_tensor = torch.stack(true_doa)
    predicted_doa_tensor = torch.stack(predicated_doa_list)
    sigma_tensor = torch.stack(sigma_list)
    sigma_avg = torch.mean(sigma_tensor, dim=0)
    emprical_sigma = empirical_error_cov(true_doa_tensor, predicted_doa_tensor)
    print(f'Var sigma avg - cov_from_doa across dataset {sigma_avg.trace()} [rad^2]')
    print(f'Var sigma mean - emprical E[eeH]- across dataset {emprical_sigma.trace()} [rad^2]\n')


if __name__ == '__main__':
    configuration_file = './configuration/SignalsSubspaceNet_M=3_T=100_SNR_10_tau=None_NarrowBand_diff_method=None_non-coherent_eta=0_bias=0.0_sv_noise=0.json'
    external_data_path = Path.cwd() / "data"
    scenario_data_path = "uncertainty"
    datasets_path = external_data_path / "datasets" / scenario_data_path
    SNR_LIST = [-3.0, 0.0, 3.0, 10.0, 20.0]
    print(f'Running with snr values of {SNR_LIST}\n')
    # Generate DataLoader objects
    set_unified_seed()

    for snr in SNR_LIST:
        print(f'Running with snr {snr}\n')
        check_coveriance_with_snr(snr=snr)
