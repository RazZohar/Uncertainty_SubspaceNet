"""Subspace-Net main script 
    Details
    -------
    Name: main.py
    Authors: R Zohar
    Created: 01/10/21
    Edited: 01/12/24

    Purpose
    --------
    This script allows the user to apply the proposed algorithms,
    by wrapping all the required procedures and parameters for the simulation.
    This scripts calls the following functions:
        * create_dataset: For creating training and testing datasets 
        * training: For training DR-MUSIC model
        * evaluate_dnn_model: For evaluating subspace hybrid models

    This script requires that requirements.txt will be installed within the Python
    environment you are running this script in.

"""
# Imports
import sys
import torch
import os
import matplotlib.pyplot as plt
import warnings
from src.system_model import SystemModelParams
from src.signal_creation import *
from src.data_handler import *
from src.criterions import set_criterions
from src.training import *
from src.evaluation import evaluate
from src.plotting import initialize_figures
from pathlib import Path
from src.models import ModelGenerator

import src.create_codebook as codebook_creation


import src.qunatizer as quantizer

import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity


# Initialization
warnings.simplefilter("ignore")
os.system("cls||clear")
plt.close("all")

# Use this flag to generate graph
plot_spectrum_flag = True


def evaluate_model_command():
    global criterion, subspace_criterion, test_dataset, generic_test_dataset, samples_model, simulation_parameters, model
    # Initialize figures dict for plotting
    figures = initialize_figures()
    # Define loss measure for evaluation
    criterion, subspace_criterion = set_criterions("rmse")
    # Load datasets for evaluation
    if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
        test_dataset, generic_test_dataset, samples_model = load_datasets(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            samples_size=samples_size,
            datasets_path=datasets_path,
            train_test_ratio=train_test_ratio,
        )
    # Generate DataLoader objects
    model_test_dataset = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, drop_last=False
    )
    generic_test_dataset = torch.utils.data.DataLoader(
        generic_test_dataset, batch_size=1, shuffle=False, drop_last=False
    )
    # Load pre-trained model
    if not commands["TRAIN_MODEL"]:
        # Define an evaluation parameters instance
        simulation_parameters = (
            TrainingParams()
            .set_model(model=model_config)
            .load_model(
                loading_path=saving_path
                             / "final_models"
                             / simulation_filename
            )
        )
        model = simulation_parameters.model
    # print simulation summary details
    simulation_summary(
        system_model_params=system_model_params,
        model_type=model_config.model_type,
        phase="evaluation",
        parameters=simulation_parameters,
    )
    # Evaluate DNN models, augmented and subspace methods
    evaluate(
        model=model,
        model_type=model_config.model_type,
        model_test_dataset=model_test_dataset,
        generic_test_dataset=generic_test_dataset,
        criterion=criterion,
        subspace_criterion=subspace_criterion,
        system_model=samples_model,
        figures=figures,
        plot_spec=plot_spectrum_flag,
    )


from itertools import cycle
from matplotlib import rcParams
from cycler import cycler


if __name__ == "__main__":
    # Initialize paths
    external_data_path = Path.cwd() / "data"
    scenario_data_path = "uniform_bias_spacing"
    datasets_path = external_data_path / "datasets" / scenario_data_path
    simulations_path = external_data_path / "simulations"
    saving_path = external_data_path / "weights"
    # create folders if not exists
    datasets_path.mkdir(parents=True, exist_ok=True)
    (datasets_path / "train").mkdir(parents=True, exist_ok=True)
    (datasets_path / "test").mkdir(parents=True, exist_ok=True)
    datasets_path.mkdir(parents=True, exist_ok=True)
    simulations_path.mkdir(parents=True, exist_ok=True)
    saving_path.mkdir(parents=True, exist_ok=True)
    # Initialize time and date
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    # Operations commands
    commands = {
        "SAVE_TO_FILE": False,  # Saving results to file or present them over CMD
        "CREATE_DATA": False,  # Creating new dataset
        "LOAD_DATA": True,  # Loading data from exist dataset
        "LOAD_MODEL": False,  # Load specific model for training
        "TRAIN_MODEL": False,  # Applying training operation
        "SAVE_MODEL": True,  # Saving tuned model
        "EVALUATE_MODE": False,  # Evaluating desired algorithms
        "CREATE_CODEBOOK" : False, # Create the codebook for VQ-VAE
        "TRAIN_QUANTIZED" : False, # Train the model for the quantization
        "TRAIN_SCALAR_QUANTIZATION" : False, # Train the model for Scalar quantization

        # Source - task based quantization
        "TRAIN_MODEL_SOURCES": False,  # Applying training operation for the sources
        "EVALUATE_MODE_SOURCES": False,  # Evaluating desired algorithms
        "CREATE_CODEBOOK_SOURCES": False,  # Create the codebook for VQ-VAE
        "TRAIN_QUANTIZED_SOURCES": False,  # Train the model for the quantization

        # Online train of the model
        "TRAIN_ONLINE_SOURCES" : False,
        "EVALUATE_ONLINE_MODE_SOURCES" : True,

        # Task ignorant quantization model
        "TRAIN_MODEL_TASK_IGNORANT": False,  # Applying training operation for the sources
        "EVALUATE_MODE_SOURCES_TASK_IGNORANT": False,  # Evaluating desired algorithms
        "CREATE_CODEBOOK_SOURCES_TASK_IGNORANT": False,  # Create the codebook for VQ-VAE
        "TRAIN_QUANTIZED_SOURCES_TASK_IGNORANT": False,  # Train the model for the quantization

        "TRAIN_SCALAR_QUANTIZATION_SOURCES" : False, # Train the model for Scalar quantization
    }

    ## Graph tools
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    dash_styles = [
        '-',
        (0, (5, 5)),
        (0, (1, 5)),
        (0, (3, 5, 1, 5)),
        (0, (5, 1, 1, 1))
    ]
    # Extend dash styles to match color length
    dash_cycle = list(itertools.islice(itertools.cycle(dash_styles), len(colors)))

    # Now same length, can combine
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors) + cycler(linestyle=dash_cycle)
    CODEBOOK_SIZE = 128

    print(f'Start Executing commands')
    # Saving simulation scores to external file
    if commands["SAVE_TO_FILE"]:
        file_path = (
            simulations_path / "results" / "scores" / Path(dt_string_for_save + ".txt")
        )
        sys.stdout = open(file_path, "w")
    # Define system model parameters
    system_model_params = (
        SystemModelParams()
        .set_parameter("N", 8)
        .set_parameter("M", 3)
        .set_parameter("T", 100)
        .set_parameter("snr", 10)
        .set_parameter("signal_type", "NarrowBand")
        .set_parameter("signal_nature", "non-coherent")
        .set_parameter("eta", 0)
        .set_parameter("bias", 0.0)
        .set_parameter("sv_noise_var", 0)
        .set_parameter("codebook_size", CODEBOOK_SIZE)
    )

    # Generate model configuration
    MAXIMAL_TAU = 8
    model_config = (
        ModelGenerator()
        .set_model_type("SignalsSubspaceNet") #"TaskIgnorantSubspaceNet", SignalsSubspaceNet
        .set_diff_method("esprit")
        .set_tau(min(MAXIMAL_TAU, system_model_params.T - 1))
        .set_model(system_model_params)
    )
    print('Generating model configuration')



    # Define samples size
    samples_size = 100000  # Overall dateset size
    train_test_ratio = 0.05  # training and testing datasets ratio
    # Sets simulation filename
    simulation_filename = get_simulation_filename(
        system_model_params=system_model_params, model_config=model_config
    )
    print(f'Set model configuration and export to configuration/{simulation_filename}')
    system_model_json = system_model_params.export_to_json()
    with open(f'configuration/{simulation_filename}.json', "w") as outfile:
        outfile.write(system_model_json)

    # Print new simulation intro
    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")
    print("date and time =", dt_string)
    # Initialize seed
    set_unified_seed()
    # Datasets creation
    if commands["CREATE_DATA"]:
        # Define which datasets to generate
        create_training_data = True  # Flag for creating training data
        create_testing_data = True  # Flag for creating test data
        print("Creating Data...")
        if create_training_data:
            # Generate training dataset
            if model_config.model_type == "SignalsSubspaceNet" or model_config.model_type == "TaskIgnorantSubspaceNet":
                model_type_name = "SubspaceNet"
            else:
                model_type_name = model_config.model_type

            train_dataset, generic_train_dataset, samples_model = create_dataset(
                system_model_params=system_model_params,
                samples_size=samples_size,
                model_type=model_type_name,
                tau=model_config.tau,
                save_datasets=True,
                datasets_path=datasets_path,
                true_doa=None,
                phase="train",
            )
        if create_testing_data:
            # Generate test dataset
            test_dataset, generic_test_dataset, samples_model = create_dataset(
                system_model_params=system_model_params,
                samples_size=int(train_test_ratio * samples_size),
                model_type=model_config.model_type,
                tau=model_config.tau,
                save_datasets=True,
                datasets_path=datasets_path,
                true_doa=None,
                phase="test",
            )
    # Datasets loading
    elif commands["LOAD_DATA"]:
        if model_config.model_type == "SignalsSubspaceNet" or model_config.model_type == "TaskIgnorantSubspaceNet":
            model_type_name = "SubspaceNet"
        else:
            model_type_name = model_config.model_type
        (
            train_dataset,
            test_dataset,
            generic_test_dataset,
            samples_model,
            generic_train_dataset,
        ) = load_datasets(
            system_model_params=system_model_params,
            model_type=model_type_name,
            samples_size=samples_size,
            datasets_path=datasets_path,
            train_test_ratio=train_test_ratio,
            is_training=True,
        )

    # Training stage
    if commands["TRAIN_MODEL"]:
        # Assign the training parameters object
        simulation_parameters = (
            TrainingParams()
            .set_batch_size(1024)
            .set_epochs(80)
            .set_model(model=model_config)
            .set_optimizer(optimizer="Adam", learning_rate=0.001, weight_decay=1e-5)
            .set_training_dataset(train_dataset)
            .set_schedular(step_size=40, gamma=0.2)
            .set_criterion()
        )

        #Update to handle root music with cohernt sources
        #scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)

        if commands["LOAD_MODEL"]:
            simulation_parameters.load_model(
                loading_path=saving_path / "final_models" / simulation_filename
            )
        # Print training simulation details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )

        # Perform simulation training and evaluation stages
#        with profile(activities=[ProfilerActivity.CPU]) as prof:
#            with record_function("model_inference"):
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            model_name=simulation_filename,
            saving_path=saving_path,
        )


        #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

        # Save model weights
        if commands["SAVE_MODEL"]:

            simulation_filename = simulation_filename
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        else:
            plt.show()

        # For this purpose we use evaluate
        #evaluate_model_command()

    if commands["CREATE_CODEBOOK"]:

        CLUSTERS_COUNT = CODEBOOK_SIZE
        codebook_creation_dataset = torch.utils.data.DataLoader(
            train_dataset, batch_size=1024, shuffle=False, drop_last=False
        )


        # Load a pretrained model
        criterion, subspace_criterion = set_criterions("rmse")


        simulation_parameters = (
            TrainingParams()
            .set_model(model=model_config)
            .load_model(
                loading_path=saving_path
                             / "final_models"
                             / simulation_filename
            )
        )
        model = simulation_parameters.model
        # Load the VQ_VAE model
        simulation_filename = simulation_filename + f'_VQVAE'

        CODEBOOK_SIZE = 256
        codebook = codebook_creation.create_codebook_command(model.encoder, codebook_creation_dataset, cb_vec_dim=4, num_clusters=CODEBOOK_SIZE)
        base_simulation_name = get_simulation_filename(
        system_model_params=system_model_params, model_config=model_config
        )
        codebook_filename = "codebook_{codebook_size}_{simulation_name}.npy".format(codebook_size=CODEBOOK_SIZE, simulation_name=base_simulation_name)
        codebook_cpu = codebook.cpu()
        np.save(saving_path / codebook_filename, codebook_cpu)

        # Start the fine tunning step
        model = model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.codebook_size = CODEBOOK_SIZE
        model.set_quantize(True)
        model.quantizer.set_codebook_size(CODEBOOK_SIZE) # Set empty codebook at requested size
        model.quantizer.active_vectors = codebook
        model.quantizer.lambda_c = 1.0

        print(
            f'Created the codebook for the subspace with VQ-VAE, with codebook size = {model.quantizer.p}')

        # Train only the decoder
        for param in model.encoder.parameters():
            param.requires_grad = False

        #Apply Small train to optimaize with the codebook
        simulation_parameters = (simulation_parameters
                                .set_batch_size(1024)
                                .set_epochs(5)
                                .set_optimizer(optimizer="Adam", learning_rate=0.00001, weight_decay=1e-9)
                                .set_training_dataset(train_dataset)
                                .set_schedular(step_size=10, gamma=0.5)
                                .set_criterion()
                                )
        # Set the New optimzer for quantize and decoder only
        optimizer = optim.Adam(list(model.decoder.parameters()), lr=0.0005, weight_decay=1e-4)
        simulation_parameters.optimizer = optimizer

        # Assign schedular for learning rate decay
        simulation_parameters.schedular = (
            torch.optim.lr_scheduler.StepLR(
            simulation_parameters.optimizer, step_size=simulation_parameters.step_size, gamma=simulation_parameters.gamma
        ))
        # Assign schedular for learning rate decay
        #simulation_parameters.schedular = (
        #    torch.optim.lr_scheduler.ReduceLROnPlateau(
        #        simulation_parameters.optimizer, mode='min', factor=0.6, patience=2, verbose=True, min_lr=1e-5
        #    ))

        simulation_filename = simulation_filename + '_Quantized_{codebook_size}'.format(codebook_size=CODEBOOK_SIZE)
        # Print training simulation details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )
        # Perform simulation training and evaluation stages
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            model_name=simulation_filename,
            saving_path=saving_path,
        )
        # Save model weights
        if commands["SAVE_MODEL"]:
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        else:
            plt.show()

        # For this purpose we use evaluate
        usage_counts = model.quantizer.visualize_codebook_usage()
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(usage_counts)), usage_counts)
        plt.xlabel("Codebook Entry Index")
        plt.ylabel("Usage Count")
        plt.title(f'Codebook Entry Usage Codebook size ={CODEBOOK_SIZE}')
        plt.show()

        print(f'The usage of codebook is {(model.quantizer.general_codebook_usage / CODEBOOK_SIZE) * 100:.2f} [%]')
        #evaluate_model_command()

    if commands["TRAIN_QUANTIZED"]:
        CODEBOOK_SIZE = 256
        CLUSTERS_COUNT = CODEBOOK_SIZE

        base_simulation_name = get_simulation_filename(
            system_model_params=system_model_params, model_config=model_config
        )

        simulation_filename = base_simulation_name + f'_VQVAE_Quantized_{CODEBOOK_SIZE}'
        # Load a pretrained model
        criterion, subspace_criterion = set_criterions("rmse")
        simulation_parameters = (
            TrainingParams()
            .set_model(model=model_config)
            .load_model(
                loading_path=saving_path
                             / "final_models"
                             / simulation_filename
            )
        )
        model = simulation_parameters.model


        print(f'Load the codebook for the subspace with VQ-VAE')
        base_simulation_name = get_simulation_filename(
            system_model_params=system_model_params, model_config=model_config
        )
        codebook_filename = "codebook_{codebook_size}_{simulation_name}.npy".format(codebook_size=CODEBOOK_SIZE,
                                                                                    simulation_name=base_simulation_name)

        codebook = np.load(saving_path / codebook_filename)


        # Start the fine tunning step
        # Start the fine tunning step
        model = model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.codebook_size = CODEBOOK_SIZE
        model.set_quantize(True)
        model.quantizer.set_codebook_size(CODEBOOK_SIZE) # Set empty codebook at requested size
        model.quantizer.active_vectors = codebook
        model.quantizer.lambda_c = 1.0
        model.quantizer.apply(lambda module: codebook_creation.init_weights_lbg(module, codebook))

        # Train only the decoder
        #simulation_filename = simulation_filename + '_Quantized_Trained'
        simulation_filename = simulation_filename + '_Quantized_Trained_{codebook_size}'.format(codebook_size=CODEBOOK_SIZE)
        #Apply Small train to optimaize with the codebook
        simulation_parameters = (simulation_parameters
                                .set_batch_size(1024)
                                .set_epochs(10)
                                .set_optimizer(optimizer="Adam", learning_rate=0.0005, weight_decay=5e-6)
                                .set_training_dataset(train_dataset)
                                .set_schedular(step_size=60, gamma=0.8)
                                .set_criterion()
                                )

        # Print training simulation details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )
        # Perform simulation training and evaluation stages
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            model_name=simulation_filename,
            saving_path=saving_path,
        )
        # Save model weights
        if commands["SAVE_MODEL"]:
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        else:
            plt.show()

        # For this purpose we use evaluate
        usage_counts = model.quantizer.visualize_codebook_usage()
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(usage_counts)), usage_counts)
        plt.xlabel("Codebook Entry Index")
        plt.ylabel("Usage Count")
        plt.title(f'Codebook Entry Usage Codebook size ={CODEBOOK_SIZE}')
        plt.show()

        print(f'The usage of codebook is {(model.quantizer.general_codebook_usage / CODEBOOK_SIZE) * 100:.2f} [%]')
        # evaluate_model_command()

    if commands["TRAIN_SCALAR_QUANTIZATION"]:

        # Load a pretrained model
        criterion, subspace_criterion = set_criterions("rmse")

        simulation_parameters = (
            TrainingParams()
            .set_model(model=model_config)
            .load_model(
                loading_path=saving_path
                             / "final_models"
                             / simulation_filename
            )
        )
        model = simulation_parameters.model

        CODEBOOK_SIZE = 32

        

        quantize_creation_dataset = torch.utils.data.DataLoader(
            train_dataset, batch_size=1024, shuffle=False, drop_last=False
        )

        max_ze, min_ze = codebook_creation.get_min_max(model.encoder, quantize_creation_dataset)



        scalar_quantizer = quantizer.ElementWiseQuantizer(min_val=min_ze, max_val=max_ze, n_levels=CODEBOOK_SIZE)

        print(f'Load the codebook for the subspace with scalar')
        base_simulation_name = get_simulation_filename(
            system_model_params=system_model_params, model_config=model_config
        )



        # Start the fine tunning step
        # Start the fine tunning step
        model = model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.codebook_size = CODEBOOK_SIZE
        model.set_quantize(True)
        model.quantizer = scalar_quantizer
        # scalar Quantization

        #Apply Small train to optimaize with the codebook
        simulation_parameters = (simulation_parameters
                                .set_batch_size(1024)
                                .set_epochs(10)
                                .set_optimizer(optimizer="Adam", learning_rate=0.0005, weight_decay=5e-6)
                                .set_training_dataset(train_dataset)
                                .set_schedular(step_size=60, gamma=0.8)
                                .set_criterion()
                                )

        # Print training simulation details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )
        # Perform simulation training and evaluation stages
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            model_name=simulation_filename,
            saving_path=saving_path,
        )

        simulation_filename = simulation_filename + '_Quantized_scalar_Trained_{codebook_size}'.format(
            codebook_size=CODEBOOK_SIZE)

        # Save model weights
        if commands["SAVE_MODEL"]:
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        else:
            plt.show()

    # Training stage special model
    if commands["TRAIN_MODEL_SOURCES"]:
        # Assign the training parameters object
        simulation_parameters = (
            TrainingParams()
            .set_batch_size(1024)
            .set_epochs(40)
            .set_model(model=model_config)
            .set_optimizer(optimizer="Adam", learning_rate=0.001, weight_decay=1e-5)
            .set_training_dataset(generic_train_dataset)
            .set_schedular(step_size=20, gamma=0.2)
            .set_criterion()
        )

        # Update to handle root music with cohernt sources
        # scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)

        if commands["LOAD_MODEL"]:
            simulation_parameters.load_model(
                loading_path=saving_path / "final_models" / simulation_filename
            )
        # Print training simulation details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )

        # Perform simulation training and evaluation stages
        #        with profile(activities=[ProfilerActivity.CPU]) as prof:
        #            with record_function("model_inference"):
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            model_name=simulation_filename,
            saving_path=saving_path,
        )

        # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

        # Save model weights
        if commands["SAVE_MODEL"]:
            simulation_filename = simulation_filename
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        else:
            plt.show()

        # For this purpose we use evaluate
        # evaluate_model_command()

    if commands["CREATE_CODEBOOK_SOURCES"]:

        CLUSTERS_COUNT = CODEBOOK_SIZE
        codebook_creation_dataset = torch.utils.data.DataLoader(
            generic_train_dataset, batch_size=1024, shuffle=False, drop_last=False
        )

        # Load a pretrained model
        criterion, subspace_criterion = set_criterions("rmse")

        simulation_parameters = (
            TrainingParams()
            .set_model(model=model_config)
            .load_model(
                loading_path=saving_path
                             / "final_models"
                             / simulation_filename
            )
        )
        model = simulation_parameters.model
        # Load the VQ_VAE model
        simulation_filename = simulation_filename + f'_VQVAE'

        CODEBOOK_SIZE = 2
        codebook = codebook_creation.create_codebook_command(model.encoder_signal, codebook_creation_dataset, cb_vec_dim=4,
                                                             num_clusters=CODEBOOK_SIZE)
        base_simulation_name = get_simulation_filename(
            system_model_params=system_model_params, model_config=model_config
        )
        codebook_filename = "codebook_sources_{codebook_size}_{simulation_name}.npy".format(codebook_size=CODEBOOK_SIZE,
                                                                                    simulation_name=base_simulation_name)
        codebook_cpu = codebook.cpu()
        np.save(saving_path / codebook_filename, codebook_cpu)

        # Start the fine tunning step
        model = model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.codebook_size = CODEBOOK_SIZE
        model.set_quantize(True)
        model.quantizer_signal.set_codebook_size(CODEBOOK_SIZE)  # Set empty codebook at requested size
        model.quantizer_signal.active_vectors = codebook
        model.quantizer_signal.lambda_c = 1.0

        print(
            f'Created the codebook for the subspace with VQ-VAE, with codebook size = {model.quantizer_signal.p}')

        # Train only the decoder
        for param in model.encoder_signal.parameters():
            param.requires_grad = False

        # Apply Small train to optimaize with the codebook
        simulation_parameters = (simulation_parameters
                                 .set_batch_size(1024)
                                 .set_epochs(5)
                                 .set_optimizer(optimizer="Adam", learning_rate=0.00001, weight_decay=1e-9)
                                 .set_training_dataset(generic_train_dataset)
                                 .set_schedular(step_size=10, gamma=0.5)
                                 .set_criterion()
                                 )
        # Set the New optimzer for quantize and decoder only
        optimizer = optim.Adam(list(model.decoder_signal.parameters()), lr=0.0005, weight_decay=1e-4)
        simulation_parameters.optimizer = optimizer

        # Assign schedular for learning rate decay
        simulation_parameters.schedular = (
            torch.optim.lr_scheduler.StepLR(
                simulation_parameters.optimizer, step_size=simulation_parameters.step_size,
                gamma=simulation_parameters.gamma
            ))
        # Assign schedular for learning rate decay
        # simulation_parameters.schedular = (
        #    torch.optim.lr_scheduler.ReduceLROnPlateau(
        #        simulation_parameters.optimizer, mode='min', factor=0.6, patience=2, verbose=True, min_lr=1e-5
        #    ))

        simulation_filename = simulation_filename + '_QuantizedSources_{codebook_size}'.format(codebook_size=CODEBOOK_SIZE)
        # Print training simulation details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )
        # Perform simulation training and evaluation stages
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            model_name=simulation_filename,
            saving_path=saving_path,
        )
        # Save model weights
        if commands["SAVE_MODEL"]:
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        else:
            plt.show()

        # For this purpose we use evaluate
        usage_counts = model.quantizer_signal.visualize_codebook_usage()
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(usage_counts)), usage_counts)
        plt.xlabel("Codebook Entry Index")
        plt.ylabel("Usage Count")
        plt.title(f'Codebook Entry Usage Codebook size ={CODEBOOK_SIZE}')
        plt.show()

        print(f'The usage of codebook is {(model.quantizer_signal.general_codebook_usage / CODEBOOK_SIZE) * 100:.2f} [%]')
        # evaluate_model_command()

    if commands["TRAIN_QUANTIZED_SOURCES"]:
        CODEBOOK_SIZE = 2
        CLUSTERS_COUNT = CODEBOOK_SIZE

        base_simulation_name = get_simulation_filename(
            system_model_params=system_model_params, model_config=model_config
        )

        simulation_filename = base_simulation_name + f'_VQVAE_QuantizedSources_{CODEBOOK_SIZE}'
        # Load a pretrained model
        criterion, subspace_criterion = set_criterions("rmse")
        simulation_parameters = (
            TrainingParams()
            .set_model(model=model_config)
            .load_model(
                loading_path=saving_path
                             / "final_models"
                             / simulation_filename
            )
        )
        model = simulation_parameters.model

        print(f'Load the codebook for the subspace with VQ-VAE')
        base_simulation_name = get_simulation_filename(
            system_model_params=system_model_params, model_config=model_config
        )
        codebook_filename = "codebook_sources_{codebook_size}_{simulation_name}.npy".format(codebook_size=CODEBOOK_SIZE,
                                                                                    simulation_name=base_simulation_name)

        codebook = np.load(saving_path / codebook_filename)

        # Start the fine tunning step
        # Start the fine tunning step
        model = model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.codebook_size = CODEBOOK_SIZE
        model.set_quantize(True)
        model.quantizer_signal.set_codebook_size(CODEBOOK_SIZE)  # Set empty codebook at requested size
        model.quantizer_signal.active_vectors = codebook
        model.quantizer_signal.lambda_c = 1.0
        model.quantizer_signal.apply(lambda module: codebook_creation.init_weights_lbg(module, codebook))

        # Train only the decoder
        # simulation_filename = simulation_filename + '_Quantized_Trained'
        simulation_filename = simulation_filename + '_QuantizedSources_Trained_{codebook_size}'.format(
            codebook_size=CODEBOOK_SIZE)
        # Apply Small train to optimaize with the codebook
        simulation_parameters = (simulation_parameters
                                 .set_batch_size(1024)
                                 .set_epochs(10)
                                 .set_optimizer(optimizer="Adam", learning_rate=0.0005, weight_decay=5e-6)
                                 .set_training_dataset(generic_train_dataset)
                                 .set_schedular(step_size=60, gamma=0.8)
                                 .set_criterion()
                                 )

        # Print training simulation details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )
        # Perform simulation training and evaluation stages
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            model_name=simulation_filename,
            saving_path=saving_path,
        )
        # Save model weights
        if commands["SAVE_MODEL"]:
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        else:
            plt.show()

        # For this purpose we use evaluate
        usage_counts = model.quantizer_signal.visualize_codebook_usage()
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(usage_counts)), usage_counts)
        plt.xlabel("Codebook Entry Index")
        plt.ylabel("Usage Count")
        plt.title(f'Codebook Entry Usage Codebook size ={CODEBOOK_SIZE}')
        plt.show()

        print(f'The usage of codebook is {(model.quantizer_signal.general_codebook_usage / CODEBOOK_SIZE) * 100:.2f} [%]')
        # evaluate_model_command()

    if commands["TRAIN_SCALAR_QUANTIZATION_SOURCES"]:

        # Load a pretrained model
        criterion, subspace_criterion = set_criterions("rmse")

        simulation_parameters = (
            TrainingParams()
            .set_model(model=model_config)
            .load_model(
                loading_path=saving_path
                             / "final_models"
                             / simulation_filename
            )
        )
        model = simulation_parameters.model

        CODEBOOK_SIZE = 64

        quantize_creation_dataset = torch.utils.data.DataLoader(
            generic_train_dataset, batch_size=1024, shuffle=False, drop_last=False
        )

        max_ze, min_ze = codebook_creation.get_min_max(model.encoder_signal, quantize_creation_dataset)

        scalar_quantizer = quantizer.ElementWiseQuantizer(min_val=min_ze, max_val=max_ze, n_levels=CODEBOOK_SIZE)

        print(f'Load the codebook for the subspace with scalar')
        base_simulation_name = get_simulation_filename(
            system_model_params=system_model_params, model_config=model_config
        )

        # Start the fine tunning step
        # Start the fine tunning step
        model = model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.codebook_size = CODEBOOK_SIZE
        model.set_quantize(True)
        model.quantizer_signal = scalar_quantizer
        # scalar Quantization

        # Apply Small train to optimaize with the codebook
        simulation_parameters = (simulation_parameters
                                 .set_batch_size(1024)
                                 .set_epochs(10)
                                 .set_optimizer(optimizer="Adam", learning_rate=0.0005, weight_decay=5e-6)
                                 .set_training_dataset(generic_train_dataset)
                                 .set_schedular(step_size=60, gamma=0.8)
                                 .set_criterion()
                                 )

        # Print training simulation details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )
        # Perform simulation training and evaluation stages
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            model_name=simulation_filename,
            saving_path=saving_path,
        )

        simulation_filename = simulation_filename + '_Sources_Quantized_scalar_Trained_{codebook_size}'.format(
            codebook_size=CODEBOOK_SIZE)

        # Save model weights
        if commands["SAVE_MODEL"]:
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        else:
            plt.show()

    if commands["TRAIN_ONLINE_SOURCES"]:
        #TODO: Create an online train model
        CODEBOOK_SIZE = 256
        CLUSTERS_COUNT = CODEBOOK_SIZE

        base_simulation_name = get_simulation_filename(
            system_model_params=system_model_params, model_config=model_config
        )

        simulation_filename = base_simulation_name + f'_VQVAE_QuantizedSources_{CODEBOOK_SIZE}'
        # Load a pretrained model
        criterion, subspace_criterion = set_criterions("rmse")
        simulation_parameters = (
            TrainingParams()
            .set_model(model=model_config)
            .load_model(
                loading_path=saving_path
                             / "final_models"
                             / simulation_filename
            )
        )
        model = simulation_parameters.model

        print(f'Load the codebook for the subspace with VQ-VAE')
        base_simulation_name = get_simulation_filename(
            system_model_params=system_model_params, model_config=model_config
        )
        codebook_filename = "codebook_sources_{codebook_size}_{simulation_name}.npy".format(codebook_size=CODEBOOK_SIZE,
                                                                                            simulation_name=base_simulation_name)

        codebook = np.load(saving_path / codebook_filename)

        # Start the fine tunning step
        # Start the fine tunning step
        model = model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.codebook_size = CODEBOOK_SIZE
        model.set_quantize(True)
        model.quantizer_signal.set_codebook_size(CODEBOOK_SIZE)  # Set empty codebook at requested size
        model.quantizer_signal.active_vectors = codebook
        model.quantizer_signal.lambda_c = 1.0
        model.quantizer_signal.apply(lambda module: codebook_creation.init_weights_lbg(module, codebook))

        # Train only the decoder
        # simulation_filename = simulation_filename + '_Quantized_Trained'
        simulation_filename = simulation_filename + '_QuantizedSources_OnlineTrained_{codebook_size}'.format(
            codebook_size=CODEBOOK_SIZE)
        # Apply Small train to optimaize with the codebook
        simulation_parameters = (simulation_parameters
                                 .set_batch_size(1024)
                                 .set_epochs(10)
                                 .set_optimizer(optimizer="Adam", learning_rate=0.0005, weight_decay=5e-6)
                                 .set_training_dataset(generic_train_dataset)
                                 .set_schedular(step_size=60, gamma=0.8)
                                 .set_criterion()
                                 )

        # Print training simulation details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )
        # Perform simulation training and evaluation stages
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            model_name=simulation_filename,
            saving_path=saving_path,
        )
        # Save model weights
        if commands["SAVE_MODEL"]:
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        else:
            plt.show()

    # For the task ignorant model
    if commands["TRAIN_MODEL_TASK_IGNORANT"]:
        # Assign the training parameters object
        simulation_parameters = (
            TrainingParams()
            .set_batch_size(1024)
            .set_epochs(40)
            .set_model(model=model_config)
            .set_optimizer(optimizer="Adam", learning_rate=0.001, weight_decay=1e-5)
            .set_training_dataset(generic_train_dataset)
            .set_schedular(step_size=20, gamma=0.2)
            .set_criterion()
        )

        # Update to handle root music with cohernt sources
        # scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)

        if commands["LOAD_MODEL"]:
            simulation_parameters.load_model(
                loading_path=saving_path / "final_models" / simulation_filename
            )
        # Print training simulation details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )

        # Perform simulation training and evaluation stages
        #        with profile(activities=[ProfilerActivity.CPU]) as prof:
        #            with record_function("model_inference"):
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            model_name=simulation_filename,
            saving_path=saving_path,
        )

        # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

        # Save model weights
        if commands["SAVE_MODEL"]:
            simulation_filename = simulation_filename
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        else:
            plt.show()

        # For this purpose we use evaluate
        # evaluate_model_command()

    if commands["CREATE_CODEBOOK_SOURCES_TASK_IGNORANT"]:

        CLUSTERS_COUNT = CODEBOOK_SIZE
        codebook_creation_dataset = torch.utils.data.DataLoader(
            generic_train_dataset, batch_size=1024, shuffle=False, drop_last=False
        )

        # Load a pretrained model
        criterion, subspace_criterion = set_criterions("rmse")

        simulation_parameters = (
            TrainingParams()
            .set_model(model=model_config)
            .load_model(
                loading_path=saving_path
                             / "final_models"
                             / simulation_filename
            )
        )
        model = simulation_parameters.model
        # Load the VQ_VAE model
        simulation_filename = simulation_filename + f'_task_ignorant'

        CODEBOOK_SIZE = 2
        codebook = codebook_creation.create_codebook_command(model.encoder_signal, codebook_creation_dataset, cb_vec_dim=4,
                                                             num_clusters=CODEBOOK_SIZE)
        base_simulation_name = get_simulation_filename(
            system_model_params=system_model_params, model_config=model_config
        )
        codebook_filename = "codebook_sources_ignorant_{codebook_size}_{simulation_name}.npy".format(codebook_size=CODEBOOK_SIZE,
                                                                                    simulation_name=base_simulation_name)
        codebook_cpu = codebook.cpu()
        np.save(saving_path / codebook_filename, codebook_cpu)

        # Start the fine tunning step
        model = model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.codebook_size = CODEBOOK_SIZE
        model.set_quantize(True)
        model.quantizer_signal.set_codebook_size(CODEBOOK_SIZE)  # Set empty codebook at requested size
        model.quantizer_signal.active_vectors = codebook
        model.quantizer_signal.lambda_c = 1.0

        print(
            f'Created the codebook for the subspace with VQ-VAE, with codebook size = {model.quantizer_signal.p}')

        # Train only the decoder
        for param in model.encoder_signal.parameters():
            param.requires_grad = False

        # Apply Small train to optimaize with the codebook
        simulation_parameters = (simulation_parameters
                                 .set_batch_size(1024)
                                 .set_epochs(5)
                                 .set_optimizer(optimizer="Adam", learning_rate=0.00001, weight_decay=1e-9)
                                 .set_training_dataset(generic_train_dataset)
                                 .set_schedular(step_size=10, gamma=0.5)
                                 .set_criterion()
                                 )
        # Set the New optimzer for quantize and decoder only
        optimizer = optim.Adam(list(model.decoder_signal.parameters()), lr=0.0005, weight_decay=1e-4)
        simulation_parameters.optimizer = optimizer

        # Assign schedular for learning rate decay
        simulation_parameters.schedular = (
            torch.optim.lr_scheduler.StepLR(
                simulation_parameters.optimizer, step_size=simulation_parameters.step_size,
                gamma=simulation_parameters.gamma
            ))
        # Assign schedular for learning rate decay
        # simulation_parameters.schedular = (
        #    torch.optim.lr_scheduler.ReduceLROnPlateau(
        #        simulation_parameters.optimizer, mode='min', factor=0.6, patience=2, verbose=True, min_lr=1e-5
        #    ))

        simulation_filename = simulation_filename + '_QuantizedSources_{codebook_size}'.format(codebook_size=CODEBOOK_SIZE)
        # Print training simulation details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )
        # Perform simulation training and evaluation stages
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            model_name=simulation_filename,
            saving_path=saving_path,
        )
        # Save model weights
        if commands["SAVE_MODEL"]:
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        else:
            plt.show()

        # For this purpose we use evaluate
        usage_counts = model.quantizer_signal.visualize_codebook_usage()
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(usage_counts)), usage_counts)
        plt.xlabel("Codebook Entry Index")
        plt.ylabel("Usage Count")
        plt.title(f'Codebook Entry Usage Codebook size ={CODEBOOK_SIZE}')
        plt.show()

        print(f'The usage of codebook is {(model.quantizer_signal.general_codebook_usage / CODEBOOK_SIZE) * 100:.2f} [%]')
        # evaluate_model_command()

    if commands["TRAIN_QUANTIZED_SOURCES_TASK_IGNORANT"]:
        CODEBOOK_SIZE = 2
        CLUSTERS_COUNT = CODEBOOK_SIZE

        base_simulation_name = get_simulation_filename(
            system_model_params=system_model_params, model_config=model_config
        )

        simulation_filename = base_simulation_name + f'_task_ignorant_QuantizedSources_{CODEBOOK_SIZE}'
        # Load a pretrained model
        criterion, subspace_criterion = set_criterions("rmse")
        simulation_parameters = (
            TrainingParams()
            .set_model(model=model_config)
            .load_model(
                loading_path=saving_path
                             / "final_models"
                             / simulation_filename
            )
        )
        model = simulation_parameters.model

        print(f'Load the codebook for the subspace with VQ-VAE')
        base_simulation_name = get_simulation_filename(
            system_model_params=system_model_params, model_config=model_config
        )
        codebook_filename = "codebook_sources_ignorant_{codebook_size}_{simulation_name}.npy".format(codebook_size=CODEBOOK_SIZE,
                                                                                    simulation_name=base_simulation_name)

        codebook = np.load(saving_path / codebook_filename)

        # Start the fine tunning step
        # Start the fine tunning step
        model = model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.codebook_size = CODEBOOK_SIZE
        model.set_quantize(True)
        model.quantizer_signal.set_codebook_size(CODEBOOK_SIZE)  # Set empty codebook at requested size
        model.quantizer_signal.active_vectors = codebook
        model.quantizer_signal.lambda_c = 1.0
        model.quantizer_signal.apply(lambda module: codebook_creation.init_weights_lbg(module, codebook))

        # Train only the decoder
        # simulation_filename = simulation_filename + '_Quantized_Trained'
        simulation_filename = simulation_filename + '_QuantizedSources_Trained_{codebook_size}'.format(
            codebook_size=CODEBOOK_SIZE)
        # Apply Small train to optimaize with the codebook
        simulation_parameters = (simulation_parameters
                                 .set_batch_size(1024)
                                 .set_epochs(10)
                                 .set_optimizer(optimizer="Adam", learning_rate=0.0005, weight_decay=5e-6)
                                 .set_training_dataset(generic_train_dataset)
                                 .set_schedular(step_size=60, gamma=0.8)
                                 .set_criterion()
                                 )

        # Print training simulation details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )
        # Perform simulation training and evaluation stages
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            model_name=simulation_filename,
            saving_path=saving_path,
        )
        # Save model weights
        if commands["SAVE_MODEL"]:
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        else:
            plt.show()

        # For this purpose we use evaluate
        usage_counts = model.quantizer_signal.visualize_codebook_usage()
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(usage_counts)), usage_counts)
        plt.xlabel("Codebook Entry Index")
        plt.ylabel("Usage Count")
        plt.title(f'Codebook Entry Usage Codebook size ={CODEBOOK_SIZE}')
        plt.show()

        print(f'The usage of codebook is {(model.quantizer_signal.general_codebook_usage / CODEBOOK_SIZE) * 100:.2f} [%]')
        # evaluate_model_command()


    # Evaluation stage
    if commands["EVALUATE_MODE"]:
        # Initialize figures dict for plotting
        figures = initialize_figures()
        # Define loss measure for evaluation
        criterion, subspace_criterion = set_criterions("rmse")
        # Load datasets for evaluation
        if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
            test_dataset, generic_test_dataset, samples_model = load_datasets(
                system_model_params=system_model_params,
                model_type=model_config.model_type,
                samples_size=samples_size,
                datasets_path=datasets_path,
                train_test_ratio=train_test_ratio,
            )
        # Generate DataLoader objects
        model_test_dataset = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        generic_test_dataset = torch.utils.data.DataLoader(
            generic_test_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        # Load pre-trained model
        if not commands["TRAIN_MODEL"]:
            #simulation_filename = simulation_filename + f'_VQVAE_Quantized_{CODEBOOK_SIZE}' + '_Quantized_Trained_{codebook_size}'.format(
            #    codebook_size=CODEBOOK_SIZE)

            # Define an evaluation parameters instance
            simulation_parameters = (
                TrainingParams()
                .set_model(model=model_config)
                .load_model(
                    loading_path=saving_path
                                 / "final_models"
                                 / simulation_filename
                )
            )
            model = simulation_parameters.model
        # print simulation summary details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            phase="evaluation",
            parameters=simulation_parameters,
        )
        # Evaluate DNN models, augmented and subspace methods
        evaluate(
            model=model,
            model_type=model_config.model_type,
            model_test_dataset=model_test_dataset,
            generic_test_dataset=generic_test_dataset,
            criterion=criterion,
            subspace_criterion=subspace_criterion,
            system_model=samples_model,
            figures=figures,
            plot_spec=plot_spectrum_flag,
        )

    # Evaluation stage
    if commands["EVALUATE_MODE_SOURCES"]:
        # Initialize figures dict for plotting
        figures = initialize_figures()
        # Define loss measure for evaluation
        criterion, subspace_criterion = set_criterions("rmse")
        # Load datasets for evaluation
        if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
            test_dataset, generic_test_dataset, samples_model = load_datasets(
                system_model_params=system_model_params,
                model_type=model_config.model_type,
                samples_size=samples_size,
                datasets_path=datasets_path,
                train_test_ratio=train_test_ratio,
            )
        # Generate DataLoader objects
        model_test_dataset = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        generic_test_dataset = torch.utils.data.DataLoader(
            generic_test_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        # Load pre-trained model
        if not commands["TRAIN_MODEL_SOURCES"]:
            # simulation_filename = simulation_filename + f'_VQVAE_Quantized_{CODEBOOK_SIZE}' + '_Quantized_Trained_{codebook_size}'.format(
            #    codebook_size=CODEBOOK_SIZE)

            # Define an evaluation parameters instance
            simulation_parameters = (
                TrainingParams()
                .set_model(model=model_config)
                .load_model(
                    loading_path=saving_path
                                 / "final_models"
                                 / simulation_filename
                )
            )
            model = simulation_parameters.model
        # print simulation summary details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            phase="evaluation",
            parameters=simulation_parameters,
        )
        # Evaluate DNN models, augmented and subspace methods
        evaluate(
            model=model,
            model_type=model_config.model_type,
            model_test_dataset=generic_test_dataset,
            generic_test_dataset=generic_test_dataset,
            criterion=criterion,
            subspace_criterion=subspace_criterion,
            system_model=samples_model,
            figures=figures,
            plot_spec=plot_spectrum_flag,
        )

    if commands["EVALUATE_MODE_SOURCES_TASK_IGNORANT"]:
        # Initialize figures dict for plotting
        figures = initialize_figures()
        # Define loss measure for evaluation
        criterion, subspace_criterion = set_criterions("rmse")
        # Load datasets for evaluation
        if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
            test_dataset, generic_test_dataset, samples_model = load_datasets(
                system_model_params=system_model_params,
                model_type=model_config.model_type,
                samples_size=samples_size,
                datasets_path=datasets_path,
                train_test_ratio=train_test_ratio,
            )
        # Generate DataLoader objects
        model_test_dataset = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        generic_test_dataset = torch.utils.data.DataLoader(
            generic_test_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        # Load pre-trained model
        if not commands["TRAIN_MODEL_TASK_IGNORANT"]:
            # simulation_filename = simulation_filename + f'_VQVAE_Quantized_{CODEBOOK_SIZE}' + '_Quantized_Trained_{codebook_size}'.format(
            #    codebook_size=CODEBOOK_SIZE)

            # Define an evaluation parameters instance
            simulation_parameters = (
                TrainingParams()
                .set_model(model=model_config)
                .load_model(
                    loading_path=saving_path
                                 / "final_models"
                                 / simulation_filename
                )
            )
            model = simulation_parameters.model
        # print simulation summary details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            phase="evaluation",
            parameters=simulation_parameters,
        )
        # Evaluate DNN models, augmented and subspace methods
        evaluate(
            model=model,
            model_type=model_config.model_type,
            model_test_dataset=generic_test_dataset,
            generic_test_dataset=generic_test_dataset,
            criterion=criterion,
            subspace_criterion=subspace_criterion,
            system_model=samples_model,
            figures=figures,
            plot_spec=plot_spectrum_flag,
        )


    # Evaluation stage
    if commands["EVALUATE_ONLINE_MODE_SOURCES"]:
        CODEBOOK_SIZE = 128
        # Initialize figures dict for plotting
        figures = initialize_figures()
        # Define loss measure for evaluation
        criterion, subspace_criterion = set_criterions("rmse")
        # Load datasets for evaluation
        if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
            test_dataset, generic_test_dataset, samples_model = load_datasets(
                system_model_params=system_model_params,
                model_type=model_config.model_type,
                samples_size=samples_size,
                datasets_path=datasets_path,
                train_test_ratio=train_test_ratio,
            )
        # Generate DataLoader objects
        model_test_dataset = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        generic_test_dataset = torch.utils.data.DataLoader(
            generic_test_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        # Load pre-trained model
        if not commands["TRAIN_MODEL_SOURCES"]:
            base_simulation_name = get_simulation_filename(
                system_model_params=system_model_params, model_config=model_config
            )
            simulation_filename = base_simulation_name + f'_VQVAE_QuantizedSources_{CODEBOOK_SIZE}' + f'_QuantizedSources_Trained_{CODEBOOK_SIZE}'

            # Define an evaluation parameters instance
            simulation_parameters = (
                TrainingParams()
                .set_model(model=model_config)
                .load_model(
                    loading_path=saving_path
                                 / "final_models"
                                 / simulation_filename
                )
            )
            model = simulation_parameters.model

        # Update model params - TODO: change for correct saving
        model.quantizer_signal.lambda_c = 1.0
        model.set_quantize(True)


        # print simulation summary details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            phase="evaluation",
            parameters=simulation_parameters,
        )
        # Evaluate DNN models, augmented and subspace methods

        evaluate(
            model=model,
            model_type=model_config.model_type,
            model_test_dataset=generic_test_dataset,
            generic_test_dataset=generic_test_dataset,
            criterion=criterion,
            subspace_criterion=subspace_criterion,
            system_model=samples_model,
            figures=figures,
            plot_spec=plot_spectrum_flag,
        )


    plt.show()
    print("end")
