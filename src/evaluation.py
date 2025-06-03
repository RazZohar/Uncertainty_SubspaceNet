"""
Subspace-Net

Details
----------
Name: evaluation.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose
----------
This module provides functions for evaluating the performance of Subspace-Net and others Deep learning benchmarks,
add for conventional subspace methods. 
This scripts also defines function for plotting the methods spectrums.
In addition, 


Functions:
----------
evaluate_dnn_model: Evaluate the DNN model on a given dataset.
evaluate_augmented_model: Evaluate an augmented model that combines a SubspaceNet model.
evaluate_model_based: Evaluate different model-based algorithms on a given dataset.
add_random_predictions: Add random predictions if the number of predictions
    is less than the number of sources.
evaluate: Wrapper function for model and algorithm evaluations.


"""
# Imports
import torch.nn as nn
from matplotlib import pyplot as plt
from src.utils import device
from src.criterions import RMSPELoss, MSPELoss
from src.criterions import RMSPE, MSPE
from src.methods import MUSIC, RootMUSIC, Esprit, MVDR
from src.utils import *
from src.models import SubspaceNet
from src.plotting import plot_spectrum


def evaluate_dnn_model(
    model,
    dataset: list,
    criterion: nn.Module,
    plot_spec: bool = False,
    figures: dict = None,
    model_type: str = "SubspaceNet",
    validation_phase = False,
):
    """
    Evaluate the DNN model on a given dataset.

    Args:
        model (nn.Module): The trained model to evaluate.
        dataset (list): The evaluation dataset.
        criterion (nn.Module): The loss criterion for evaluation.
        plot_spec (bool, optional): Whether to plot the spectrum for SubspaceNet model. Defaults to False.
        figures (dict, optional): Dictionary containing figure objects for plotting. Defaults to None.
        model_type (str, optional): The type of the model. Defaults to "SubspaceNet".

    Returns:
        float: The overall evaluation loss.

    Raises:
        Exception: If the loss criterion is not defined for the specified model type.
        Exception: If the model type is not defined.
    """

    # Initialize values
    overall_loss = 0.0
    test_length = 0
    # Set model to eval mode
    model.eval()
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for data in dataset:
            X, DOA = data
            test_length += DOA.shape[0]
            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)
            # Get model output
            model_output = model(X)
            if model_type.startswith("DA-MUSIC"):
                # Deep Augmented MUSIC
                DOA_predictions = model_output
            elif model_type.startswith("DeepCNN"):
                # Deep CNN
                if isinstance(criterion, nn.BCELoss):
                    # If evaluation performed over validation set, loss is BCE
                    DOA_predictions = model_output
                    # find peaks in the pseudo spectrum of probabilities
                    DOA_predictions = (
                        get_k_peaks(361, DOA.shape[1], DOA_predictions[0]) * D2R
                    )
                    DOA_predictions = DOA_predictions.view(1, DOA_predictions.shape[0])
                elif isinstance(criterion, [RMSPELoss, MSPELoss]):
                    # If evaluation performed over testset, loss is RMSPE / MSPE
                    DOA_predictions = model_output
                else:
                    raise Exception(
                        f"evaluate_dnn_model: Loss criterion is not defined for {model_type} model"
                    )
            elif model_type.startswith("SubspaceNet"):
                # Default - SubSpaceNet
                DOA_predictions = model_output[0]
            elif model_type.startswith("SignalsSubspaceNet"):
                DOA_predictions = model_output[0]
            elif model_type.startswith("TaskIgnorantSubspaceNet"):
                # for the task igonrant validation is on the restoreation task but test will be on accuracy of DOA
                DOA_predictions = model_output[0]
                train_loss = model_output[-1]
            else:
                raise Exception(
                    f"evaluate_dnn_model: Model type {model_type} is not defined"
                )
            # Compute prediction loss
            if model_type.startswith("DeepCNN") and isinstance(criterion, RMSPELoss):
                eval_loss = criterion(DOA_predictions.float(), DOA.float())
            elif model_type.startswith("TaskIgnorantSubspaceNet") and validation_phase == True:
                eval_loss = train_loss
            else:
                eval_loss = criterion(DOA_predictions, DOA)
            # add the batch evaluation loss to epoch loss
            overall_loss += eval_loss.item()
        overall_loss = overall_loss / test_length
    # Plot spectrum for SubspaceNet model
    if plot_spec and model_type.startswith("SubspaceNet"):

        if model.diff_method == 'root_music':
            DOA_all = model_output[1]
            roots = model_output[2]

            DOA_all = model_output[1].cpu().detach().numpy()
            roots = model_output[2].cpu().detach().numpy()
            DOA = DOA.cpu().detach().numpy()
            plot_spectrum(
                predictions=DOA_all * R2D,
                true_DOA=DOA[0] * R2D,
                roots=roots,
                algorithm="SubNet+R-MUSIC",
                figures=figures,
            )

    return overall_loss

def evaluate_dnn_model_online(
    model,
    dataset: list,
    criterion: nn.Module,
    plot_spec: bool = False,
    figures: dict = None,
    model_type: str = "SubspaceNet",
    validation_phase = False,
):
    """
    Evaluate the DNN model on a given dataset while online infrenece

    Args:
        model (nn.Module): The trained model to evaluate.
        dataset (list): The evaluation dataset.
        criterion (nn.Module): The loss criterion for evaluation.
        plot_spec (bool, optional): Whether to plot the spectrum for SubspaceNet model. Defaults to False.
        figures (dict, optional): Dictionary containing figure objects for plotting. Defaults to None.
        model_type (str, optional): The type of the model. Defaults to "SubspaceNet".

    Returns:
        float: The overall evaluation loss.

    Raises:
        Exception: If the loss criterion is not defined for the specified model type.
        Exception: If the model type is not defined.
    """

    # Initialize values
    overall_loss = 0.0
    test_length = 0
    eval_loss = 0.0
    # Set model to eval mode
    model.eval()
    model.reset_online_history()

    # Init online inference parameters
    T_prime = model.sub_horizon_time_constant
    sample = dataset.dataset[0]
    signals,doa = sample
    elements, time_horizon = signals.shape
    T = time_horizon

    step_loss_history = [[] for _ in range(T//T_prime)]

    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for data in dataset:
            X, DOA = data
            test_length += DOA.shape[0]
            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)

            #Rest current eval loss and init step loss
            eval_loss = 0.0
            step_loss = 0.0

            for t in range(0, T, T_prime):
                if t + T_prime > T:  # Avoid index overflow
                    continue

                window_batch = X[:, :, t:t + T_prime]  # Shape [batch, N, T', 2]

                model_output = model(window_batch)  # Forward pass (no time index)


                if model_type.startswith("SignalsSubspaceNet"):
                    DOA_predictions = model_output[0]
                elif model_type.startswith("TaskIgnorantSubspaceNet"):
                    # for the task igonrant validation is on the restoreation task but test will be on accuracy of DOA
                    DOA_predictions = model_output[0]
                else:
                    raise Exception(
                        f"evaluate_dnn_model_online: Model type {model_type} has no online capabilities"
                    )
                # Compute prediction loss
                step_loss = criterion(DOA_predictions, DOA)
                eval_loss += step_loss.item()
                #print(f'Step index {t / T_prime } loss {step_loss.item()}')
                step_loss_history[t // T_prime].append(step_loss.item())

            # add the batch evaluation loss to epoch loss
            overall_loss += (eval_loss / (T / T_prime))

            #REset online history
            model.reset_online_history()



        overall_loss = overall_loss / test_length
        step_loss_mean = [np.mean(sub_arr) for sub_arr in step_loss_history]
        print(f'Mean of step loss {step_loss_mean}')

    # Plot spectrum for SubspaceNet model
    if plot_spec and model_type.startswith("SubspaceNet"):

        if model.diff_method == 'root_music':
            DOA_all = model_output[1]
            roots = model_output[2]

            DOA_all = model_output[1].cpu().detach().numpy()
            roots = model_output[2].cpu().detach().numpy()
            DOA = DOA.cpu().detach().numpy()
            plot_spectrum(
                predictions=DOA_all * R2D,
                true_DOA=DOA[0] * R2D,
                roots=roots,
                algorithm="SubNet+R-MUSIC",
                figures=figures,
            )

    return overall_loss


def evaluate_augmented_model(
    model: SubspaceNet,
    dataset,
    system_model,
    criterion=RMSPE,
    algorithm: str = "music",
    plot_spec: bool = False,
    figures: dict = None,
):
    """
    Evaluate an augmented model that combines a SubspaceNet model with another subspace method on a given dataset.

    Args:
    -----
        model (nn.Module): The trained SubspaceNet model.
        dataset: The evaluation dataset.
        system_model (SystemModel): The system model for the hybrid algorithm.
        criterion: The loss criterion for evaluation. Defaults to RMSPE.
        algorithm (str): The hybrid algorithm to use (e.g., "music", "mvdr", "esprit"). Defaults to "music".
        plot_spec (bool): Whether to plot the spectrum for the hybrid algorithm. Defaults to False.
        figures (dict): Dictionary containing figure objects for plotting. Defaults to None.

    Returns:
    --------
        float: The average evaluation loss.

    Raises:
    -------
        Exception: If the algorithm is not supported.
        Exception: If the algorithm is not supported
    """
    # Initialize parameters for evaluation
    hybrid_loss = []
    if not isinstance(model, SubspaceNet):
        raise Exception("evaluate_augmented_model: model is not from type SubspaceNet")
    # Set model to eval mode
    model.eval()
    # Initialize instances of subspace methods
    methods = {
        "mvdr": MVDR(system_model),
        "music": MUSIC(system_model),
        "esprit": Esprit(system_model),
        "r-music": RootMUSIC(system_model),
    }
    # If algorithm is not in methods
    if methods.get(algorithm) is None:
        raise Exception(
            f"evaluate_augmented_model: Algorithm {algorithm} is not supported."
        )
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for i, data in enumerate(dataset):
            # Debug only
            if i != len(dataset.dataset) - 1:
                continue
            X, DOA = data
            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)
            # Apply method with SubspaceNet augmentation
            method_output = methods[algorithm].narrowband(
                X=X, mode="SubspaceNet", model=model
            )

            #print(f'Finish augmentation {i} out of {len(dataset)}')
            # Calculate loss, if algorithm is "music" or "esprit"
            if not algorithm.startswith("mvdr"):
                predictions, M = method_output[0], method_output[-1]
                # If the amount of predictions is less than the amount of sources
                predictions = add_random_predictions(M, predictions, algorithm)

                DOA = DOA.cpu().detach().numpy()
                # Calculate loss criterion
                loss = criterion(predictions, DOA * R2D)
                hybrid_loss.append(loss)
            else:
                hybrid_loss.append(0)
            # Plot spectrum, if algorithm is "music" or "mvdr"
            if not algorithm.startswith("esprit"):
                if plot_spec and i == len(dataset.dataset) - 1:
                    predictions, spectrum = method_output[0], method_output[1]


                    if algorithm.startswith("r-music"):
                        DOA_all = method_output[2]
                        roots = method_output[1]
                        plot_spectrum(
                            predictions=DOA_all,
                            true_DOA=DOA[0] * R2D,
                            roots=roots,
                            algorithm="SubNet+R-MUSIC_aug",
                            figures=figures,
                        )
                    else:
                        figures[algorithm]["norm factor"] = np.max(spectrum)
                        plot_spectrum(
                            predictions=predictions,
                            true_DOA=DOA * R2D,
                            system_model=system_model,
                            spectrum=spectrum,
                            algorithm="SubNet+" + algorithm.upper(),
                            figures=figures,
                        )

                        np.save("Augmentation_music.npy", spectrum)
    return np.mean(hybrid_loss)


def evaluate_augmented_model_online(
    model,
    dataset: list,
    system_model,
    criterion=RMSPE,
    algorithm: str = "music",
    plot_spec: bool = False,
    figures: dict = None,
):
    """
    Evaluate the DNN model on a given dataset while online infrenece

    Args:
        model (nn.Module): The trained model to evaluate.
        dataset (list): The evaluation dataset.
        criterion (nn.Module): The loss criterion for evaluation.
        plot_spec (bool, optional): Whether to plot the spectrum for SubspaceNet model. Defaults to False.
        figures (dict, optional): Dictionary containing figure objects for plotting. Defaults to None.
        model_type (str, optional): The type of the model. Defaults to "SubspaceNet".

    Returns:
        float: The overall evaluation loss.

    Raises:
        Exception: If the loss criterion is not defined for the specified model type.
        Exception: If the model type is not defined.
    """

    # Initialize values
    overall_loss = 0.0
    test_length = 0
    eval_loss = 0.0
    # Set model to eval mode
    model.eval()

    # Init online inference parameters
    T_prime = model.sub_horizon_time_constant
    sample = dataset.dataset[0]
    signals,doa = sample
    elements, time_horizon = signals.shape
    T = time_horizon

    step_loss_history = [[] for _ in range(T//T_prime)]
    spectrum_history = [[] for _ in range(T//T_prime)]

    # Initialize parameters for evaluation
    hybrid_loss = []
    if not isinstance(model, SubspaceNet):
        raise Exception("evaluate_augmented_model: model is not from type SubspaceNet")
    # Set model to eval mode
    model.eval()
    model.reset_online_history()

    # Initialize instances of subspace methods
    methods = {
        "mvdr": MVDR(system_model),
        "music": MUSIC(system_model),
        "esprit": Esprit(system_model),
        "r-music": RootMUSIC(system_model),
    }
    # If algorithm is not in methods
    if methods.get(algorithm) is None:
        raise Exception(
            f"evaluate_augmented_model: Algorithm {algorithm} is not supported."
        )
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for i, data in enumerate(dataset):
            # Debug only
            if i != len(dataset.dataset) - 1:
                continue
            X, DOA = data
            test_length += DOA.shape[0]
            # Convert observations and DoA to device
            X = X.to(device)
            #DOA = DOA.to(device)
            DOA = DOA.cpu().detach().numpy()

            #Rest current eval loss and init step loss
            eval_loss = 0.0
            step_loss = 0.0

            for t in range(0, T, T_prime):
                if t + T_prime > T:  # Avoid index overflow
                    continue

                window_batch = X[:, :, t:t + T_prime]  # Shape [batch, N, T', 2]

                model_output = model(window_batch)  # Forward pass (no time index)

                #DOA_predictions = model_output[0]


                # Apply method with SubspaceNet augmentation
                method_output = methods[algorithm].narrowband(
                    X=window_batch, mode="SubspaceNet", model=model
                )

                predictions, M = method_output[0], method_output[-1]
                # If the amount of predictions is less than the amount of sources
                predictions = add_random_predictions(M, predictions, algorithm)


                # Calculate loss criterion
                step_loss = criterion(predictions, DOA * R2D)

                eval_loss += step_loss.item()
                #print(f'Step index {t / T_prime } loss {step_loss.item()}')
                step_loss_history[t // T_prime].append(step_loss.item())
                spectrum_history[t // T_prime].append(method_output[1])

            # add the batch evaluation loss to epoch loss
            overall_loss += (eval_loss / (T / T_prime))

            #REset online history
            model.reset_online_history()

        # Plot spectrum, if algorithm is "music" or "mvdr"
        """if not algorithm.startswith("esprit"):
            if plot_spec and i == len(dataset.dataset) - 1:
                predictions, spectrum = method_output[0], method_output[1]

                if algorithm.startswith("r-music"):
                    DOA_all = method_output[2]
                    roots = method_output[1]
                    plot_spectrum(
                        predictions=DOA_all,
                        true_DOA=DOA[0] * R2D,
                        roots=roots,
                        algorithm="SubNet+R-MUSIC_aug",
                        figures=figures,
                    )
                else:
                    figures[algorithm]["norm factor"] = np.max(spectrum)
                    plot_spectrum(
                        predictions=predictions,
                        true_DOA=DOA * R2D,
                        system_model=system_model,
                        spectrum=spectrum,
                        algorithm="SubNet+" + algorithm.upper(),
                        figures=figures,
                    )
        """
        overall_loss = overall_loss / test_length
        step_loss_mean = [np.mean(sub_arr) for sub_arr in step_loss_history]
        print(f'Mean of step loss {step_loss_mean} with augmentation {algorithm}')

    # Plot spectrum for SubspaceNet model
    np.save("Spectrum_music_online.npy", [spectrum[-1] for spectrum in spectrum_history])
    print(f'Doa is {DOA * R2D} ')
    for idx, spectrum_step in enumerate(spectrum_history):
        spectrum_last = spectrum_step[-1]
        figures[algorithm]["norm factor"] = np.max(spectrum_last)
        print(f'Plotting spectrum {idx + 1} / {len(spectrum_history)}')
        label_name = f'RSSN+{algorithm.upper()}+p={idx+1}'
        plot_spectrum(
            predictions=predictions,
            true_DOA=DOA * R2D,
            system_model=system_model,
            spectrum=spectrum_last,
            algorithm=algorithm,
            label=label_name,
            figures=figures,
        )

    return overall_loss


def evaluate_model_based(
    dataset: list,
    system_model,
    criterion: RMSPE,
    plot_spec=False,
    algorithm: str = "music",
    figures: dict = None,
):
    """
    Evaluate different model-based algorithms on a given dataset.

    Args:
        dataset (list): The evaluation dataset.
        system_model (SystemModel): The system model for the algorithms.
        criterion: The loss criterion for evaluation. Defaults to RMSPE.
        plot_spec (bool): Whether to plot the spectrum for the algorithms. Defaults to False.
        algorithm (str): The algorithm to use (e.g., "music", "mvdr", "esprit", "r-music"). Defaults to "music".
        figures (dict): Dictionary containing figure objects for plotting. Defaults to None.

    Returns:
        float: The average evaluation loss.

    Raises:
        Exception: If the algorithm is not supported.
    """
    # Initialize parameters for evaluation
    loss_list = []
    for i, data in enumerate(dataset):
        X, doa = data
        X = X[0]

        doa = doa.cpu().detach().numpy()

        # Root-MUSIC algorithms
        if "r-music" in algorithm:
            root_music = RootMUSIC(system_model)
            if algorithm.startswith("sps"):
                # Spatial smoothing
                predictions, roots, predictions_all, _, M = root_music.narrowband(
                    X=X, mode="spatial_smoothing"
                )
            else:
                # Conventional
                predictions, roots, predictions_all, _, M = root_music.narrowband(
                    X=X, mode="sample"
                )
            # If the amount of predictions is less than the amount of sources
            predictions = add_random_predictions(M, predictions, algorithm)
            # Calculate loss criterion
            loss = criterion(predictions, doa * R2D)
            loss_list.append(loss)
            # Plot spectrum
            if plot_spec and i == len(dataset.dataset) - 1:
                plot_spectrum(
                    predictions=predictions_all,
                    true_DOA=doa[0] * R2D,
                    roots=roots,
                    algorithm=algorithm.upper(),
                    figures=figures,
                )
        # MUSIC algorithms
        elif "music" in algorithm:
            music = MUSIC(system_model)
            if algorithm.startswith("bb"):
                # Broadband MUSIC
                predictions, spectrum, M = music.broadband(X=X)
            elif algorithm.startswith("sps"):
                # Spatial smoothing
                predictions, spectrum, M = music.narrowband(
                    X=X, mode="spatial_smoothing"
                )
            elif algorithm.startswith("music"):
                # Conventional
                predictions, spectrum, M = music.narrowband(X=X, mode="sample")
            # If the amount of predictions is less than the amount of sources
            predictions = add_random_predictions(M, predictions, algorithm)
            # Calculate loss criterion
            loss = criterion(predictions, doa * R2D)
            loss_list.append(loss)
            # Plot spectrum
            if plot_spec and i == len(dataset.dataset) - 1:
                plot_spectrum(
                    predictions=predictions,
                    true_DOA=doa * R2D,
                    system_model=system_model,
                    spectrum=spectrum,
                    algorithm=algorithm.upper(),
                    figures=figures,
                )

        # ESPRIT algorithms
        elif "esprit" in algorithm:
            esprit = Esprit(system_model)
            if algorithm.startswith("sps"):
                # Spatial smoothing
                predictions, M = esprit.narrowband(X=X, mode="spatial_smoothing")
            else:
                # Conventional
                predictions, M = esprit.narrowband(X=X, mode="sample")
            # If the amount of predictions is less than the amount of sources
            predictions = add_random_predictions(M, predictions, algorithm)


            # Calculate loss criterion
            loss = criterion(predictions, doa * R2D)
            loss_list.append(loss)

        # MVDR algorithm
        elif algorithm.startswith("mvdr"):
            mvdr = MVDR(system_model)
            # Conventional
            _, spectrum = mvdr.narrowband(X=X, mode="sample")
            # Plot spectrum
            if plot_spec and i == len(dataset.dataset) - 1:
                plot_spectrum(
                    predictions=None,
                    true_DOA=doa * R2D,
                    system_model=system_model,
                    spectrum=spectrum,
                    algorithm=algorithm.upper(),
                    figures=figures,
                )
        else:
            raise Exception(
                f"evaluate_augmented_model: Algorithm {algorithm} is not supported."
            )
    return np.mean(loss_list)


def add_random_predictions(M: int, predictions: np.ndarray, algorithm: str):
    """
    Add random predictions if the number of predictions is less than the number of sources.

    Args:
        M (int): The number of sources.
        predictions (np.ndarray): The predicted DOA values.
        algorithm (str): The algorithm used.

    Returns:
        np.ndarray: The updated predictions with random values.

    """
    # Convert to np.ndarray array
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    while predictions.shape[0] < M:
        # print(f"{algorithm}: cant estimate M sources")
        predictions = np.insert(
            predictions, 0, np.round(np.random.rand(1) * 180, decimals=2) - 90.00
        )
    return predictions


def evaluate(
    model: nn.Module,
    model_type: str,
    model_test_dataset: list,
    generic_test_dataset: list,
    criterion: nn.Module,
    subspace_criterion,
    system_model,
    figures: dict,
    plot_spec: bool = True,
    augmented_methods: list = None,
    subspace_methods: list = None,
):
    """
    Wrapper function for model and algorithm evaluations.

    Parameters:
        model (nn.Module): The DNN model.
        model_type (str): Type of the model.
        model_test_dataset (list): Test dataset for the model.
        generic_test_dataset (list): Test dataset for generic subspace methods.
        criterion (nn.Module): Loss criterion for (DNN) model evaluation.
        subspace_criterion: Loss criterion for subspace method evaluation.
        system_model: instance of SystemModel.
        figures (dict): Dictionary to store figures.
        plot_spec (bool, optional): Whether to plot spectrums. Defaults to True.
        augmented_methods (list, optional): List of augmented methods for evaluation.
            Defaults to None.
        subspace_methods (list, optional): List of subspace methods for evaluation.
            Defaults to None.

    Returns:
        None
    """
    # Set default methods for SubspaceNet augmentation
    if not isinstance(augmented_methods, list) and "SubspaceNet" in model_type:
        augmented_methods = [
            # "mvdr",
            #"r-music",
            #"esprit",
            "music",
        ]
    # Set default model-based subspace methods
    if not isinstance(subspace_methods, list):
        subspace_methods = [
            #"esprit",
            #"music",
            #"r-music",
            #"mvdr",
            # "sps-r-music",
            # "sps-esprit",
            # "sps-music"
            # "bb-music",
        ]
    # Evaluate SubspaceNet + differentiable algorithm performances
    model_test_loss = evaluate_dnn_model(
        model=model,
        dataset=model_test_dataset,
        criterion=criterion,
        plot_spec=plot_spec,
        figures=figures,
        model_type=model_type,
    )
    print(f"{model_type} Test loss = {model_test_loss}")

    if "SubspaceNet" in model_type:

        # Evaluate SubspaceNet augmented methods
        for algorithm in augmented_methods:
            loss = evaluate_augmented_model(
                model=model,
                dataset=model_test_dataset,
                system_model=system_model,
                criterion=subspace_criterion,
                algorithm=algorithm,
                plot_spec=plot_spec,
                figures=figures,
            )
            print("augmented {} test loss = {}".format(algorithm, loss * D2R))

    if model_type == 'SignalsSubspaceNet':
        # init model params
        model.init_online_history(25)

        model_test_loss_online = evaluate_dnn_model_online(
            model=model,
            dataset=model_test_dataset,
            criterion=criterion,
            plot_spec=plot_spec,
            figures=figures,
            model_type=model_type,
        )
        print(f"{model_type} Online Test loss = {model_test_loss_online}")

        print(f"{model_type} Online Augmentation ")
        # Evaluate SubspaceNet augmented methods
        for algorithm in augmented_methods:
            loss = evaluate_augmented_model_online(
                model=model,
                dataset=model_test_dataset,
                system_model=system_model,
                criterion=subspace_criterion,
                algorithm=algorithm,
                plot_spec=plot_spec,
                figures=figures,
            )
            print("augmented {} test loss = {}".format(algorithm, loss * D2R))

    # Evaluate classical subspace methods
    for algorithm in subspace_methods:
        loss = evaluate_model_based(
            generic_test_dataset,
            system_model,
            criterion=subspace_criterion,
            plot_spec=plot_spec,
            algorithm=algorithm,
            figures=figures,
        )
        print("{} test loss = {}".format(algorithm.lower(), loss * D2R))
