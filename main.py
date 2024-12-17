import os
import json
import torch
import numpy as np
import warnings
import optuna

from create_training import gen_profiles
from normalize import calculate_global_stats, process_profiles
from utils import load_config, create_directories
from train import train_model_from_config, objective

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main(gen_profiles_bool: bool = False,
         normalize_data_bool: bool = False,
         create_model: bool = False,
         create_and_hypertune: bool = False) -> None:
    """
    Main function that orchestrates data generation, normalization, and
    model training (with optional hyperparameter tuning).

    Args:
        gen_profiles_bool (bool): If True, generate training profiles.
        normalize_data_bool (bool): If True, normalize the data.
        create_model (bool): If True, create and train the model.
        create_and_hypertune (bool): If True, perform hyperparameter tuning.

    Raises:
        ValueError: If both create_model and create_and_hypertune are True.
        FileNotFoundError: If the JSON parameter file is not found.
        KeyError: If the config file lacks a 'pressure_range' key.
    """
    # Safety check
    if create_model and create_and_hypertune:
        raise ValueError(
            "Both 'create_model' and 'create_and_hypertune' cannot be True at the same time."
        )

    # Load parameters and configuration
    params_path = "inputs/model_input_params.json"
    try:
        with open(params_path, "r") as f:
            params = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Parameter file not found at {params_path}") from e

    config = load_config(config_file='inputs/parameters.json')

    # Make sure config keys exist
    if "pressure_range" not in config:
        raise KeyError("Config missing 'pressure_range' key in parameters.json")

    params["nlev"] = config['pressure_range']['points']

    # Generate profiles
    if gen_profiles_bool:
        print("\n" + "=" * 70)
        print(f"{'Creating the Training Data':^70}")
        print("=" * 70 + "\n")
        create_directories('inputs', 'data', 'figures')

        pressure_range = config['pressure_range']
        P = np.logspace(
            np.log10(pressure_range['min']),
            np.log10(pressure_range['max']),
            num=pressure_range['points']
        )
        gen_profiles(config, P)

    # Normalize data
    if normalize_data_bool:
        print("\n" + "=" * 70)
        print(f"{'Normalizing the Data':^70}")
        print("=" * 70 + "\n")
        input_folder = "data/profiles"
        output_folder = "data/normalize_profiles"
        os.makedirs(output_folder, exist_ok=True)

        # Remove old normalized files
        for file_name in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Calculate global scaling statistics
        global_stats = calculate_global_stats(
            input_folder=input_folder,
            method='min-max',
            use_robust_scaling=True,
            clip_outliers_before_scaling=True
        )

        # Normalize the profiles
        process_profiles(
            input_folder=input_folder,
            output_folder=output_folder,
            global_stats=global_stats,
            method='min-max'
        )

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_folder = "data/normalize_profiles"
    model_save_path = "data/model"
    os.makedirs(model_save_path, exist_ok=True)

    # Print GPU info
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Hyperparameter tuning
    if create_and_hypertune:
        print("\n" + "=" * 70)
        print(f"{'Starting Hyperparameter Tuning':^70}")
        print("=" * 70 + "\n")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, params, data_folder, model_save_path, device), n_trials=20)

        print("\nBest hyperparameters found:")
        print(study.best_params)

        # --- Save best hyperparams to a JSON file ---
        best_params_output = "data/model/best_hparams.json"
        with open(best_params_output, "w") as fp:
            json.dump(study.best_params, fp, indent=2)
        print(f"Best hyperparams saved to {best_params_output}")

    # Direct model training
    elif create_model:
        print("\n" + "=" * 70)
        print(f"{'Starting Model Training':^70}")
        print("=" * 70 + "\n")
        train_model_from_config(params, data_folder, model_save_path, device)

    else:
        print("Not creating the ML model")


if __name__ == "__main__":
    main(
        gen_profiles_bool=False,
        normalize_data_bool=False,
        create_model=False,
        create_and_hypertune=True
    )
