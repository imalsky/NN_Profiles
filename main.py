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

def main(gen_profiles_bool=False,
         normalize_data_bool=False,
         create_model=False,
         create_and_hypertune=False):
    """
    Main function to generate profiles, normalize data, or train the model.

    Parameters:
        gen_profiles_bool (bool): If True, generate training profiles.
        normalize_data_bool (bool): If True, normalize the data.
        create_model (bool): If True, create and train the model.
        create_and_hypertune (bool): If True, perform hyperparameter tuning.
    """

    if create_model and create_and_hypertune:
        raise ValueError("Both 'create_model' and 'create_and_hypertune' cannot be True at the same time.")

    # Load parameters and configuration
    params_path = "Inputs/model_input_params.json"
    with open(params_path, "r") as f:
        params = json.load(f)
    config = load_config(config_file='Inputs/parameters.json')
    params["nlev"] = config['pressure_range']['points']

    # Generate profiles
    if gen_profiles_bool:
        print("\n" + "=" * 70)
        print(f"{'Creating the Training Data':^70}")
        print("=" * 70 + "\n")
        create_directories('Inputs', 'Data', 'Figures')
        pressure_range = config['pressure_range']
        P = np.logspace(np.log10(pressure_range['min']), np.log10(pressure_range['max']), num=pressure_range['points'])
        gen_profiles(config, P)

    # Normalize data
    if normalize_data_bool:
        print("\n" + "=" * 70)
        print(f"{'Normalizing the Data':^70}")
        print("=" * 70 + "\n")
        input_folder = "Data/Profiles"
        output_folder = "Data/Normalized_Profiles"
        os.makedirs(output_folder, exist_ok=True)

        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Get the global param values
        global_stats = calculate_global_stats(input_folder,
                                              'min-max',
                                              use_robust_scaling=True,
                                              clip_outliers_before_scaling=True)

        # Normalize the profils
        process_profiles(input_folder, output_folder, global_stats, 'min-max')

    # Common paths and device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_folder = "Data/Normalized_Profiles"
    model_save_path = "Data/Model"
    os.makedirs(model_save_path, exist_ok=True)

    # Hyperparameter tuning or model training
    if create_and_hypertune:
        print("\n" + "=" * 70)
        print(f"{'Starting Hyperparameter Tuning':^70}")
        print("=" * 70 + "\n")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, params, data_folder, model_save_path, device), n_trials=20)
        print("\nBest hyperparameters found:")
        print(study.best_params)

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
        create_model=True,
        create_and_hypertune=False
    )
