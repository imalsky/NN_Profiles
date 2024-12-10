# main.py

import os
import json
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models import BasicRNN, RNN_New
from dataset import NormalizedProfilesDataset
from train import train_model, evaluate_model
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from create_training import gen_profiles
from normalize import calculate_global_stats, process_profiles
import warnings
import random

from utils import (
    load_config,
    create_directories,
    sample_constant_or_distribution
)

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main(gen_profiles_bool=False,
         normalize_data_bool=False,
         create_rnn_model=False,
         epochs=100,
         nneur=(32, 32),
         batch_size=8,
         learning_rate=1e-4,
         input_variables=None,
         target_variables=None,
         model_type='BasicRNN'):
    """
    Main function to generate profiles, normalize data, and train the RNN model.

    Parameters:
        gen_profiles_bool (bool): If True, generate training profiles.
        normalize_data_bool (bool): If True, normalize the data.
        create_rnn_model (bool): If True, create and train the RNN model.
        epochs (int): Number of training epochs.
        nneur (tuple): Number of neurons in each RNN layer.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        input_variables (list of str): List of input variable names.
        target_variables (list of str): List of target variable names to predict.
        model_type (str): Type of RNN model to use ('BasicRNN' or 'RNN_New').
    """

    if input_variables is None:
        input_variables = ['pressure', 'temperature']  # Default input variables

    if target_variables is None:
        target_variables = ['heating_rate']  # Default target variables

    # Load the run params
    config = load_config(config_file='Inputs/parameters.json')
    nlev = config['pressure_range']['points']

    # Generate the training set
    if gen_profiles_bool:
        # Ensure required directories exist
        create_directories('Inputs', 'Data', 'Figures')

        # Generate the pressure array
        pressure_range = config['pressure_range']
        P = np.logspace(np.log10(pressure_range['min']), np.log10(pressure_range['max']), num=pressure_range['points'])

        # Generate profiles
        gen_profiles(config, P)

    if normalize_data_bool:
        pressure_normalization_method = 'min-max'  # Options: 'standard' or 'min-max'

        # Create the output directory if it doesn't exist
        input_folder = "Data/Profiles"
        output_folder = "Data/Normalized_Profiles"
        os.makedirs(output_folder, exist_ok=True)

        # Delete existing files in the output directory
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("Deleted previous Normalized Profiles")

        # Run the normalization process
        global_stats = calculate_global_stats(input_folder, pressure_normalization_method)

        # Process profiles with global stats
        if global_stats:
            process_profiles(input_folder, output_folder, global_stats, pressure_normalization_method)

    # Create the RNN from the training set
    if create_rnn_model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_folder = "Data/Normalized_Profiles"
        model_save_path = "Data/Model"
        os.makedirs(model_save_path, exist_ok=True)

        # Remove existing model checkpoint if it exists
        best_model_path = os.path.join(model_save_path, "best_model.pth")
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
            print(f"Removed existing model checkpoint at {best_model_path}")

        # Load profiles
        profile_files_full = [f for f in os.listdir(data_folder) if f.endswith(".json") and f != "normalization_metadata.json"]
        num_profiles = int(len(profile_files_full) / 1)
        profile_files = random.sample(profile_files_full, num_profiles)
        print("Training on", num_profiles, "Profiles")

        if not profile_files:
            raise ValueError("No profiles found in the specified data folder.")

        # Training pipeline
        first_profile_path = os.path.join(data_folder, profile_files[0])
        with open(first_profile_path, "r") as f:
            first_profile = json.load(f)

        expected_length = nlev
        if len(first_profile["temperature"]) != expected_length:
            raise ValueError("Something is wrong with the levels of the model")

        dataset = NormalizedProfilesDataset(data_folder,
                                            expected_length,
                                            input_variables=input_variables,
                                            target_variables=target_variables)

        # Split dataset into training, validation, and test sets
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )

        # Determine input features dynamically
        input_features = len(input_variables)
        target_features = len(target_variables)

        print(f"Using parameters: Epochs={epochs}, Batch size={batch_size}, Hidden layers={nneur}, Learning rate={learning_rate}")
        print(f"Input Variables: {input_variables}")
        print(f"Target Variables: {target_variables}")

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        if model_type == 'BasicRNN':
            model = BasicRNN(
                RNN_type='LSTM',
                nx=input_features,
                ny=target_features,
                nx_sfc=0,
                nneur=nneur,
                outputs_one_longer=False,
                concat=False
            )
        elif model_type == 'RNN_New':
            model = RNN_New(
                RNN_type='LSTM',
                nx=input_features,
                ny=target_features,
                nneur=nneur,
                outputs_one_longer=False,
                concat=False
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Define loss criterion
        criterion = nn.MSELoss()

        # Initialize optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        print("Starting Training...")

        # Train the model
        train_model(model,
                    train_loader,
                    val_loader,
                    optimizer,
                    criterion,
                    scheduler,
                    num_epochs=epochs,
                    early_stopping_patience=10,
                    device=device,
                    save_path=model_save_path
                    )

        # Save model parameters to JSON
        model_params = {
            'model_type': model_type,
            'RNN_type': 'LSTM',
            'nx': input_features,
            'ny': target_features,
            'nx_sfc': 0,
            'nneur': nneur,
            'outputs_one_longer': False,
            'concat': False,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'input_variables': input_variables,
            'target_variables': target_variables
        }

        model_params_path = os.path.join(model_save_path, 'model_parameters.json')
        with open(model_params_path, 'w') as f:
            json.dump(model_params, f, indent=4)
        print(f"Model parameters saved to {model_params_path}")

        # Load the best model for evaluation
        model.load_state_dict(torch.load(os.path.join(model_save_path, "best_model.pth")))

        # Load the test dataset
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print("\nEvaluating on Test Set...")
        test_loss = evaluate_model(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.3e}")


if __name__ == "__main__":
    main(
        gen_profiles_bool=False,
        normalize_data_bool=True,
        create_rnn_model=True,
        epochs=500,
        nneur=(32, 32),
        batch_size=4,
        learning_rate=1e-4,
        input_variables=['pressure', 'temperature', 'Tstar', 'flux_surface_down'],
        target_variables=['net_flux'],  # Can be single or multiple targets
        model_type='BasicRNN'
    )
