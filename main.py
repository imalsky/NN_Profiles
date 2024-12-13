# main.py

import os
import json
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models import BasicRNN, RNN_New

from transformer_model import AtmosphericModel

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
         model_type='BasicRNN',
         frac_of_training_data = 1.0):
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
        frac_of_training_data (float): Fraction of training data available to train on
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
        print("\n" + "=" * 70)
        print(f"{'Creating the Training Data':^70}")
        print("=" * 70)
        print()
        print()
        # Ensure required directories exist
        create_directories('Inputs', 'Data', 'Figures')

        # Generate the pressure array
        pressure_range = config['pressure_range']
        P = np.logspace(np.log10(pressure_range['min']), np.log10(pressure_range['max']), num=pressure_range['points'])

        # Generate profiles
        gen_profiles(config, P)

    if normalize_data_bool:
        print("\n" + "=" * 70)
        print(f"{'Normalizing the Data':^70}")
        print("=" * 70)
        print()

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
        global_stats = calculate_global_stats(
            input_folder,
            pressure_normalization_method,
            use_robust_scaling=True,  # Change to True for robust scaling
            clip_outliers_before_scaling=True  # Change to True to clip outliers
        )

        # Process profiles with global stats
        if global_stats:
            process_profiles(input_folder, output_folder, global_stats, pressure_normalization_method)

    # Create the RNN from the training set
    if create_rnn_model:
        print("\n" + "=" * 70)
        print(f"{'Starting the Training':^70}")
        print("=" * 70)
        print()

        # Try to run on GPU, run on CPU if not
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")

        # Set the place to save the model
        data_folder = "Data/Normalized_Profiles"
        model_save_path = "Data/Model"
        os.makedirs(model_save_path, exist_ok=True)

        # Remove existing model checkpoint if it exists
        best_model_path = os.path.join(model_save_path, "best_model.pth")
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
            print(f"Removed existing model checkpoint at {best_model_path}")

        # Load profiles
        # Inside this folder is normalization metadata, skip that one
        profile_files_full = [f for f in os.listdir(data_folder) if f.endswith(".json") and f != "normalization_metadata.json"]

        # Sometimes you might not want to train on the full thing
        num_profiles = int(len(profile_files_full) * frac_of_training_data)
        profile_files = random.sample(profile_files_full, num_profiles)
        print("Training on", num_profiles, "Profiles")

        # Error if no training data
        if not profile_files:
            raise ValueError("No profiles found in the specified data folder.")

        # Training pipeline
        first_profile_path = os.path.join(data_folder, profile_files[0])
        with open(first_profile_path, "r") as f:
            first_profile = json.load(f)

        # Check the length of the data
        assert len(first_profile["temperature"]) == nlev, "Mismatch in model levels"

        # Get the dataset for the model
        dataset = NormalizedProfilesDataset(data_folder,
                                            nlev,
                                            input_variables=input_variables,
                                            target_variables=target_variables)

        # Split dataset into training, validation, and test sets
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                                generator=torch.Generator().manual_seed(42))

        # Determine input features dynamically
        input_features = len(input_variables)
        target_features = len(target_variables)

        print()
        print(f"Epochs={epochs}, Batch size={batch_size}, Hidden layers={nneur}, Learning rate={learning_rate}")
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
        elif model_type == 'Transformer':
            model = AtmosphericModel(
                nx=input_features,
                ny=target_features,
                nneur=nneur
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")


        # Define loss criterion
        criterion = nn.MSELoss()

        # Initialize optimizer and scheduler
        #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        # Train the model
        best_val_loss, best_model_path = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            num_epochs=epochs,
            early_stopping_patience=10,
            device=device,
            save_path='Data/Model'
        )

        # Save model parameters to JSON
        model_params = {
            'model_type': model_type,
            'RNN_type': model.RNN_type,  # Fetch from the model instance
            'nx': model.nx,
            'ny': model.ny,
            'nneur': model.nneur,
            'outputs_one_longer': model.concat,  # Reflect the actual model config
            'concat': model.concat,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'input_variables': input_variables,
            'target_variables': target_variables
        }

        # Save model parameters
        model_params_path = os.path.join(model_save_path, 'model_parameters.json')
        with open(model_params_path, 'w') as f:
            json.dump(model_params, f, indent=4)
        print(f"Model parameters saved to {model_params_path}")

        # Save the trained model's state_dict
        best_model_path = os.path.join(model_save_path, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Model state_dict saved to {best_model_path}")

        # Load the best model for evaluation
        # Ensure consistency with loading mechanisms
        state_dict = torch.load(best_model_path, map_location=device)
        try:
            model.load_state_dict(state_dict, strict=True)  # Load exact matches
        except RuntimeError:
            print("State_dict mismatch; attempting partial load.")
            model_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(filtered_state_dict)
            model.load_state_dict(model_dict)

        model.to(device)
        model.eval()

        # Load the test dataset
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print("\nEvaluating on Test Set...")
        test_loss = evaluate_model(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.3e}")


if __name__ == "__main__":
    main(
        gen_profiles_bool=False,
        normalize_data_bool=False,
        create_rnn_model=True,
        epochs=500,
        nneur=(32, 32),
        batch_size=8,
        learning_rate=1e-4,
        input_variables=['pressure', 'temperature', 'flux_surface_down'],
        target_variables=['net_flux'],  # Can be single or multiple targets
        model_type='Transformer',
        frac_of_training_data=1.0
    )


