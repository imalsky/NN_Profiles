# main.py

import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models import BasicRNN, RNN_New
from dataset import NormalizedProfilesDataset
from train import train_model, evaluate_model
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main(epochs=100,
         nneur=(32, 32),
         batch_size=8,
         learning_rate=1e-4,
         weight_decay=1e-4,
         include_Tstar=False,
         model_type='BasicRNN'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_folder = "Data/Normalized_Profiles"
    model_save_path = "Data/Model"
    os.makedirs(model_save_path, exist_ok=True)

    # Remove existing model checkpoint if it exists
    best_model_path = os.path.join(model_save_path, "best_model.pth")
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        print(f"Removed existing model checkpoint at {best_model_path}")

    profile_files = [
        f for f in os.listdir(data_folder)
        if f.endswith(".json") and f != "normalization_metadata.json"
    ]
    if not profile_files:
        raise ValueError("No profiles found in the specified data folder.")

    # Training pipeline
    first_profile_path = os.path.join(data_folder, profile_files[0])
    with open(first_profile_path, "r") as f:
        first_profile = json.load(f)

    expected_length = len(first_profile["temperature"])
    dataset = NormalizedProfilesDataset(data_folder, expected_length, include_Tstar=include_Tstar)

    # Split dataset into training, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Determine input features
    input_features = 2  # Pressure and Temperature are always included
    if include_Tstar:
        input_features += 1

    print(f"Using parameters: Epochs={epochs}, Batch size={batch_size}, Hidden layers={nneur}, Learning rate={learning_rate}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    if model_type == 'BasicRNN':
        model = BasicRNN(
            RNN_type='LSTM',
            nx=input_features,
            ny=1,
            nx_sfc=0,
            nneur=nneur,
            outputs_one_longer=False,
            concat=False
        )
    elif model_type == 'RNN_New':
        model = RNN_New(
            RNN_type='LSTM',
            nx=input_features,
            ny=1,
            nneur=nneur,
            outputs_one_longer=False,
            concat=False
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Define loss criterion
    criterion = nn.MSELoss()

    # Initialize optimizer and scheduler
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    print("Starting Training...")

    # Train the model
    train_model(
        model, train_loader, val_loader, optimizer, criterion, scheduler,
        num_epochs=epochs, early_stopping_patience=10, device=device, save_path=model_save_path
    )

    # Save model parameters to JSON file
    model_params = {
        'model_type': model_type,
        'RNN_type': 'LSTM',
        'nx': input_features,
        'ny': 1,
        'nx_sfc': 0,
        'nneur': nneur,
        'outputs_one_longer': False,
        'concat': False,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'include_Tstar': include_Tstar
    }

    # Save model parameters to JSON
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
    # You can adjust these parameters as needed
    # Models are:
    # BasicRNN
    # RNN_New
    main(
        epochs=200,
        nneur=(32, 32),
        batch_size=8,
        learning_rate=1e-4,
        weight_decay=1e-4,
        include_Tstar=False,
        model_type='RNN_New'
    )
