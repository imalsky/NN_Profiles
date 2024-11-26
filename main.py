import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from my_rnn import MyRNN  # Your original MyRNN architecture
from visualize import model_predictions  # For visualizing predictions
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress future warnings


class NormalizedProfilesDataset(Dataset):
    def __init__(self, data_folder, expected_length, include_pressure=True):
        self.data_folder = data_folder
        self.file_list = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".json")]
        self.expected_length = expected_length
        self.include_pressure = include_pressure
        self.valid_files = self._filter_valid_files()

        if not self.valid_files:
            raise ValueError(f"No valid JSON profiles of length {self.expected_length} found in {data_folder}")
    
    def _filter_valid_files(self):
        valid_files = []
        for file_path in self.file_list:
            with open(file_path, 'r') as f:
                profile = json.load(f)
                if len(profile["temperature"]) == self.expected_length and \
                   len(profile["net_flux"]) == self.expected_length and \
                   (not self.include_pressure or len(profile["pressure"]) == self.expected_length):
                    valid_files.append(file_path)
                else:
                    print(f"Skipping invalid profile: {file_path} (Incorrect length)")
        return valid_files

    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        file_path = self.valid_files[idx]
        with open(file_path, 'r') as f:
            profile = json.load(f)
        
        temperatures = torch.tensor(profile["temperature"], dtype=torch.float32)
        net_fluxes = torch.tensor(profile["net_flux"], dtype=torch.float32)

        if self.include_pressure:
            pressures = torch.tensor(profile["pressure"], dtype=torch.float32)
            inputs = torch.stack([pressures, temperatures], dim=1)
        else:
            inputs = temperatures.unsqueeze(1)

        return inputs, net_fluxes


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, early_stopping_patience, device, save_path):
    """Train the RNN model."""
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs_main=inputs)
            loss = criterion(outputs.squeeze(-1), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.3e}, Val Loss: {val_loss:.3e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break

    return best_val_loss


def evaluate_model(model, data_loader, criterion, device):
    """Evaluate the model on a validation or test dataset."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs_main=inputs)
            loss = criterion(outputs.squeeze(-1), targets)
            total_loss += loss.item()

    return total_loss / len(data_loader)


def main(train_model_bool=True, tune_params=True, include_pressure=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_folder = "Data/Normalized_Profiles"
    model_save_path = "Data/Model"
    os.makedirs(model_save_path, exist_ok=True)

    profile_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]
    if not profile_files:
        raise ValueError("No profiles found in the specified data folder.")
    
    first_profile_path = os.path.join(data_folder, profile_files[0])
    with open(first_profile_path, "r") as f:
        first_profile = json.load(f)
    
    expected_length = len(first_profile["temperature"])
    dataset = NormalizedProfilesDataset(data_folder, expected_length, include_pressure=include_pressure)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    input_features = 2 if include_pressure else 1

    if tune_params:
        # Perform hyperparameter tuning
        batch_sizes = [4, 8]
        nneur_options = [(4,4),(32,32),(128, 128)]
        learning_rates = [1e-4]

        best_config = None
        best_val_loss = float("inf")

        for batch_size in batch_sizes:
            for nneur in nneur_options:
                for lr in learning_rates:
                    print(f"Testing config: Batch size={batch_size}, Hidden layers={nneur}, Learning rate={lr}")

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                    model = MyRNN(
                        RNN_type='LSTM',
                        nx=input_features,
                        ny=1,
                        nx_sfc=0,
                        nneur=nneur,
                        outputs_one_longer=False,
                        concat=False
                    )

                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

                    val_loss = train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=100, early_stopping_patience=10, device=device, save_path=model_save_path)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_config = {"batch_size": batch_size, "nneur": nneur, "learning_rate": lr}

        print(f"Best Config: {best_config}, Best Validation Loss: {best_val_loss:.3e}")

        batch_size, nneur, lr = best_config["batch_size"], best_config["nneur"], best_config["learning_rate"]

    else:
        # Use default parameters
        batch_size, nneur, lr = 4, (32, 32), 1e-4

        print(f"Using default parameters: Batch size={batch_size}, Hidden layers={nneur}, Learning rate={lr}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = MyRNN(
            RNN_type='LSTM',
            nx=input_features,
            ny=1,
            nx_sfc=0,
            nneur=nneur,
            outputs_one_longer=False,
            concat=False
        )

        criterion = nn.MSELoss()

        #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        optimizer = optim.Adam(model.parameters())
        #optimizer = optim.AdamW(model.parameters())


        print("Starting Training...")
        train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=200, early_stopping_patience=10, device=device, save_path=model_save_path)

    # Load the best or default model for evaluation
    model = MyRNN(
        RNN_type='LSTM',
        nx=input_features,
        ny=1,
        nx_sfc=0,
        nneur=nneur,
        outputs_one_longer=False,
        concat=False
    )
    model.load_state_dict(torch.load(os.path.join(model_save_path, "best_model.pth")))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\nEvaluating on Test Set...")
    test_loss = evaluate_model(model, test_loader, nn.MSELoss(), device)
    print(f"Test Loss: {test_loss:.3e}")

    print("\nVisualizing Predictions...")
    model_predictions(model, test_loader, save_path="Figures", device=device, N=5)


if __name__ == "__main__":
    main(train_model_bool=True, tune_params=False, include_pressure=True)
