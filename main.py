import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from my_rnn import MyRNN  # Your original MyRNN architecture
from visualize import model_predictions  # For visualizing predictions

class NormalizedProfilesDataset(Dataset):
    def __init__(self, data_folder, expected_length):
        self.data_folder = data_folder
        self.file_list = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".json")]
        self.expected_length = expected_length
        self.valid_files = self._filter_valid_files()

        if not self.valid_files:
            raise ValueError(f"No valid JSON profiles of length {self.expected_length} found in {data_folder}")
    
    def _filter_valid_files(self):
        valid_files = []
        for file_path in self.file_list:
            with open(file_path, 'r') as f:
                profile = json.load(f)
                if len(profile["pressure"]) == self.expected_length and \
                   len(profile["temperature"]) == self.expected_length and \
                   len(profile["net_flux"]) == self.expected_length:
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
        
        pressures = torch.tensor(profile["pressure"], dtype=torch.float32)
        temperatures = torch.tensor(profile["temperature"], dtype=torch.float32)
        net_fluxes = torch.tensor(profile["net_flux"], dtype=torch.float32)
        
        inputs = torch.stack([pressures, temperatures], dim=1)
        return inputs, net_fluxes

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, early_stopping_patience, device):
    """Train the RNN model."""
    model.to(device)  # Move the model to the specified device
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs_main=inputs)
            loss = criterion(outputs.squeeze(-1), targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)

        # Evaluate on validation set
        val_loss = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "Data/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break

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

def main(train_model_bool=True):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if train_model_bool:
        # Paths and parameters
        data_folder = "Data/Normalized_Profiles"

        # Automatically determine the expected length
        profile_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]
        if not profile_files:
            raise ValueError("No profiles found in the specified data folder.")
        
        first_profile_path = os.path.join(data_folder, profile_files[0])
        with open(first_profile_path, "r") as f:
            first_profile = json.load(f)
        
        # Use the length of the "temperature" field as the expected length
        expected_length = len(first_profile["temperature"])
                
        # Create dataset
        dataset = NormalizedProfilesDataset(data_folder, expected_length)
        
        # Split dataset into train, validation, and test sets
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Initialize model
        model = MyRNN(
            RNN_type='LSTM',
            nx=2,  # Number of input features (e.g., pressure, temperature)
            ny=1,  # Number of output features (e.g., net_flux)
            nx_sfc=0,  # Set to 0 if no surface inputs are provided
            nneur=(64, 64),
            outputs_one_longer=False,
            concat=False
        )

        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

        # Train the model
        print("Starting Training...")
        train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=100, early_stopping_patience=10, device=device)
        
        # Evaluate on test set
        print("\nEvaluating on Test Set...")
        test_loss = evaluate_model(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}")
    
    # Visualize predictions
    print("\nVisualizing Predictions...")
    model_predictions(model, test_loader, save_path="Figures", device=device, N=5)

if __name__ == "__main__":
    train_model_bool = True

    # Ensure Data folder exists
    os.makedirs("Data", exist_ok=True)
    main(train_model_bool)
