# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import FluxDataset
from model import FluxPredictor
import os
import itertools
import matplotlib.pyplot as plt
import time
import numpy as np

# Load dataset
data_folder = 'Data'
dataset = FluxDataset(data_folder)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# Data loaders
batch_size = 1  # Process one profile at a time
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Hyperparameters (tuned for better performance)
hidden_size = 64
num_layers = 2
learning_rate = 0.001
dropout = 0.2
weight_decay = 1e-5
num_epochs = 50  # Adjust as needed

# Initialize model, optimizer, and loss function
model = FluxPredictor(
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout
)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

# Custom loss function to enforce net flux conservation
def loss_function(outputs, targets):
    mse_loss = nn.MSELoss()(outputs, targets)
    # Net flux conservation penalty
    total_flux_pred = torch.sum(outputs)
    total_flux_true = torch.sum(targets)
    flux_conservation_penalty = torch.abs(total_flux_pred - total_flux_true)
    # Total loss combines MSE and conservation penalty
    total_loss = mse_loss + 0.01 * flux_conservation_penalty  # Penalty weight can be tuned
    return total_loss

# Training loop with time estimation
print("Starting training...")
start_time = time.time()
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0.0

    for i, (inputs, targets, Tstar) in enumerate(train_loader):
        # inputs shape: (batch_size=1, seq_length, 2)
        # targets shape: (seq_length,)
        # Tstar shape: scalar

        # Forward pass
        outputs = model(inputs, Tstar)
        loss = loss_function(outputs, targets[0])

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets, Tstar in val_loader:
            outputs = model(inputs, Tstar)
            val_loss += loss_function(outputs, targets[0]).item()

    avg_val_loss = val_loss / len(val_loader)

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    time_elapsed = epoch_end_time - start_time
    time_per_epoch = time_elapsed / (epoch + 1)
    epochs_left = num_epochs - (epoch + 1)
    time_left = epochs_left * time_per_epoch

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
          f"Time Left: {time_left / 60:.2f} minutes")

# Save the trained model
torch.save(model.state_dict(), 'best_flux_model.pth')
print("Training completed.")

# Load the best model for evaluation (optional here since we have the trained model)
best_model = model
best_model.eval()

# Evaluate on the validation set
with torch.no_grad():
    targets_all = []
    outputs_all = []
    for inputs, targets, Tstar in val_loader:
        outputs = best_model(inputs, Tstar)
        targets_all.extend(targets[0].numpy())
        outputs_all.extend(outputs.numpy())

    targets_all = np.array(targets_all)
    outputs_all = np.array(outputs_all)

    # Plot true vs. predicted net fluxes
    plt.figure(figsize=(10, 6))
    plt.plot(targets_all, label='True Net Fluxes')
    plt.plot(outputs_all, label='Predicted Net Fluxes')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Net Flux')
    plt.title('True vs Predicted Net Fluxes')
    plt.show()
