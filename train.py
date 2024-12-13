# train.py

import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler,
                num_epochs=100, early_stopping_patience=10, device='cpu', save_path='Data/Model'):
    """
    Train the RNN model with a learning rate scheduler.

    Parameters:
        model (torch.nn.Module): The RNN model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        num_epochs (int): Number of training epochs.
        early_stopping_patience (int): Patience for early stopping.
        device (torch.device or str): Device to train on.
        save_path (str): Path to save the best model.

    Returns:
        float: Best validation loss achieved.
        str: Path to the best model.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    log_path = os.path.join(save_path, "training_log.txt")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs_main=inputs)

            # Ensure outputs and targets have the same shape
            if outputs.shape != targets.shape:
                outputs = outputs.view_as(targets)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backpropagation and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Gradient clipping
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader, criterion, device)

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log training info
        with open(log_path, "a") as log_file:
            log_file.write(f"Epoch {epoch+1}/{num_epochs}, LR: {current_lr:.2e}, "
                           f"Train Loss: {train_loss:.3e}, Val Loss: {val_loss:.3e}\n")

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, LR: {current_lr:.2e}, "
                  f"Train Loss: {train_loss:.3e}, Val Loss: {val_loss:.3e}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}!")
                break

    return best_val_loss, os.path.join(save_path, "best_model.pth")


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on a validation or test dataset.

    Parameters:
        model (torch.nn.Module): The trained RNN model.
        data_loader (DataLoader): DataLoader for the validation or test set.
        criterion (torch.nn.Module): Loss function.
        device (torch.device or str): Device to evaluate on.

    Returns:
        float: Average loss over the dataset.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs_main=inputs)

            # Ensure outputs and targets have the same shape
            if outputs.shape != targets.shape:
                outputs = outputs.view_as(targets)

            # Compute loss
            loss = criterion(outputs, targets)

            total_loss += loss.item()

    return total_loss / len(data_loader)
