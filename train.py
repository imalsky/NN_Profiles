import os
import torch
import optuna

from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from transformer_model import AtmosphericModel
from dataset import NormalizedProfilesDataset

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler,
                num_epochs=100, early_stopping_patience=10, device='cpu', save_path='Data/Model'):
    """
    Train the Transformer-based model with a learning rate scheduler.

    Parameters:
        model (torch.nn.Module): The model to train.
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
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    os.makedirs(save_path, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs_main=inputs)
            loss = criterion(outputs.view_as(targets), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # Log progress
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.3e}, Val Loss: {val_loss:.3e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return None



def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on a validation or test dataset.

    Parameters:
        model (torch.nn.Module): The trained model.
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
            outputs = model(inputs_main=inputs)
            total_loss += criterion(outputs.view_as(targets), targets).item()

    return total_loss / len(data_loader)




def train_model_from_config(config, data_folder, model_save_path, device):
    """
    Train the model with given configuration.

    Parameters:
        config (dict): Configuration for training.
        data_folder (str): Path to the normalized profiles data.
        model_save_path (str): Path to save the trained model.
        device (torch.device): Device to train the model on.
    """
    dataset = NormalizedProfilesDataset(data_folder,
                                        config["nlev"],
                                        input_variables=config["input_variables"],
                                        target_variables=config["target_variables"])

    # Apply the fraction of training data
    dataset_size = int(len(dataset) * config.get("frac_of_training_data", 1.0))
    dataset = torch.utils.data.Subset(dataset, range(dataset_size))

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)


    model = AtmosphericModel(
        nx=len(config["input_variables"]),
        ny=len(config["target_variables"]),
        nneur=config["nneur"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"]
    )
    model.to(device)

    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                num_epochs=config["epochs"],
                early_stopping_patience=10,
                device=device,
                save_path=model_save_path
            )

    val_loss = evaluate_model(model, val_loader, criterion, device)
    return val_loss

def objective(trial, params, data_folder, model_save_path, device):
    """
    Objective function for Optuna hyperparameter tuning.

    Parameters:
        trial (optuna.trial.Trial): Optuna trial object.
        params (dict): Configuration base.
        data_folder (str): Path to normalized profiles data.
        model_save_path (str): Path to save trained model.
        device (torch.device): Device for training.

    Returns:
        float: Validation loss.
    """
    params.update({
        "d_model": trial.suggest_categorical("d_model", [64, 128]),
        "nhead": trial.suggest_categorical("nhead", [4, 8]),
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 6),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [256, 512, 1024]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "layer_norm_eps": trial.suggest_loguniform("layer_norm_eps", 1e-6, 1e-4),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    })

    val_loss = train_model_from_config(params, data_folder, model_save_path, device)
    return val_loss