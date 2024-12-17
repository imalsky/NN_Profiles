import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Tuple

from transformer_model import AtmosphericModel
from dataset import NormalizedProfilesDataset


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                scheduler: optim.lr_scheduler._LRScheduler,
                num_epochs: int = 100,
                early_stopping_patience: int = 10,
                device: str = 'cpu',
                save_path: str = 'data/model') -> Tuple[str, float]:
    """
    Train the Transformer-based model with a learning rate scheduler.

    Args:
        model (nn.Module): The initialized model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (optim.Optimizer): The optimizer for training.
        criterion (nn.Module): Loss function (e.g., SmoothL1Loss).
        scheduler (optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int): Maximum number of epochs for training.
        early_stopping_patience (int): Number of epochs with no improvement to allow before stopping.
        device (str): 'cuda' or 'cpu'.
        save_path (str): Directory to save the best model checkpoint.

    Returns:
        (str, float): (best_model_path, best_val_loss).
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    os.makedirs(save_path, exist_ok=True)
    best_model_path = os.path.join(save_path, "best_model.pth")

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

        # Step the scheduler
        scheduler.step()

        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}  |  Train Loss: {train_loss:.3e}  |  Val Loss: {val_loss:.3e}")

        # Early Stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Ensure the best model is saved
    if not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at final epoch to {best_model_path}")

    print(f"Training completed. Best validation loss: {best_val_loss:.3e}")
    return best_model_path, best_val_loss


def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: str) -> float:
    """
    Evaluate the model on a validation or test dataset.

    Args:
        model (nn.Module): Model to evaluate.
        data_loader (DataLoader): DataLoader for validation/test set.
        criterion (nn.Module): Loss function.
        device (str): 'cuda' or 'cpu'.

    Returns:
        float: The average loss across the dataset.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs_main=inputs)
            batch_loss = criterion(outputs.view_as(targets), targets)
            total_loss += batch_loss.item()

    return total_loss / len(data_loader)


def train_model_from_config(config: dict,
                            data_folder: str,
                            model_save_path: str,
                            device: torch.device) -> float:
    """
    Train the model using parameters from `config`, and evaluate on a held-out test set.

    Args:
        config (dict): Dictionary containing model and training hyperparameters.
        data_folder (str): Path to the normalized profile data folder.
        model_save_path (str): Directory to save the best model checkpoint.
        device (torch.device): Training device ('cpu' or 'cuda').

    Returns:
        float: The best validation loss achieved during training.
    """
    dataset = NormalizedProfilesDataset(
        data_folder=data_folder,
        expected_length=config["nlev"],
        input_variables=config["input_variables"],
        target_variables=config["target_variables"]
    )

    # Subset the dataset if needed
    dataset_size = int(len(dataset) * config.get("frac_of_training_data", 1.0))
    dataset = torch.utils.data.Subset(dataset, range(dataset_size))

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize the Transformer model
    model = AtmosphericModel(
        nx=len(config["input_variables"]),
        ny=len(config["target_variables"]),
        nneur=config.get("nneur", (32, 32)),
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        layer_norm_eps=config.get("layer_norm_eps", 1e-6),
        batch_first=config.get("batch_first", True),
        norm_first=config.get("norm_first", True),
        bias=config.get("bias", True),
        attention_dropout=config.get("attention_dropout", 0.0),
        ffn_activation=config.get("ffn_activation", "gelu"),
        pos_encoding=config.get("pos_encoding", "absolute"),
        layer_dropout=config.get("layer_dropout", 0.0),
        return_attn_weights=config.get("return_attn_weights", False),
        max_seq_length=config.get("max_seq_length", 512),
        output_proj=config.get("output_proj", True)
    ).to(device)

    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Train the model
    best_model_path, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_epochs=config["epochs"],
        early_stopping_patience=10,
        device=device.type,
        save_path=model_save_path
    )

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss = evaluate_model(model, test_loader, criterion, device.type)
    print(f"Test Loss: {test_loss:.3e}")

    return best_val_loss


def objective(trial, params: dict, data_folder: str, model_save_path: str, device: torch.device) -> float:
    """
    Objective function for Optuna hyperparameter tuning.

    Args:
        trial (optuna.Trial): The current trial object.
        params (dict): Dictionary containing default hyperparameters.
        data_folder (str): Path to the normalized profile data folder.
        model_save_path (str): Directory to save the best model checkpoint.
        device (torch.device): 'cuda' or 'cpu'.

    Returns:
        float: The validation loss after training for the current trial.
    """
    # Update hyperparameters from the search space
    params.update({
        "d_model": trial.suggest_categorical("d_model", [64, 128]),
        "nhead": trial.suggest_categorical("nhead", [4, 8]),
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 6),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [256, 512, 1024]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "layer_norm_eps": trial.suggest_loguniform("layer_norm_eps", 1e-6, 1e-4),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
    })

    val_loss = train_model_from_config(params, data_folder, model_save_path, device)
    return val_loss
