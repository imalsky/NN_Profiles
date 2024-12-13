# Default hyperparameters for AtmosphericModel
hyperparameters = {
    "model_architecture": {
        "nx": 4,                        # Number of input features
        "ny": 1,                        # Number of output features
        "nneur": (64, 64),              # Encoder hidden layer sizes
        "d_model": 128,                 # Embedding dimension for Transformer
        "nhead": 4,                     # Number of attention heads
        "num_encoder_layers": 4,        # Number of encoder layers
        "dim_feedforward": 512,         # Dimension of feedforward layers
        "dropout": 0.2,                 # Dropout rate
        "activation": 'gelu',           # Activation function ('gelu' or 'relu')
        "layer_norm_eps": 1e-6,         # Epsilon for layer normalization
        "batch_first": True,            # Input tensors are (batch, seq, feature)
        "norm_first": True,             # Apply normalization before attention/feedforward
        "bias": True                    # Use bias in Linear and LayerNorm layers
    },
    "training": {
        "epochs": 100,                  # Number of training epochs
        "batch_size": 8,                # Batch size for DataLoader
        "learning_rate": 1e-4,          # Initial learning rate
        "weight_decay": 1e-4,           # Weight decay for optimizer
        "early_stopping_patience": 10   # Patience for early stopping
    },
    "scheduler": {
        "scheduler_type": "OneCycleLR", # Learning rate scheduler type
        "max_lr": 1e-3,                 # Maximum learning rate for OneCycleLR
        "anneal_strategy": "cos",       # Annealing strategy ('cos' for cosine decay)
        "div_factor": 25.0,             # Max_lr / div_factor determines initial LR
        "final_div_factor": 1e4         # Determines final LR as max_lr / final_div_factor
    },
    "dataset": {
        "input_variables": ["pressure", "temperature", "flux_surface_down"],
        "target_variables": ["net_flux"],
        "frac_of_training_data": 1.0    # Fraction of data used for training
    }
}
