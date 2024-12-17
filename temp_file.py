import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam



class AtmosphericTransformer(nn.Module):
    def __init__(self,
                 seq_input_dim,   # Number of features in the sequence data (e.g., temperature, pressure)
                 scalar_input_dim,  # Number of scalar features (e.g., stellar temperature, flux)
                 seq_length,       # Dynamically set the sequence length
                 hidden_dim=128,   # Transformer hidden dimension
                 num_heads=4,      # Number of attention heads
                 num_layers=3,     # Number of transformer layers
                 dropout=0.1):     # Dropout rate
        super(AtmosphericTransformer, self).__init__()

        # Embedding layers for sequence inputs
        self.seq_embedding = nn.Linear(seq_input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        # Projection layer for scalar inputs
        self.scalar_projection = nn.Linear(scalar_input_dim, hidden_dim)

        # Transformer Encoder
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Attention weights for scalar inputs
        self.scalar_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # MLP head for final predictions
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, seq_length)  # Output matches input sequence length
        )

    def forward(self, seq_data, scalar_data):
        """
        Args:
            seq_data (torch.Tensor): Shape (batch_size, N, seq_input_dim), N is the number of atmospheric layers.
            scalar_data (torch.Tensor): Shape (batch_size, scalar_input_dim), single-value inputs.

        Returns:
            torch.Tensor: Predicted output of shape (batch_size, N).
        """
        # Embed sequence data and apply positional encoding
        seq_embedded = self.seq_embedding(seq_data)  # (batch_size, N, hidden_dim)
        seq_embedded = self.positional_encoding(seq_embedded)

        # Project scalar data and repeat to match sequence length
        scalar_embedded = self.scalar_projection(scalar_data)  # (batch_size, hidden_dim)
        scalar_embedded = scalar_embedded.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        scalar_embedded_repeated = scalar_embedded.expand(-1, seq_embedded.size(1), -1)

        # Integrate scalar information into sequence using attention
        combined_embedded, _ = self.scalar_attention(
            query=seq_embedded,
            key=scalar_embedded_repeated,
            value=scalar_embedded_repeated
        )

        # Pass through Transformer layers with residual connections
        transformer_output = combined_embedded
        for layer in self.encoder_layers:
            transformer_output = self.layer_norm(transformer_output + layer(transformer_output))

        # Use the sequence output for predictions
        output = self.mlp(transformer_output.mean(dim=1))  # (batch_size, seq_length)

        return output


class PositionalEncoding(nn.Module):
    """Applies sinusoidal positional encoding to sequence data."""
    def __init__(self, hidden_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # (1, max_len, hidden_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, N, hidden_dim).

        Returns:
            torch.Tensor: Positional-encoded sequence of shape (batch_size, N, hidden_dim).
        """
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)





# Custom Dataset
class AtmosphericDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder

        # Filter out normalization_metadata.json
        self.files = [
            f for f in os.listdir(data_folder)
            if f.endswith('.json') and f != "normalization_metadata.json"
        ]

        self.data = []
        self.targets = []

        for file in self.files:
            with open(os.path.join(data_folder, file), 'r') as f:
                sample = json.load(f)

                # Ensure pressure and temperature are lists
                pressure = sample["pressure"]
                temperature = sample["temperature"]
                if isinstance(pressure, dict):
                    pressure = list(pressure.values())
                if isinstance(temperature, dict):
                    temperature = list(temperature.values())

                # Convert to PyTorch tensors
                seq_data = torch.tensor([pressure, temperature]).T.float()
                scalar_data = torch.tensor([
                    sample["Tstar"],
                    sample["orbital_sep"],
                    sample["flux_surface_down"]
                ]).float()
                target = torch.tensor(sample["net_flux"]).float()

                self.data.append((seq_data, scalar_data))
                self.targets.append(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# model Training Function
def train_model(model, train_loader, criterion, optimizer, epochs=20, device='cpu'):
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for (seq_data, scalar_data), targets in train_loader:
            seq_data = seq_data.to(device)  # (batch_size, N, seq_input_dim)
            scalar_data = scalar_data.to(device)  # (batch_size, scalar_input_dim)
            targets = targets.to(device)  # (batch_size,)

            optimizer.zero_grad()
            predictions = model(seq_data, scalar_data).squeeze(-1)  # Forward pass
            loss = criterion(predictions, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4e}")


# model Evaluation Function
def evaluate_model(model, val_loader, criterion, device='cpu'):
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for (seq_data, scalar_data), targets in val_loader:
            seq_data = seq_data.to(device)
            scalar_data = scalar_data.to(device)
            targets = targets.to(device)

            predictions = model(seq_data, scalar_data).squeeze(-1)
            loss = criterion(predictions, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss











# Main Script
if __name__ == "__main__":
    # Configurations
    data_folder = "data/normalize_profiles"
    batch_size = 32
    hidden_dim = 128
    num_heads = 4
    num_layers = 3
    output_dim = 100
    learning_rate = 0.001
    epochs = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Dataset
    dataset = AtmosphericDataset(data_folder)

    # Split Dataset into Train and Validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Dynamically calculate sequence length
    seq_length = len(dataset[0][0][0])  # Length of pressure or temperature array

    # Initialize the model with dynamic output dimension
    model = AtmosphericTransformer(
        seq_input_dim=2,  # Pressure and temperature
        scalar_input_dim=3,  # Tstar, orbital_sep, flux_surface_down
        seq_length=seq_length,  # Match the input sequence length
        hidden_dim=128,
        num_heads=4,
        num_layers=3
    )

    # Define Loss and Optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Train model
    print("Starting Training...")
    train_model(model, train_loader, criterion, optimizer, epochs=epochs, device=device)

    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, val_loader, criterion, device=device)

    # Save the model
    model_save_path = "atmospheric_transformer.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"model saved to {model_save_path}")
