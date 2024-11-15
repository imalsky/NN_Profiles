# model.py

import torch
import torch.nn as nn

class FluxPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, dropout=0.0):
        super(FluxPredictor, self).__init__()
        # LSTM to process pressure and temperature sequences
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Fully connected layers to combine LSTM output with Tstar
        self.fc1 = nn.Linear(hidden_size * 2 + 1, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x, Tstar):
        # x shape: (batch_size=1, seq_length, input_size=2)
        out, _ = self.rnn(x)  # out shape: (batch_size=1, seq_length, hidden_size*2)

        # Repeat Tstar to match sequence length
        Tstar = Tstar.unsqueeze(0).unsqueeze(2)  # Shape: (batch_size=1, 1, 1)
        Tstar = Tstar.expand(-1, x.size(1), -1)  # Shape: (batch_size=1, seq_length, 1)

        # Concatenate LSTM output with Tstar
        out = torch.cat((out, Tstar), dim=2)  # Shape: (batch_size=1, seq_length, hidden_size*2 + 1)

        # Pass through fully connected layers
        out = self.relu(self.fc1(out))
        out = self.fc2(out)  # Shape: (batch_size=1, seq_length, 1)

        out = out.squeeze()  # Remove unnecessary dimensions
        return out
