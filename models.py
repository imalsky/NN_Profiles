# models.py

import sys
import torch
import torch.nn as nn


class BasicRNN(nn.Module):
    def __init__(self, RNN_type='LSTM', nx=4, nx_sfc=0, ny=1, nneur=(32, 32), outputs_one_longer=False, concat=False):
        """
        Simple bidirectional RNN (Either LSTM or GRU)

        Parameters:
            RNN_type (str): Type of RNN ('LSTM' or 'GRU').
            nx (int): Number of input features.
            nx_sfc (int): Number of surface input features (set to 0 if not used).
            ny (int): Number of output features.
            nneur (tuple): Number of neurons in each RNN layer.
            outputs_one_longer (bool): If True, the output sequence is one timestep longer than input.
            concat (bool): If True, concatenate outputs from all RNN layers.
        """
        super(BasicRNN, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nx_sfc = nx_sfc
        self.nneur = nneur
        self.outputs_one_longer = outputs_one_longer
        if len(nneur) < 1 or len(nneur) > 3:
            sys.exit(
                "Number of RNN layers and length of nneur should be between 1 and 3")

        self.RNN_type = RNN_type
        if self.RNN_type == 'LSTM':
            RNN_model = nn.LSTM
        elif self.RNN_type == 'GRU':
            RNN_model = nn.GRU
        else:
            raise NotImplementedError(f"Unsupported RNN type: {self.RNN_type}")

        self.concat = concat

        if self.nx_sfc > 0:
            self.mlp_surface1 = nn.Linear(nx_sfc, self.nneur[0])
            if self.RNN_type == "LSTM":
                self.mlp_surface2 = nn.Linear(nx_sfc, self.nneur[0])

        self.rnn1 = RNN_model(nx, self.nneur[0], batch_first=True)
        self.rnn2 = RNN_model(self.nneur[0], self.nneur[1], batch_first=True)
        if len(self.nneur) == 3:
            self.rnn3 = RNN_model(
                self.nneur[1], self.nneur[2], batch_first=True)

        if concat:
            nh_rnn = sum(nneur)
        else:
            nh_rnn = nneur[-1]

        self.mlp_output = nn.Linear(nh_rnn, self.ny)

    def forward(self, inputs_main, inputs_sfc=None):
        if self.nx_sfc > 0:
            if inputs_sfc is None:
                raise ValueError(f"Expected surface inputs (inputs_sfc) for nx_sfc={self.nx_sfc}, but got None.")

            sfc1 = torch.tanh(self.mlp_surface1(inputs_sfc))
            if self.RNN_type == "LSTM":
                sfc2 = torch.tanh(self.mlp_surface2(inputs_sfc))
                hidden = (sfc1.unsqueeze(0), sfc2.unsqueeze(0))
            else:
                hidden = sfc1.unsqueeze(0)
        else:
            hidden = None

        # Reverse input sequence (if needed)
        inputs_main = torch.flip(inputs_main, [1])

        # RNN layers
        out, hidden = self.rnn1(inputs_main, hidden)
        out = torch.flip(out, [1])

        out2, hidden2 = self.rnn2(out)

        if len(self.nneur) == 3:
            rnn3_input = torch.flip(out2, [1])
            out3, hidden3 = self.rnn3(rnn3_input)
            out3 = torch.flip(out3, [1])
            rnnout = torch.cat((out3, out2, out),
                               dim=2) if self.concat else out3
        else:
            rnnout = torch.cat((out2, out), dim=2) if self.concat else out2

        # Final output layer
        out = self.mlp_output(rnnout)
        return out















class RNN_New(nn.Module):
    def __init__(self,
                 RNN_type='LSTM',
                 nx=4,
                 ny=1,
                 nneur=(64, 64),
                 outputs_one_longer=False,
                 concat=False,
                 pressure_threshold=0.05,
                 pressure_index=0):
        """
        Initialize the RNN_New model with Layer Normalization and log-pressure-based gating.

        Parameters:
            RNN_type (str): Type of RNN ('LSTM' or 'GRU').
            nx (int): Number of input features.
            ny (int): Number of output features.
            nneur (tuple): Number of neurons in each RNN layer.
            outputs_one_longer (bool): If True, the output sequence is one timestep longer than input.
            concat (bool): If True, concatenate outputs from all RNN layers.
            pressure_threshold (float): Threshold for gating based on log-pressure differences.
            pressure_index (int): Index of the pressure feature in the input variables.
        """
        super(RNN_New, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nneur = nneur
        self.outputs_one_longer = outputs_one_longer
        self.concat = concat
        self.pressure_threshold = pressure_threshold
        self.pressure_index = pressure_index

        # Validate number of layers
        if len(nneur) < 1 or len(nneur) > 3:
            raise ValueError(
                "Number of RNN layers and length of nneur should be between 1 and 3")

        # Select RNN model
        if RNN_type == 'LSTM':
            RNN_model = nn.LSTM
        elif RNN_type == 'GRU':
            RNN_model = nn.GRU
        else:
            raise NotImplementedError(f"Unsupported RNN type: {RNN_type}")

        # Initialize RNN layers
        self.rnn_layers = nn.ModuleList()
        for i, hidden_size in enumerate(nneur):
            input_size = nx if i == 0 else nneur[i-1]*2  # 2 for bidirectional
            self.rnn_layers.append(
                RNN_model(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                )
            )

        # Initialize Layer Normalization layers
        self.layer_norms = nn.ModuleList([
            # 2 for bidirectional
            nn.LayerNorm(hidden_size * 2) for hidden_size in nneur
        ])

        # Output MLP
        if self.concat:
            # Sum over all bidirectional layers
            nh_rnn = sum(n * 2 for n in nneur)
        else:
            nh_rnn = nneur[-1] * 2  # Last layer's output size

        self.mlp_output = nn.Linear(nh_rnn, self.ny)

    def forward(self, inputs_main):
        """
        Forward pass of the RNN_New model.

        Parameters:
            inputs_main (torch.Tensor): Main input tensor of shape (batch_size, sequence_length, nx).

        Returns:
            torch.Tensor: Predicted outputs of shape (batch_size, sequence_length, ny).
        """
        batch_size, sequence_length, _ = inputs_main.shape

        # Check that pressure_index is valid
        if self.pressure_index >= self.nx:
            raise ValueError(f"pressure_index ({self.pressure_index}) is out of bounds for input size nx={self.nx}")

        # Extract pressure values using pressure_index
        pressure = inputs_main[:, :, self.pressure_index]

        # Compute absolute pressure differences
        # (batch_size, seq_len, seq_len)
        pressure_diff = torch.abs(
            pressure[:, :, None] - pressure[:, None, :])

        # Compute gating mask based on pressure differences
        # (batch_size, seq_len, seq_len)
        mask = torch.sigmoid(-pressure_diff / self.pressure_threshold)

        # Aggregate gated inputs
        # (batch_size, seq_len, nx)
        gated_inputs = torch.einsum('bij,bjk->bik', mask, inputs_main)

        rnn_out = gated_inputs

        # Forward pass through RNN layers
        for i, rnn in enumerate(self.rnn_layers):
            out, _ = rnn(rnn_out)  # Forward pass through RNN layer
            out = self.layer_norms[i](out)  # Apply Layer Normalization
            rnn_out = out  # Update input for next layer

        # Concatenate outputs from all RNN layers if concat=True
        if self.concat:
            combined_out = torch.cat(
                [layer_out for layer_out in rnn_out], dim=2)
        else:
            combined_out = rnn_out  # Use output from the last RNN layer

        # Pass through output MLP to get predictions
        outputs = self.mlp_output(combined_out)
        return outputs