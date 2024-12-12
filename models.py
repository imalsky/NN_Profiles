import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Sequential, Conv1d, ReLU


class BasicRNN(nn.Module):
    def __init__(self, RNN_type='LSTM', nx=4, nx_sfc=0, ny=1, nneur=(32, 32), outputs_one_longer=False, concat=False):
        """
        Simple RNN (LSTM or GRU) model.

        Parameters:
            RNN_type (str): 'LSTM' or 'GRU'
            nx (int): number of input features per timestep
            nx_sfc (int): number of surface features (optional)
            ny (int): number of output features
            nneur (tuple): hidden sizes for each layer
            outputs_one_longer (bool): not used in this example
            concat (bool): whether to concatenate outputs from all RNN layers
        """
        super(BasicRNN, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nx_sfc = nx_sfc
        self.nneur = nneur
        self.outputs_one_longer = outputs_one_longer
        self.concat = concat

        # Optionally handle surface features if needed
        if self.nx_sfc > 0:
            self.mlp_surface = nn.Linear(nx_sfc, nneur[0])

        if RNN_type == 'LSTM':
            RNN_model = nn.LSTM
        elif RNN_type == 'GRU':
            RNN_model = nn.GRU
        else:
            raise NotImplementedError(f"Unsupported RNN type: {RNN_type}")

        # Create RNN layers
        self.rnn_layers = nn.ModuleList()
        input_size = nx
        for h in nneur:
            self.rnn_layers.append(RNN_model(input_size, h, batch_first=True))
            input_size = h

        # Determine output dimension after RNN layers
        if concat:
            nh_rnn = sum(nneur)
        else:
            nh_rnn = nneur[-1]

        self.mlp_output = nn.Linear(nh_rnn, ny)

    def forward(self, inputs_main, inputs_sfc=None):
        """
        Forward pass.

        Parameters:
            inputs_main (torch.Tensor): shape (batch, seq_len, nx)
            inputs_sfc (torch.Tensor or None): shape (batch, nx_sfc) if used
        """
        x = inputs_main
        # If needed, incorporate surface features:
        # For simplicity, we won't do anything with inputs_sfc here.

        outputs = []
        for rnn in self.rnn_layers:
            x, _ = rnn(x)
            outputs.append(x)

        if self.concat:
            x = torch.cat(outputs, dim=-1)
        else:
            x = outputs[-1]

        return self.mlp_output(x)


class RNN_New(nn.Module):
    def __init__(self,
                 RNN_type='LSTM',
                 nx=4,
                 ny=1,
                 nneur=(64, 64),
                 outputs_one_longer=False,
                 concat=False,
                 use_pressure_gating=False,
                 pressure_index=0,
                 pressure_threshold=0.1):
        """
        Flexible model with optional pressure gating, supporting LSTM, GRU, Transformer, and TCN.

        Parameters:
            RNN_type (str): 'LSTM', 'GRU', 'Transformer', 'TCN'
            nx (int): Number of input features
            ny (int): Number of output features
            nneur (tuple): Hidden sizes (for Transformer, nneur[0] = d_model)
            outputs_one_longer (bool): Not used here
            concat (bool): Whether to concatenate outputs
            use_pressure_gating (bool): If True, apply gating
            pressure_index (int): Index of pressure feature
            pressure_threshold (float): Threshold for gating
        """
        super(RNN_New, self).__init__()
        self.RNN_type = RNN_type
        self.nx = nx
        self.ny = ny
        self.nneur = nneur
        self.concat = concat
        self.use_pressure_gating = use_pressure_gating
        self.pressure_index = pressure_index
        self.pressure_threshold = pressure_threshold

        if self.RNN_type == 'LSTM':
            self.model = nn.ModuleList([
                nn.LSTM(input_size=nx if i == 0 else nneur[i - 1],
                        hidden_size=h,
                        batch_first=True)
                for i, h in enumerate(nneur)
            ])
            self.concat_output_dim = nneur[-1]

        elif self.RNN_type == 'GRU':
            self.model = nn.ModuleList([
                nn.GRU(input_size=nx if i == 0 else nneur[i - 1],
                       hidden_size=h,
                       batch_first=True)
                for i, h in enumerate(nneur)
            ])
            self.concat_output_dim = nneur[-1]

        elif self.RNN_type == 'Transformer':
            # For Transformer, d_model = nneur[0]
            # If nx != d_model, we need a projection layer
            self.d_model = nneur[0]
            if nx != self.d_model:
                self.input_projection = nn.Linear(nx, self.d_model)
            else:
                self.input_projection = None

            transformer_layer = TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1
            )
            self.model = TransformerEncoder(transformer_layer, num_layers=len(nneur))
            self.concat_output_dim = self.d_model

        elif self.RNN_type == 'TCN':
            layers = []
            in_channels = nx
            for out_channels in nneur:
                layers.append(Conv1d(in_channels, out_channels, kernel_size=3, dilation=2, padding=2))
                layers.append(ReLU())
                in_channels = out_channels
            self.model = Sequential(*layers)
            self.concat_output_dim = nneur[-1]

        else:
            raise NotImplementedError(f"Unsupported RNN type: {self.RNN_type}")

        self.output_layer = nn.Linear(self.concat_output_dim, ny)

    def forward(self, inputs_main, inputs_sfc=None):
        """
        Forward pass with optional pressure gating.

        Parameters:
            inputs_main (torch.Tensor): shape (batch, seq_len, nx)
            inputs_sfc (torch.Tensor or None): optional surface inputs
        """
        x = inputs_main

        # Apply pressure gating if enabled
        if self.use_pressure_gating:
            pressure = x[:, :, self.pressure_index]  # (batch, seq_len)
            pressure_diff = torch.abs(pressure[:, :, None] - pressure[:, None, :])  # (batch, seq_len, seq_len)
            mask = torch.sigmoid(-pressure_diff / self.pressure_threshold)
            x = torch.einsum('bij,bjk->bik', mask, x)

        if self.RNN_type in ['LSTM', 'GRU']:
            outputs = []
            for rnn in self.model:
                x, _ = rnn(x)
                outputs.append(x)
            if self.concat:
                x = torch.cat(outputs, dim=-1)
            else:
                x = outputs[-1]

        elif self.RNN_type == 'Transformer':
            # Project inputs if needed
            if self.input_projection is not None:
                x = self.input_projection(x)  # Now x is (batch, seq_len, d_model)

            # Transformer expects (seq_len, batch, d_model)
            x = x.transpose(0, 1)
            x = self.model(x)
            x = x.transpose(0, 1)  # back to (batch, seq_len, d_model)

        elif self.RNN_type == 'TCN':
            # TCN expects (batch, nx, seq_len)
            x = x.permute(0, 2, 1)
            x = self.model(x)
            x = x.permute(0, 2, 1)

        return self.output_layer(x)
