import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
                 concat=False,
                 use_pressure_gating=False,
                 pressure_index=0,
                 pressure_threshold=0.1,
                 nhead=4,
                 dim_feedforward=None,
                 dropout=0.1,
                 activation=None,
                 layer_norm_eps=1e-5,
                 norm_first=False,
                 **kwargs):
        """
        Flexible model supporting LSTM, GRU, Transformer, and TCN with optional pressure gating.

        Parameters:
            RNN_type (str): 'LSTM', 'GRU', 'Transformer', 'TCN'
            nx (int): Number of input features
            ny (int): Number of output features
            nneur (tuple): Hidden sizes (for Transformer, nneur[0] = d_model)
            concat (bool): Whether to concatenate outputs
            use_pressure_gating (bool): If True, apply pressure gating
            pressure_index (int): Index of pressure feature
            pressure_threshold (float): Threshold for gating
            nhead (int): Number of attention heads for Transformer
            dim_feedforward (int): Feedforward dimension for Transformer
            dropout (float): Dropout rate
            activation (callable): Activation function for Transformer
            layer_norm_eps (float): Epsilon for layer normalization
            norm_first (bool): Whether to apply layer normalization first in Transformer
            kwargs: Additional arguments for flexibility (e.g., for custom configurations).
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

        # Transformer-specific parameters
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * nneur[0]
        self.dropout = dropout
        self.activation = activation if activation is not None else torch.nn.functional.relu
        self.layer_norm_eps = layer_norm_eps
        self.norm_first = norm_first

        # Initialize model
        self.model = self._initialize_model()
        self.output_layer = nn.Linear(self.concat_output_dim, ny)

    def _initialize_model(self):
        """Initialize the main model based on the specified RNN type."""
        if self.RNN_type in ['LSTM', 'GRU']:
            rnn_class = nn.LSTM if self.RNN_type == 'LSTM' else nn.GRU
            rnn_layers = [
                rnn_class(
                    input_size=self.nx if i == 0 else self.nneur[i - 1],
                    hidden_size=h,
                    batch_first=True
                )
                for i, h in enumerate(self.nneur)
            ]
            self.concat_output_dim = sum(self.nneur) if self.concat else self.nneur[-1]
            return nn.ModuleList(rnn_layers)

        elif self.RNN_type == 'Transformer':
            # Set d_model from the first element of nneur
            self.d_model = self.nneur[0]
            # Add an input projection layer if nx != d_model
            self.input_projection = nn.Linear(self.nx, self.d_model) if self.nx != self.d_model else None

            # Define a TransformerEncoderLayer with flexible parameters
            transformer_layer = TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.activation,
                layer_norm_eps=self.layer_norm_eps,
                batch_first=True,
                norm_first=self.norm_first
            )

            # Initialize the TransformerEncoder
            self.concat_output_dim = self.d_model
            return TransformerEncoder(transformer_layer, num_layers=len(self.nneur))

        elif self.RNN_type == 'TCN':
            layers = []
            in_channels = self.nx
            for out_channels in self.nneur:
                layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2))
                layers.append(nn.ReLU())
                in_channels = out_channels
            self.concat_output_dim = self.nneur[-1]
            return nn.Sequential(*layers)

        else:
            raise NotImplementedError(f"Unsupported RNN type: {self.RNN_type}")

    def _apply_pressure_gating(self, x):
        """Apply pressure gating if enabled."""
        pressure = x[:, :, self.pressure_index]  # (batch, seq_len)
        pressure_diff = torch.abs(pressure[:, :, None] - pressure[:, None, :])  # (batch, seq_len, seq_len)
        mask = torch.sigmoid(-pressure_diff / self.pressure_threshold)
        return torch.einsum('bij,bjk->bik', mask, x)

    def forward(self, inputs_main, inputs_sfc=None):
        """
        Forward pass with optional pressure gating.

        Parameters:
            inputs_main (torch.Tensor): shape (batch, seq_len, nx)
            inputs_sfc (torch.Tensor or None): Optional surface inputs.
        """
        x = inputs_main

        # Apply pressure gating if enabled
        if self.use_pressure_gating:
            x = self._apply_pressure_gating(x)

        # Process with RNN, Transformer, or TCN
        if self.RNN_type in ['LSTM', 'GRU']:
            outputs = []
            for rnn in self.model:
                x, _ = rnn(x)
                outputs.append(x)
            x = torch.cat(outputs, dim=-1) if self.concat else outputs[-1]

        elif self.RNN_type == 'Transformer':
            if self.input_projection:
                x = self.input_projection(x)
            x = self.model(x)

        elif self.RNN_type == 'TCN':
            x = x.permute(0, 2, 1)  # TCN expects (batch, nx, seq_len)
            x = self.model(x)
            x = x.permute(0, 2, 1)  # Restore to (batch, seq_len, nx)

        return self.output_layer(x)
