import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Union, Tuple, Callable, Optional


class AtmosphericModel(nn.Module):
    """
    A Transformer-based model for 1D sequential data regression,
    optionally including a small MLP block after each encoder layer
    to improve representational capacity.
    """
    def __init__(
        self,
        nx: int,
        ny: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: Union[str, Callable] = 'gelu',
        layer_norm_eps: float = 1e-6,
        batch_first: bool = True,
        norm_first: bool = True,
        bias: bool = True,
        attention_dropout: float = 0.0,      # separate dropout for attention (if needed)
        ffn_activation: str = 'relu',       # feedforward activation name
        pos_encoding: str = 'absolute',     # 'absolute' or 'relative'
        layer_dropout: float = 0.0,         # extra dropout after each layer
        return_attn_weights: bool = False,  # set True to expose attention weights
        max_seq_length: int = 512,
        output_proj: bool = True,
    ):
        super().__init__()

        self.nx = nx
        self.ny = ny
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.activation = self._map_activation(activation)
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.bias = bias
        self.attention_dropout = attention_dropout
        self.ffn_activation = ffn_activation
        self.pos_encoding_type = pos_encoding
        self.layer_dropout = layer_dropout
        self.return_attn_weights = return_attn_weights
        self.max_seq_length = max_seq_length
        self.output_proj = output_proj

        # Optional input projection
        self.input_projection = nn.Linear(nx, d_model, bias=self.bias) if nx != d_model else None

        # Positional encoding
        self.pos_encoder = None
        if self.pos_encoding_type == 'absolute':
            self.pos_encoder = SinusoidalPositionalEncoding(
                d_model=self.d_model,
                max_len=self.max_seq_length,
                batch_first=self.batch_first
            )

        # Build the TransformerEncoder layers, each optionally followed by a residual MLP
        self.transformer_encoder = self._build_transformer_encoder()

        # Final output projection
        self.output_layer = nn.Linear(self.d_model, ny, bias=self.bias) if self.output_proj else None

    def _build_transformer_encoder(self) -> nn.Sequential:
        """
        Build a multi-layer Transformer encoder with optional residual MLP blocks.
        """
        layers = []
        for _ in range(self.num_encoder_layers):
            encoder_layer = TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout_p,
                activation=self.activation,
                layer_norm_eps=self.layer_norm_eps,
                batch_first=self.batch_first,
                norm_first=self.norm_first
            )
            layers.append(encoder_layer)

            # Optional dropout applied right after each layer
            if self.layer_dropout > 0:
                layers.append(nn.Dropout(self.layer_dropout))

            # A lightweight residual MLP block can improve model capacity:
            # This is optional, but can sometimes enhance performance
            # by giving the model additional representation power.
            # The feed-forward dimension is a fraction or multiple of d_model
            mlp_hidden = max(64, self.d_model // 2)
            layers.append(ResidualMLPBlock(input_dim=self.d_model, hidden_dim=mlp_hidden))

        return nn.Sequential(*layers)

    def forward(self, inputs_main: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          inputs_main: (batch, seq_len, nx) if batch_first=True
        Returns:
          (batch, seq_len, ny) if output_proj=True
        """
        x = inputs_main
        # Project input dimension
        if self.input_projection is not None:
            x = self.input_projection(x)

        # Apply positional encoding if available
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        # Pass through stacked TransformerEncoder (with optional MLPs)
        x = self.transformer_encoder(x)

        # Optional final projection to ny
        if self.output_layer is not None:
            x = self.output_layer(x)

        return x

    @staticmethod
    def _map_activation(activation: Union[str, Callable]) -> Callable:
        """Map string -> torch.nn.functional activation or return callable."""
        if isinstance(activation, str):
            act_map = {
                'relu': torch.nn.functional.relu,
                'gelu': torch.nn.functional.gelu,
                'selu': torch.nn.functional.selu,
                'leaky_relu': torch.nn.functional.leaky_relu,
            }
            return act_map.get(activation.lower(), torch.nn.functional.gelu)
        elif callable(activation):
            return activation
        else:
            return torch.nn.functional.gelu


class ResidualMLPBlock(nn.Module):
    """
    A lightweight residual MLP block to follow each TransformerEncoder layer.
    This can enhance model capacity and potentially improve accuracy.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, seq_len, d_model)
        """
        residual = x
        out = self.mlp(x)
        out = self.layer_norm(out + residual)
        return out


class SinusoidalPositionalEncoding(nn.Module):
    """
    Absolute sinusoidal positional encoding from "Attention is All You Need."
    """
    def __init__(self, d_model: int, max_len: int = 512, batch_first: bool = True):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.batch_first = batch_first

        # Create sinusoidal position encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor:
          x shape: (batch, seq_len, d_model) if batch_first=True
        """
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
            pos_emb = self.pe[:, :seq_len, :].expand(batch_size, -1, -1)
            x = x + pos_emb
        else:
            seq_len, batch_size, _ = x.size()
            pos_emb = self.pe[:, :seq_len, :].transpose(0, 1).expand(seq_len, batch_size, -1)
            x = x + pos_emb
        return x
