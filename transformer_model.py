import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class AtmosphericModel(nn.Module):
    def __init__(self,
                 nx,
                 ny,
                 nneur=(32, 32),
                 d_model=64,
                 nhead=4,
                 num_encoder_layers=4,
                 dim_feedforward=256,
                 dropout=0.1,
                 activation='gelu',
                 layer_norm_eps=1e-6,
                 batch_first=True,
                 norm_first=True,
                 bias=True,
                 attention_dropout=0.0,
                 ffn_activation='relu',
                 pos_encoding='absolute',
                 layer_dropout=0.0,
                 return_attn_weights=False,
                 max_seq_length=512,
                 output_proj=True):
        """
        Transformer-based model for sequential data regression.

        Parameters:
            nx (int): Number of input features.
            ny (int): Number of output features.
            nneur (tuple): Encoder hidden layer sizes.
            d_model (int): Number of expected features in the encoder inputs.
            nhead (int): Number of heads in the multiheadattention models.
            num_encoder_layers (int): Number of sub-encoder-layers in the encoder.
            dim_feedforward (int): Dimension of the feedforward network model.
            dropout (float): Dropout value for the encoder.
            activation (str or callable): Activation function of encoder intermediate layer.
            layer_norm_eps (float): Epsilon value in layer normalization components.
            batch_first (bool): If True, input/output tensors are (batch, seq, feature).
            norm_first (bool): If True, perform LayerNorms before other attention/feedforward ops.
            bias (bool): If False, Linear and LayerNorm layers will not learn an additive bias.
            attention_dropout (float): Dropout specific to the attention mechanism.
            ffn_activation (str): Feedforward activation ('relu', 'gelu', etc.).
            pos_encoding (str): Type of positional encoding ('absolute' or 'relative').
            layer_dropout (float): Dropout applied to transformer layers (layer-wise dropout).
            return_attn_weights (bool): If True, returns attention weights for interpretability.
            max_seq_length (int): Maximum sequence length for positional encoding.
            output_proj (bool): If True, applies an output projection layer.
        """
        super(AtmosphericModel, self).__init__()

        self.nx = nx
        self.ny = ny
        self.nneur = nneur
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = torch.nn.functional.relu if activation == 'relu' else torch.nn.functional.gelu
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.bias = bias
        self.attention_dropout = attention_dropout
        self.ffn_activation = ffn_activation
        self.pos_encoding = pos_encoding
        self.layer_dropout = layer_dropout
        self.return_attn_weights = return_attn_weights
        self.max_seq_length = max_seq_length
        self.output_proj = output_proj

        # Add an input projection layer if nx != d_model
        self.input_projection = nn.Linear(self.nx, self.d_model) if self.nx != self.d_model else None

        # Define a TransformerEncoderLayer with all options explicitly stated
        transformer_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,  # Applies to feedforward and multi-head attention
            activation=self.activation,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=self.batch_first,
            norm_first=self.norm_first  # Whether LayerNorm is applied before or after operations
        )

        # Initialize the TransformerEncoder
        self.model = TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=self.num_encoder_layers
        )

        # Optional output projection
        if self.output_proj:
            self.output_layer = nn.Linear(self.d_model, ny)
        else:
            self.output_layer = None

    def forward(self, inputs_main):
        """
        Forward pass for the Transformer-based model.

        Parameters:
            inputs_main (torch.Tensor): shape (batch, seq_len, nx).
        """
        x = inputs_main

        # Apply input projection if necessary
        if self.input_projection:
            x = self.input_projection(x)

        # Process with Transformer
        x = self.model(x)

        # Final output layer (if output_proj is enabled)
        if self.output_proj:
            x = self.output_layer(x)

        return x
