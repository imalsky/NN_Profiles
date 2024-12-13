import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class AtmosphericModel(nn.Module):
    def __init__(self,
                 nx=4,
                 ny=1,
                 nneur=(128, 128),
                 d_model=128,
                 nhead=4,
                 num_encoder_layers=4,
                 dim_feedforward=512,
                 dropout=0.2,
                 activation='gelu',
                 layer_norm_eps=1e-6,
                 batch_first=True,
                 norm_first=True,
                 bias=True):
        """
        Streamlined Transformer-based model for sequential data.

        Parameters:
            nx (int): Number of input features
            ny (int): Number of output features
            nneur (tuple): Encoder hidden layer sizes
            d_model (int): Number of expected features in the encoder inputs
            nhead (int): Number of heads in the multiheadattention models
            num_encoder_layers (int): Number of sub-encoder-layers in the encoder
            dim_feedforward (int): Dimension of the feedforward network model
            dropout (float): Dropout value
            activation (str or callable): Activation function of encoder intermediate layer
            layer_norm_eps (float): Epsilon value in layer normalization components
            batch_first (bool): If True, input and output tensors are (batch, seq, feature)
            norm_first (bool): If True, perform LayerNorms before other attention/feedforward ops
            bias (bool): If False, Linear and LayerNorm layers will not learn an additive bias
        """
        super(AtmosphericModel, self).__init__()

        self.nx = nx
        self.ny = ny
        self.nneur = nneur  # Added nneur as a parameter
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

        # Add an input projection layer if nx != d_model
        self.input_projection = nn.Linear(self.nx, self.d_model) if self.nx != self.d_model else None

        # Define a TransformerEncoderLayer
        transformer_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=self.batch_first,
            norm_first=self.norm_first
        )

        # Initialize the TransformerEncoder
        self.model = TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=self.num_encoder_layers
        )

        self.output_layer = nn.Linear(self.d_model, ny)

    def forward(self, inputs_main):
        """
        Forward pass for the Transformer-based model.

        Parameters:
            inputs_main (torch.Tensor): shape (batch, seq_len, nx)
        """
        x = inputs_main

        # Apply input projection if necessary
        if self.input_projection:
            x = self.input_projection(x)

        # Process with Transformer
        x = self.model(x)

        # Final output layer
        return self.output_layer(x)
