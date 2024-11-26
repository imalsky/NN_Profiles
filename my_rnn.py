# PyTorch core
import torch
import torch.nn as nn
import torch.optim as optim

# Data handling
import os
import json
from torch.utils.data import Dataset, DataLoader

# Numerical operations
import numpy as np

# System utilities
import sys


class MyRNN(nn.Module):
    def __init__(self, RNN_type='LSTM', nx = 4, nx_sfc=3, ny = 4, nneur=(64,64), 
        outputs_one_longer=False, # if True, inputs are a sequence of N but outputs are N+1 (e.g. predicting fluxes)
        concat=False):
        # Simple bidirectional RNN (Either LSTM or GRU) for predicting column 
        # outputs shaped either (B, L, Ny) or (B, L+1, Ny) from column inputs
        # (B, L, Nx) and optionally surface inputs (B, Nx_sfc) 
        # If surface inputs exist, they are used to initialize first (upward) RNN 
        # Assumes top-of-atmosphere is first in memory i.e. at index 0 
        # if it's not the flip operations need to be moved!
        super(MyRNN, self).__init__()
        self.nx = nx
        self.ny = ny 
        self.nx_sfc = nx_sfc 
        self.nneur = nneur 
        self.outputs_one_longer=outputs_one_longer
        if len(nneur) < 1 or len(nneur) > 3:
            sys.exit("Number of RNN layers and length of nneur should be 2 or 3")

        self.RNN_type=RNN_type
        if self.RNN_type=='LSTM':
            RNN_model = nn.LSTM
        elif self.RNN_type=='GRU':
            RNN_model = nn.GRU
        else:
            raise NotImplementedError()
                    
        self.concat=concat

        if self.nx_sfc > 0:
            self.mlp_surface1  = nn.Linear(nx_sfc, self.nneur[0])
            if self.RNN_type=="LSTM":
                self.mlp_surface2  = nn.Linear(nx_sfc, self.nneur[0])

        self.rnn1      = RNN_model(nx,            self.nneur[0], batch_first=True) # (input_size, hidden_size, num_layers=1
        self.rnn2      = RNN_model(self.nneur[0], self.nneur[1], batch_first=True)
        if len(self.nneur)==3:
            self.rnn3      = RNN_model(self.nneur[1], self.nneur[2], batch_first=True)

        # The final hidden variable is either the output from the last RNN, or
        # the  concatenated outputs from all RNNs
        if concat:
            nh_rnn = sum(nneur)
        else:
            nh_rnn = nneur[-1]

        self.mlp_output = nn.Linear(nh_rnn, self.ny)
        
            
    def forward(self, inputs_main, inputs_sfc=None):
        """
        Forward pass through the RNN.
        - inputs_main: Tensor of shape (B, L, nx), where B is batch size, L is sequence length, and nx is input size.
        - inputs_sfc: Optional surface input of shape (B, nx_sfc), where nx_sfc is the number of surface features.
        """
        if self.nx_sfc > 0:
            if inputs_sfc is None:
                raise ValueError(f"Expected surface inputs (inputs_sfc) for nx_sfc={self.nx_sfc}, but got None.")
            
            sfc1 = self.mlp_surface1(inputs_sfc)
            sfc1 = torch.tanh(sfc1)
            
            if self.RNN_type == "LSTM":
                sfc2 = self.mlp_surface2(inputs_sfc)
                sfc2 = torch.tanh(sfc2)
                hidden = (sfc1.view(1, -1, self.nneur[0]), sfc2.view(1, -1, self.nneur[0]))  # (h0, c0)
            else:
                hidden = sfc1.view(1, -1, self.nneur[0])
        else:
            hidden = None

        # Reverse input sequence (TOA first)
        inputs_main = torch.flip(inputs_main, [1])

        # RNN layers
        out, hidden = self.rnn1(inputs_main, hidden)
        out = torch.flip(out, [1])

        out2, hidden2 = self.rnn2(out)
        
        if len(self.nneur) == 3:
            rnn3_input = torch.flip(out2, [1])
            out3, hidden3 = self.rnn3(rnn3_input)
            out3 = torch.flip(out3, [1])
            rnnout = torch.cat((out3, out2, out), dim=2) if self.concat else out3
        else:
            rnnout = torch.cat((out2, out), dim=2) if self.concat else out2

        # Final output layer
        out = self.mlp_output(rnnout)
        return out


