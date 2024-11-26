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
    def __init__(self, RNN_type='LSTM', nx=4, nx_sfc=3, ny=4, nneur=(64, 64), outputs_one_longer=False, concat=False, dropout=0.2):
        super(MyRNN, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nneur = nneur
        self.outputs_one_longer = outputs_one_longer  # Add this back
        self.dropout = dropout

        if RNN_type == 'LSTM':
            RNN_model = nn.LSTM
        elif RNN_type == 'GRU':
            RNN_model = nn.GRU
        else:
            raise NotImplementedError()

        self.concat = concat

        if nx_sfc > 0:
            self.mlp_surface1 = nn.Linear(nx_sfc, nneur[0])
            if RNN_type == "LSTM":
                self.mlp_surface2 = nn.Linear(nx_sfc, nneur[0])

        # RNN Layers
        self.rnn1 = RNN_model(nx, nneur[0], batch_first=True)
        self.rnn2 = RNN_model(nneur[0], nneur[1], batch_first=True)
        if len(nneur) == 3:
            self.rnn3 = RNN_model(nneur[1], nneur[2], batch_first=True)

        self.dropout_layer = nn.Dropout(p=dropout)

        nh_rnn = sum(nneur) if concat else nneur[-1]
        self.mlp_output = nn.Sequential(
            nn.Linear(nh_rnn, ny),
            nn.Dropout(p=dropout)
        )

            
    def forward(self, inputs_main, inputs_sfc=None):
        """
        Forward pass of the RNN.
        Parameters:
        - inputs_main: Tensor of shape (B, L, nx), where B = batch size, L = number of layers, nx = input features (pressure, temperature).
        - inputs_sfc: Tensor of shape (B, nx_sfc), where nx_sfc = number of surface inputs (optional).
        
        Returns:
        - out: Tensor of shape (B, L, ny), where ny = output features (net flux).
        """ 
        # batch_size = inputs_main.shape[0]
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
        if inputs_sfc is not None:
            sfc1 = self.mlp_surface1(inputs_sfc)
            sfc1 = nn.Tanh()(sfc1)
            
            if self.RNN_type=="LSTM":
                sfc2 = self.mlp_surface2(inputs_sfc)
                sfc2 = nn.Tanh()(sfc2)
                hidden = (sfc1.view(1,-1,self.nneur[0]), sfc2.view(1,-1,self.nneur[0])) # (h0, c0)
            else:
                hidden = (sfc1.view(1,-1,self.nneur[0]))
        else:
            hidden = None

        # print(f'Using state1 {hidden}')
        # TOA is first in memory, so we need to flip the axis
        inputs_main = torch.flip(inputs_main, [1])
      
        out, hidden = self.rnn1(inputs_main, hidden)
        
        if self.outputs_one_longer:
            out = torch.cat((sfc1, out),axis=1)

        out = torch.flip(out, [1]) # the surface was processed first, but for
        # the second RNN (and the final output) we want TOA first
        
        out2, hidden2 = self.rnn2(out) 
        
        if len(self.nneur)==3:
            rnn3_input = torch.flip(out2, [1])
            
            out3, hidden3 = self.rnn3(rnn3_input) 
            
            out3 = torch.flip(out3, [1])
            
            if self.concat:
                rnnout = torch.cat((out3, out2, out),axis=2)
            else:
                rnnout = out3
        else:
            if self.concat:
                rnnout = torch.cat((out2, out),axis=2)
            else:
                rnnout = out2
                
        out = self.mlp_output(rnnout)
            
        return out 