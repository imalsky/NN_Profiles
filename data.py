# data.py

import json
import torch
from torch.utils.data import Dataset
import os
import glob

class FluxDataset(Dataset):
    def __init__(self, data_folder):
        self.inputs = []
        self.targets = []

        # Load all JSON files matching prof_*.json
        json_files = glob.glob(os.path.join(data_folder, 'prof_*.json'))

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract data
            pressure = torch.tensor(data['pressure'], dtype=torch.float32)
            temperature = torch.tensor(data['temperature'], dtype=torch.float32)
            net_fluxes = torch.tensor(data['net_fluxes'], dtype=torch.float32)
            Tstar = torch.tensor(data['Tstar'], dtype=torch.float32)

            # Stack input features (pressure and temperature)
            # Inputs shape: [sequence_length, feature_size]
            inputs = torch.stack((pressure, temperature), dim=1)

            # For each profile, store inputs, targets, and Tstar
            self.inputs.append((inputs, Tstar))
            self.targets.append(net_fluxes)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Return inputs, targets, and Tstar for each profile
        inputs, Tstar = self.inputs[idx]
        targets = self.targets[idx]
        return inputs, targets, Tstar
