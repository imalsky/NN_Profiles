# dataset.py

import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np

class NormalizedProfilesDataset(Dataset):
    def __init__(self, data_folder, expected_length=None, include_Tstar=True):
        """
        Initialize the dataset.

        Parameters:
            data_folder (str): Path to the folder containing JSON profile files.
            expected_length (int, optional): Expected length of the profiles. If None, no length filtering is applied.
            include_Tstar (bool): Whether to include Tstar as an input feature.
        """
        self.data_folder = data_folder
        self.expected_length = expected_length
        self.include_Tstar = include_Tstar
        self.file_list = [
            os.path.join(data_folder, f)
            for f in os.listdir(data_folder)
            if f.endswith(".json") and f != "normalization_metadata.json"
        ]
        self.valid_files = self._filter_valid_files()

        if not self.valid_files:
            raise ValueError(f"No valid JSON profiles found in {data_folder}")

    def _filter_valid_files(self):
        """
        Filter out invalid JSON profiles that do not meet the required criteria.

        Returns:
            list: List of valid file paths.
        """
        valid_files = []
        for file_path in self.file_list:
            with open(file_path, "r") as f:
                profile = json.load(f)
            required_keys = ["pressure", "temperature", "net_flux", "Tstar"]
            if all(key in profile for key in required_keys):
                if self.expected_length is None or len(profile["temperature"]) == self.expected_length:
                    valid_files.append(file_path)
                else:
                    print(f"Skipping {file_path}: Incorrect profile length.")
            else:
                print(f"Skipping {file_path}: Missing one of the required keys {required_keys}.")
        return valid_files

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        """
        Retrieve the inputs and targets for a given index.

        Parameters:
            idx (int): Index of the data sample.

        Returns:
            tuple: (inputs, targets)
                - inputs (torch.Tensor): Tensor of shape (sequence_length, num_features).
                - targets (torch.Tensor): Tensor of shape (sequence_length, 1).
        """
        file_path = self.valid_files[idx]
        with open(file_path, "r") as f:
            profile = json.load(f)

        pressures = profile["pressure"]  # Always included
        temperatures = profile["temperature"]
        net_fluxes = profile["net_flux"]

        # Prepare input features
        features = [pressures, temperatures]  # Pressure and Temperature are always included

        if self.include_Tstar:
            Tstar = profile["Tstar"]
            Tstar_array = [Tstar] * len(pressures)  # Create an array of Tstar values
            features.append(Tstar_array)

        # Stack features to create (sequence_length, num_features)
        inputs = np.stack(features, axis=1)
        targets = np.array(net_fluxes)

        # Convert to torch tensors
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        # Reshape targets to (sequence_length, 1)
        targets = targets.view(-1, 1)

        return inputs, targets
