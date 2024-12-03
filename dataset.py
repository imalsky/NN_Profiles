# dataset.py

import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np


class NormalizedProfilesDataset(Dataset):
    def __init__(self, data_folder, expected_length=None, input_variables=None, target_variable='net_flux'):
        """
        Initialize the dataset.

        Parameters:
            data_folder (str): Path to the folder containing JSON profile files.
            expected_length (int, optional): Expected length of the profiles. If None, no length filtering is applied.
            input_variables (list of str, optional): List of variable names to be used as input features.
            target_variable (str): Name of the variable to be used as the target.
        """
        self.data_folder = data_folder
        self.expected_length = expected_length
        self.input_variables = input_variables or ['pressure', 'temperature']
        self.target_variable = target_variable
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

            # Check if all required variables are present
            required_keys = set(self.input_variables + [self.target_variable])
            if all(key in profile for key in required_keys):
                if self.expected_length is None or len(profile["pressure"]) == self.expected_length:
                    valid_files.append(file_path)
                else:
                    print(f"Skipping {file_path}: Incorrect profile length.")
            else:
                missing_keys = required_keys - profile.keys()
                print(f"Skipping {file_path}: Missing required keys {missing_keys}.")
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
                - targets (torch.Tensor): Tensor of shape (sequence_length,).
        """
        file_path = self.valid_files[idx]
        with open(file_path, "r") as f:
            profile = json.load(f)

        # Prepare input features
        features = []
        for var_name in self.input_variables:
            var_data = profile[var_name]
            if isinstance(var_data, list):
                var_array = np.array(var_data)
                if len(var_array) != self.expected_length:
                    raise ValueError(f"Variable '{var_name}' in profile '{file_path}' has length {len(var_array)}, expected {self.expected_length}")
            else:
                # If scalar, expand to match expected_length
                var_array = np.full(self.expected_length, var_data)
            features.append(var_array)

        # Stack features to create (sequence_length, num_features)
        inputs = np.stack(features, axis=1)

        # Get target variable
        target_data = profile[self.target_variable]
        if isinstance(target_data, list):
            target = np.array(target_data)
            if len(target) != self.expected_length:
                raise ValueError(f"Target variable '{self.target_variable}' in profile '{file_path}' has length {len(target)}, expected {self.expected_length}")
        else:
            target = np.full(self.expected_length, target_data)

        # Convert to torch tensors
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(target, dtype=torch.float32)

        return inputs, targets
