import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np


class NormalizedProfilesDataset(Dataset):
    def __init__(self, data_folder, expected_length=None, input_variables=None, target_variables=None):
        """
        Initialize the dataset.

        Parameters:
            data_folder (str): Path to the folder containing JSON profile files.
            expected_length (int, optional): Expected length of the profiles. If None, no length filtering is applied.
            input_variables (list of str): List of input variable names.
            target_variables (list of str): List of target variable names.
        """
        self.data_folder = data_folder
        self.expected_length = expected_length
        self.input_variables = input_variables or ['pressure', 'temperature']
        self.target_variables = target_variables or ['heating_rate']

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
            required_keys = set(self.input_variables + self.target_variables)
            if all(key in profile for key in required_keys):
                # Validate the length of sequence data
                if self.expected_length is None or len(profile[self.input_variables[0]]) == self.expected_length:
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
                - targets (torch.Tensor): Tensor of shape (sequence_length, num_target_features).
        """
        file_path = self.valid_files[idx]
        with open(file_path, "r") as f:
            profile = json.load(f)

        # Prepare input features
        inputs = []
        for var in self.input_variables:
            if isinstance(profile[var], list):
                inputs.append(profile[var])
            else:
                # Handle scalar values by expanding them to match sequence length
                inputs.append([profile[var]] * self.expected_length)

        inputs = np.stack(inputs, axis=1)

        # Prepare target features
        targets = []
        for var in self.target_variables:
            if isinstance(profile[var], list):
                targets.append(profile[var])
            else:
                # Handle scalar values by expanding them to match sequence length
                targets.append([profile[var]] * self.expected_length)

        targets = np.stack(targets, axis=1)

        # Convert to torch tensors
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        return inputs, targets
