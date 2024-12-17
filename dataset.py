import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Optional


class NormalizedProfilesDataset(Dataset):
    """
    PyTorch Dataset to load and process normalized atmospheric profile data from JSON files.

    Args:
        data_folder (str): Path to the folder containing JSON profile files.
        expected_length (int, optional): Expected length of the profile sequences. If None, no length filtering is applied.
        input_variables (List[str], optional): List of input variable names to load from JSON.
        target_variables (List[str], optional): List of target variable names to load from JSON.

    Raises:
        ValueError: If no valid JSON profiles are found matching the required input/target variables.
    """
    def __init__(
        self,
        data_folder: str,
        expected_length: Optional[int] = None,
        input_variables: Optional[List[str]] = None,
        target_variables: Optional[List[str]] = None
    ):
        super().__init__()
        self.data_folder = data_folder
        self.expected_length = expected_length
        self.input_variables = input_variables or ['pressure', 'temperature']
        self.target_variables = target_variables or ['heating_rate']

        # Gather JSON files in the folder
        self.file_list = [
            os.path.join(data_folder, f)
            for f in os.listdir(data_folder)
            if f.endswith(".json") and f != "normalization_metadata.json"
        ]
        self.valid_files = self._filter_valid_files()

        if not self.valid_files:
            raise ValueError(f"No valid JSON profiles found in {data_folder} matching the required variables.")

    def _filter_valid_files(self) -> List[str]:
        """
        Filter out invalid JSON profiles that do not meet the required criteria
        (presence of required variables and matching length if specified).

        Returns:
            List[str]: List of valid JSON file paths.
        """
        valid_files = []
        required_keys = set(self.input_variables + self.target_variables)

        for file_path in self.file_list:
            with open(file_path, "r") as f:
                profile = json.load(f)

            # Check presence of required keys
            if not all(k in profile for k in required_keys):
                print(f"Skipping {file_path}: Missing one or more of the required keys {required_keys}.")
                continue

            # Validate sequence length
            if self.expected_length is not None:
                first_input_var = self.input_variables[0]
                if isinstance(profile[first_input_var], list):
                    if len(profile[first_input_var]) != self.expected_length:
                        print(f"Skipping {file_path}: Incorrect profile length.")
                        continue
                else:
                    # If it's a scalar for the first var, it doesn't strictly disqualify the file
                    # but needs special handling during data loading
                    pass

            valid_files.append(file_path)

        return valid_files

    def __len__(self) -> int:
        return len(self.valid_files)

    def __getitem__(self, idx: int):
        """
        Retrieve the inputs and targets for a given index.

        Returns:
            (torch.Tensor, torch.Tensor): (inputs, targets)
                - inputs.shape == (seq_len, num_input_features)
                - targets.shape == (seq_len, num_target_features)
        """
        file_path = self.valid_files[idx]
        with open(file_path, "r") as f:
            profile = json.load(f)

        # Build input features
        inputs_list = []
        seq_length = None

        for var in self.input_variables:
            val = profile[var]
            if isinstance(val, list):
                seq_length = len(val)
                inputs_list.append(val)
            else:
                # Handle scalar values by expanding them to match expected_length or discovered seq_length
                if seq_length is None:
                    seq_length = self.expected_length or 1
                inputs_list.append([val] * seq_length)

        inputs = np.stack(inputs_list, axis=1)  # shape: (seq_len, num_input_vars)

        # Build target features
        targets_list = []
        for var in self.target_variables:
            val = profile[var]
            if isinstance(val, list):
                targets_list.append(val)
            else:
                # Handle scalar targets
                if seq_length is None:
                    seq_length = self.expected_length or 1
                targets_list.append([val] * seq_length)

        targets = np.stack(targets_list, axis=1)  # shape: (seq_len, num_target_vars)

        # Convert to tensors
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        return inputs, targets
