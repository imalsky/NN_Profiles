import os
import json
import numpy as np


def load_config(config_file='Inputs/parameters.json'):
    """Load the configuration from a JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def create_directories(*dirs):
    """Ensure required directories exist."""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def sample_constant_or_distribution(param_config):
    """Sample a value based on the distribution specified in the config."""
    if param_config['dist'] == 'fixed':
        return param_config['value']
    elif param_config['dist'] == 'uniform':
        return np.random.uniform(param_config['low'], param_config['high'])
    elif param_config['dist'] == 'normal':
        return np.random.normal(param_config['mean'], param_config['std'])
    else:
        raise ValueError(f"Unsupported distribution type: {param_config['dist']}")


def delete_old_profiles(folder='Data', base_filename='prof'):
    """
    Delete all old profile files in the specified folder.

    Parameters:
    - folder (str): Directory to clean up.
    - base_filename (str): Base name of profile files to delete.
    """
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist. Nothing to delete.")
        return

    deleted_files = 0
    for file in os.listdir(folder):
        if file.startswith(base_filename) and file.endswith('.json'):
            os.remove(os.path.join(folder, file))
            deleted_files += 1

    print(f"Deleted {deleted_files} old profile(s) in '{folder}'.")
