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


def delete_old_profiles(folder='Data/Profiles', base_filename='prof'):
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

    # print(f"Deleted {deleted_files} old profile(s) in '{folder}'.")


def save_data(data, folder='Data/Profiles', base_filename='profile_'):
    """
    Save data in dictionary format to a JSON file with a unique ordered filename.

    Parameters:
    - data (dict): Dictionary containing the data to save.
    - folder (str): Path to the folder where data will be saved.
    - base_filename (str): Base name for the saved file.
    """
    os.makedirs(folder, exist_ok=True)

    # Find the next available index
    existing_files = [f for f in os.listdir(folder) if f.startswith(
        base_filename) and f.endswith('.json')]
    indices = [int(f.split('_')[-1].split('.')[0])
               for f in existing_files if '_' in f and f.split('_')[-1].split('.')[0].isdigit()]
    next_index = max(indices) + 1 if indices else 1

    # Construct unique filename
    filename = os.path.join(folder, f"{base_filename}_{next_index}.json")

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Load normalization metadata


def load_normalization_metadata(metadata_path="Data/Normalized_Profiles/normalization_metadata.json"):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Normalization metadata not found at {metadata_path}")

    with open(metadata_path, 'r') as f:
        normalization_metadata = json.load(f)
    return normalization_metadata



