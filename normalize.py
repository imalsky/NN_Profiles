import os
import json
import numpy as np


def calculate_global_stats(input_folder, pressure_normalization_method):
    """Calculate global stats for all numerical variables that need standardization."""
    # Initialize dictionaries to hold values for each key
    key_values = {}
    profile_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    if not profile_files:
        print("No profile files found.")
        return None

    for profile_file in profile_files:
        input_path = os.path.join(input_folder, profile_file)
        with open(input_path, "r") as f:
            profile = json.load(f)

        for key, value in profile.items():
            if key not in key_values:
                key_values[key] = []

            # Handle scalar numerical values
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    print(f"Debug: NaN encountered in key '{key}' as scalar in file '{profile_file}'.")
                key_values[key].append(value)

            # Handle lists of numerical values
            elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                if key == "pressure":
                    # Apply log10 to pressures
                    log_pressures = np.log10(value)
                    if np.isnan(log_pressures).any():
                        print(
                            f"Debug: NaN encountered in 'log_pressures' for key '{key}' in file '{profile_file}'. Values: {value}")
                    key_values[key].extend(log_pressures)
                else:
                    if np.isnan(value).any():
                        print(f"Debug: NaN encountered in key '{key}' list in file '{profile_file}'. Values: {value}")
                    key_values[key].extend(value)



    # Calculate global stats for standardization
    stats = {}
    for key, values in key_values.items():
        values = np.array(values)
        if key == "pressure":
            # Pressure normalization uses log10(pressure)
            if pressure_normalization_method == 'standard':
                stats[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values)
                }
            elif pressure_normalization_method == 'min-max':
                stats[key] = {
                    "min": np.min(values),
                    "max": np.max(values)
                }
        else:
            # For other keys, calculate mean and std for standard normalization
            stats[key] = {
                "mean": np.mean(values),
                "std": np.std(values)
            }

    # Add normalization methods to stats
    stats["normalization_methods"] = {
        key: "standard" if key != "pressure" else pressure_normalization_method
        for key in key_values.keys()
    }

    return stats


def normalize_standard(data, mean, std):
    """Standardize data to have mean 0 and standard deviation 1."""
    if std == 0:
        return data - mean  # This will be zero if data == mean
    return (data - mean) / (std)


def normalize_min_max(data, min_value, max_value):
    """Normalize data to [0, 1] using min and max values."""
    if max_value == min_value:
        return np.zeros_like(data)
    return (data - min_value) / (max_value - min_value)


def process_profiles(input_folder, output_folder, stats, pressure_normalization_method):
    """Process and normalize all elements in the input JSON profiles."""
    profile_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    if not profile_files:
        print("No profile files found to process.")
        return

    # Save normalization metadata
    normalization_metadata = stats.copy()
    metadata_path = os.path.join(output_folder, "normalization_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(normalization_metadata, f, indent=4)
    print(f"\n✔ Normalization metadata saved to: {metadata_path}")

    for profile_file in profile_files:
        input_path = os.path.join(input_folder, profile_file)
        output_path = os.path.join(output_folder, profile_file)

        with open(input_path, "r") as f:
            profile = json.load(f)

        normalized_profile = {}

        for key, value in profile.items():
            if key not in stats:
                # Retain non-numerical or unknown structures as-is
                normalized_profile[key] = value
                continue

            normalization_method = stats["normalization_methods"].get(key, "standard")
            key_stats = stats[key]

            if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                # Handle lists of numerical values
                data = np.array(value)
                if key == "pressure":
                    data = np.log10(data)

                if normalization_method == "standard":
                    mean = key_stats["mean"]
                    std = key_stats["std"]
                    normalized_values = normalize_standard(data, mean, std)
                elif normalization_method == "min-max":
                    min_val = key_stats["min"]
                    max_val = key_stats["max"]
                    normalized_values = normalize_min_max(data, min_val, max_val)
                else:
                    normalized_values = data  # No normalization

                normalized_profile[key] = normalized_values.tolist()

            elif isinstance(value, (int, float)):
                # Handle scalar numerical values
                data = value
                if normalization_method == "standard":
                    mean = key_stats["mean"]
                    std = key_stats["std"]
                    normalized_value = normalize_standard(data, mean, std)
                elif normalization_method == "min-max":
                    min_val = key_stats["min"]
                    max_val = key_stats["max"]
                    normalized_value = normalize_min_max(data, min_val, max_val)
                else:
                    normalized_value = data  # No normalization

                normalized_profile[key] = normalized_value
            else:
                # Retain non-numerical or unknown structures as-is
                normalized_profile[key] = value

        # Save the normalized profile
        with open(output_path, "w") as f:
            json.dump(normalized_profile, f, indent=4)

    print(f"✔ Processed and saved normalized profiles to {output_folder}")
