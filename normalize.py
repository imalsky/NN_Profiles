import os
import json
import numpy as np


def calculate_iqr(values):
    """Calculate the median and IQR for a given array of values."""
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    median = np.median(values)
    return median, iqr


def robust_scale(data, median, iqr):
    """Robust scaling using median and IQR."""
    if iqr == 0:
        return data - median
    return (data - median) / iqr


def clip_outliers(values, lower_quantile=0.01, upper_quantile=0.99):
    """Clip outliers outside the specified quantile range to reduce their impact."""
    lower_bound = np.percentile(values, lower_quantile * 100)
    upper_bound = np.percentile(values, upper_quantile * 100)
    return np.clip(values, lower_bound, upper_bound)


def calculate_global_stats(input_folder, pressure_normalization_method,
                           use_robust_scaling=False, clip_outliers_before_scaling=False):
    """
    Calculate global stats for all numerical variables that need normalization.
    Adds robust scaling options and outlier clipping.
    """
    key_values = {}
    profile_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    if not profile_files:
        print("No profile files found.")
        return None

    # Collect all values for each key
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
                arr = np.array(value)

                # Apply log10 to pressure
                if key == "pressure":
                    arr = np.log10(arr)
                    if np.isnan(arr).any():
                        print(f"Debug: NaN in 'log_pressures' for '{key}' in '{profile_file}'. Values: {value}")

                if np.isnan(arr).any():
                    print(f"Debug: NaN encountered in '{key}' array in '{profile_file}'. Values: {value}")

                key_values[key].extend(arr)

    # Optionally clip outliers before computing statistics
    if clip_outliers_before_scaling:
        for k in key_values:
            arr = np.array(key_values[k])
            arr = clip_outliers(arr, lower_quantile=0.01, upper_quantile=0.99)
            key_values[k] = arr

    # Calculate stats for each key
    stats = {}
    for key, values in key_values.items():
        values = np.array(values)

        if key == "pressure":
            # Pressure normalization: 'standard' or 'min-max'
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
            # For other keys, choose between standardization and robust scaling
            if use_robust_scaling:
                median, iqr = calculate_iqr(values)
                stats[key] = {
                    "median": median,
                    "iqr": iqr
                }
            else:
                stats[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values)
                }

    # Determine normalization methods for each key
    normalization_methods = {}
    for key in key_values.keys():
        if key == "pressure":
            normalization_methods[key] = pressure_normalization_method
        else:
            normalization_methods[key] = "robust" if use_robust_scaling else "standard"

    # Add normalization methods and configuration to stats
    stats["normalization_methods"] = normalization_methods
    stats["config"] = {
        "use_robust_scaling": use_robust_scaling,
        "clip_outliers_before_scaling": clip_outliers_before_scaling
    }

    return stats


def normalize_standard(data, mean, std):
    """Standardize data to have mean 0 and std 1."""
    if std == 0:
        return data - mean
    return (data - mean) / std


def normalize_min_max(data, min_value, max_value):
    """Normalize data to [0, 1] using min and max values."""
    if max_value == min_value:
        return np.zeros_like(data)
    return (data - min_value) / (max_value - min_value)


def normalize_robust(data, median, iqr):
    """Apply robust scaling."""
    return robust_scale(data, median, iqr)


def process_profiles(input_folder, output_folder, stats, pressure_normalization_method):
    """Process and normalize all elements in the input JSON profiles using the given stats."""
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

    methods = stats["normalization_methods"]

    for profile_file in profile_files:
        input_path = os.path.join(input_folder, profile_file)
        output_path = os.path.join(output_folder, profile_file)

        with open(input_path, "r") as f:
            profile = json.load(f)

        normalized_profile = {}

        for key, value in profile.items():
            # If stats do not exist for the key, leave as-is
            if key not in stats or key == "normalization_methods" or key == "config":
                normalized_profile[key] = value
                continue

            normalization_method = methods.get(key, "standard")
            key_stats = stats[key]

            def normalize_data(data):
                if normalization_method == "standard":
                    mean = key_stats.get("mean", 0)
                    std = key_stats.get("std", 1)
                    return normalize_standard(data, mean, std)
                elif normalization_method == "min-max":
                    min_val = key_stats.get("min", 0)
                    max_val = key_stats.get("max", 1)
                    return normalize_min_max(data, min_val, max_val)
                elif normalization_method == "robust":
                    median = key_stats.get("median", 0)
                    iqr = key_stats.get("iqr", 1)
                    return normalize_robust(data, median, iqr)
                else:
                    return data  # No normalization

            if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                data = np.array(value)
                if key == "pressure":
                    data = np.log10(data)
                normalized_values = normalize_data(data)
                normalized_profile[key] = normalized_values.tolist()

            elif isinstance(value, (int, float)):
                data = np.array(value)
                normalized_value = normalize_data(data)
                if isinstance(normalized_value, np.ndarray):
                    normalized_value = normalized_value.item()
                normalized_profile[key] = normalized_value
            else:
                # Retain non-numerical or unknown structures as-is
                normalized_profile[key] = value

        # Save the normalized profile
        with open(output_path, "w") as f:
            json.dump(normalized_profile, f, indent=4)

    print(f"✔ Processed and saved normalized profiles to {output_folder}")
