import os
import json
import numpy as np

# Global constants
STEFAN_BOLTZMANN_CONSTANT = 5.67e-8  # Stefan-Boltzmann constant in W/m^2/K^4

# Create the output directory if it doesn't exist
input_folder = "Data/Profiles"
output_folder = "Data/Normalized_Profiles"
os.makedirs(output_folder, exist_ok=True)


def calculate_global_stats(input_folder):
    """Calculate global stats for each variable."""
    pressure_values = []
    temperature_values = []
    net_flux_values = []
    tstar_values = []

    profile_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    if not profile_files:
        print("No profile files found.")
        return None

    for profile_file in profile_files:
        input_path = os.path.join(input_folder, profile_file)
        with open(input_path, "r") as f:
            profile = json.load(f)

        pressure_values.extend(np.log10(profile["pressure"]))
        temperature_values.extend(profile["temperature"])
        net_flux_values.extend(profile["net_flux"])
        tstar_values.append(profile["Tstar"])

    # Calculate stats
    stats = {
        "pressure": {"min": min(pressure_values), "max": max(pressure_values)},
        "temperature": {
            "min": min(temperature_values),
            "max": max(temperature_values),
            "mean": np.mean(temperature_values),
            "std": np.std(temperature_values),
        },
        "net_flux": {
            "mean": np.mean(net_flux_values),
            "std": np.std(net_flux_values),
        },
        "Tstar": {"min": min(tstar_values), "max": max(tstar_values)},
    }

    # Print stats for debugging
    print(f"Pressure Min: {stats['pressure']['min']}, Max: {stats['pressure']['max']}")
    print(
        f"Temperature Min: {stats['temperature']['min']}, Max: {stats['temperature']['max']}, "
        f"Mean: {stats['temperature']['mean']}, Std: {stats['temperature']['std']}"
    )
    print(
        f"Net Flux Mean: {stats['net_flux']['mean']}, Std: {stats['net_flux']['std']}"
    )
    print(f"Tstar Min: {stats['Tstar']['min']}, Max: {stats['Tstar']['max']}")

    return stats


def normalize_min_max(data, global_min, global_max):
    """Normalize data to [0, 1] using global min and max."""
    if global_min == global_max:
        return np.zeros_like(data)
    return (data - global_min) / (global_max - global_min)


def normalize_standard(data, mean, std):
    """Standardize data to have mean 0 and standard deviation 1."""
    if std == 0:
        return data - mean
    return (data - mean) / std


def process_profiles(input_folder, output_folder, stats):
    """Process and normalize all profiles in the input folder."""
    profile_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    if not profile_files:
        print("No profile files found to process.")
        return

    # Save normalization metadata
    normalization_metadata = {
        **stats,
        "stefan_boltzmann_constant": STEFAN_BOLTZMANN_CONSTANT,
    }
    metadata_path = os.path.join(output_folder, "normalization_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(normalization_metadata, f, indent=4)
    print(f"✔ Normalization metadata saved to: {metadata_path}")

    for profile_file in profile_files:
        input_path = os.path.join(input_folder, profile_file)
        output_path = os.path.join(output_folder, profile_file)

        with open(input_path, "r") as f:
            profile = json.load(f)

        # Normalize pressure
        profile["pressure"] = normalize_min_max(np.log10(profile["pressure"]),stats["pressure"]["min"],stats["pressure"]["max"]).tolist()

        # Normalize temperature: Use standardization (mean/std)
        temp_mean = stats["temperature"]["mean"]
        temp_std = stats["temperature"]["std"]
        profile["temperature"] = normalize_standard(np.array(profile["temperature"]), temp_mean, temp_std).tolist()

        # Normalize net flux: Use standardization (mean/std)
        net_flux_mean = stats["net_flux"]["mean"]
        net_flux_std = stats["net_flux"]["std"]
        profile["net_flux"] = normalize_standard(np.array(profile["net_flux"]), net_flux_mean, net_flux_std).tolist()

        # Normalize Tstar
        profile["Tstar"] = normalize_min_max(profile["Tstar"], stats["Tstar"]["min"], stats["Tstar"]["max"]).tolist()

        # Save the normalized profile
        with open(output_path, "w") as f:
            json.dump(profile, f, indent=4)


# Calculate global stats
global_stats = calculate_global_stats(input_folder)

# Process profiles with global stats
if global_stats:
    print("Global stats calculated:", global_stats)
    process_profiles(input_folder, output_folder, global_stats)
