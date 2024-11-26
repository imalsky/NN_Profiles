import os
import json
import numpy as np

# Global constants for normalization
TEMPERATURE_MIN = 0  # Minimum temperature (in K)
TEMPERATURE_MAX = 5000  # Maximum temperature (in K)
NET_FLUX_MIN = 0  # Minimum net flux
NET_FLUX_MAX = 1e6  # Maximum net flux

# Create the output directory if it doesn't exist
input_folder = "Data/Profiles"
output_folder = "Data/Normalized_Profiles"
os.makedirs(output_folder, exist_ok=True)

def calculate_global_min_max(input_folder):
    """Calculate global min and max for each variable."""
    pressure_values = []
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
        tstar_values.append(profile["Tstar"])

    global_min_max = {
        "pressure": (min(pressure_values), max(pressure_values)),
        "Tstar": (min(tstar_values), max(tstar_values))
    }
    
    return global_min_max

def normalize(data, global_min, global_max):
    """Normalize data to [0, 1] using global min and max."""
    if global_min == global_max:
        # If min and max are the same, all values are constant; assign normalized value of 0
        return 0 if isinstance(data, (int, float)) else np.zeros_like(data)
    return (data - global_min) / (global_max - global_min)

def normalize_temperature(temperature):
    """Normalize temperature to [0, 1] based on global TEMPERATURE_MIN and TEMPERATURE_MAX."""
    return normalize(temperature, TEMPERATURE_MIN, TEMPERATURE_MAX)

def normalize_net_flux(net_flux):
    """Normalize net flux to [0, 1] based on global NET_FLUX_MIN and NET_FLUX_MAX."""
    return normalize(net_flux, NET_FLUX_MIN, NET_FLUX_MAX)

def process_profiles(input_folder, output_folder, global_min_max):
    """Process and normalize all profiles in the input folder."""
    profile_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    if not profile_files:
        print("No profile files found to process.")
        return

    # Save normalization metadata
    normalization_metadata = {
        **global_min_max,
        "temperature": {"min": TEMPERATURE_MIN, "max": TEMPERATURE_MAX},
        "net_flux": {"min": NET_FLUX_MIN, "max": NET_FLUX_MAX}
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

        # Normalize each variable using global min and max
        pressure_log = np.log10(profile["pressure"])
        profile["pressure"] = normalize(pressure_log, *global_min_max["pressure"]).tolist()

        # Keep a copy of the original unnormalized temperature for debugging
        unnormalized_temperature = np.array(profile["temperature"])

        # Normalize the temperature using global TEMPERATURE_MIN and TEMPERATURE_MAX
        profile["temperature"] = normalize_temperature(unnormalized_temperature).tolist()

        # Normalize the net flux using global NET_FLUX_MIN and NET_FLUX_MAX
        profile["net_flux"] = normalize_net_flux(np.array(profile["net_flux"])).tolist()

        profile["Tstar"] = normalize(profile["Tstar"], *global_min_max["Tstar"])

        # Save the normalized profile
        with open(output_path, "w") as f:
            json.dump(profile, f, indent=4)

# Calculate global min and max
global_min_max = calculate_global_min_max(input_folder)

# Process profiles with global min and max
if global_min_max:
    print("Global min and max values calculated:", global_min_max)
    process_profiles(input_folder, output_folder, global_min_max)
