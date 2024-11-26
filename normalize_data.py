import os
import json
import numpy as np

# Global constants
STEFAN_BOLTZMANN_CONSTANT = 5.67e-8  # Stefan-Boltzmann constant in W/m^2/K^4

# Create the output directory if it doesn't exist
input_folder = "Data/Profiles"
output_folder = "Data/Normalized_Profiles"
os.makedirs(output_folder, exist_ok=True)

def calculate_global_min_max(input_folder):
    """Calculate global min and max for each variable."""
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

    # Get global max temperature
    global_temp_max = max(temperature_values)

    # Dynamically calculate NET_FLUX_MAX
    global_net_flux_max = STEFAN_BOLTZMANN_CONSTANT * global_temp_max**4

    # Print the global max for temperature and net flux
    print(f"Temperature Max: {global_temp_max}")
    print(f"Net Flux Max: {global_net_flux_max}")

    global_min_max = {
        "pressure": (min(pressure_values), max(pressure_values)),
        "temperature_max": global_temp_max,
        "net_flux_max": global_net_flux_max,
        "Tstar": (min(tstar_values), max(tstar_values))
    }
    
    return global_min_max

def normalize(data, global_min, global_max):
    """Normalize data to [0, 1] using global min and max."""
    if global_min == global_max:
        # If min and max are the same, all values are constant; assign normalized value of 0
        return 0 if isinstance(data, (int, float)) else np.zeros_like(data)
    return (data - global_min) / (global_max - global_min)

def normalize_temperature(temperature, temp_max):
    """Normalize temperature to [0, 1] based on MAX_TEMPERATURE."""
    return normalize(temperature, 0, temp_max)

def normalize_net_flux(net_flux, net_flux_max):
    """Normalize net flux to [0, 1] based on MAX_NET_FLUX."""
    return normalize(net_flux, 0, net_flux_max)

def process_profiles(input_folder, output_folder, global_min_max):
    """Process and normalize all profiles in the input folder."""
    profile_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    if not profile_files:
        print("No profile files found to process.")
        return

    # Save normalization metadata
    normalization_metadata = {
        **global_min_max,
        "stefan_boltzmann_constant": STEFAN_BOLTZMANN_CONSTANT
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

        # Normalize the temperature using MAX_TEMPERATURE
        temp_max = global_min_max["temperature_max"]
        profile["temperature"] = normalize_temperature(unnormalized_temperature, temp_max).tolist()

        # Normalize the net flux using MAX_NET_FLUX
        net_flux_max = global_min_max["net_flux_max"]
        profile["net_flux"] = normalize_net_flux(np.array(profile["net_flux"]), net_flux_max).tolist()

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
