import os
import json
import numpy as np

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

    global_min_max = {
        "pressure": (min(pressure_values), max(pressure_values)),
        "temperature": (min(temperature_values), max(temperature_values)),
        "net_flux": (min(net_flux_values), max(net_flux_values)),
        "Tstar": (min(tstar_values), max(tstar_values))
    }
    
    return global_min_max

def normalize(data, global_min, global_max):
    """Normalize data to [0, 1] using global min and max."""
    if global_min == global_max:
        # If min and max are the same, all values are constant; assign normalized value of 0
        return 0 if isinstance(data, (int, float)) else np.zeros_like(data)
    return (data - global_min) / (global_max - global_min)

def process_profiles(input_folder, output_folder, global_min_max):
    """Process and normalize all profiles in the input folder."""
    profile_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    if not profile_files:
        print("No profile files found to process.")
        return

    # Save normalization metadata
    metadata_path = os.path.join(output_folder, "normalization_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(global_min_max, f, indent=4)
    print(f"✔ Normalization metadata saved to: {metadata_path}")

    for profile_file in profile_files:
        input_path = os.path.join(input_folder, profile_file)
        output_path = os.path.join(output_folder, profile_file)
        
        with open(input_path, "r") as f:
            profile = json.load(f)

        # Normalize each variable using global min and max
        pressure_log = np.log10(profile["pressure"])
        profile["pressure"] = normalize(pressure_log, *global_min_max["pressure"]).tolist()
        profile["temperature"] = normalize(np.array(profile["temperature"]), *global_min_max["temperature"]).tolist()
        profile["net_flux"] = normalize(np.array(profile["net_flux"]), *global_min_max["net_flux"]).tolist()
        profile["Tstar"] = normalize(profile["Tstar"], *global_min_max["Tstar"])
        
        # Save the normalized profile
        with open(output_path, "w") as f:
            json.dump(profile, f, indent=4)
        print(f"✔ Processed and saved: {profile_file}")

# Calculate global min and max
global_min_max = calculate_global_min_max(input_folder)

# Process profiles with global min and max
if global_min_max:
    print("Global min and max values calculated:", global_min_max)
    process_profiles(input_folder, output_folder, global_min_max)
