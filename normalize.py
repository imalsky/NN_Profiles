import os
import json
import numpy as np

def calculate_global_stats(input_folder, pressure_normalization_method):
    """Calculate global stats for variables that need standardization."""
    temperature_ratios = []
    net_flux_values = []
    tstar_values = []
    log_pressure_values = []

    profile_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    if not profile_files:
        print("No profile files found.")
        return None

    for profile_file in profile_files:
        input_path = os.path.join(input_folder, profile_file)
        with open(input_path, "r") as f:
            profile = json.load(f)

        # Check if all required keys are present
        required_keys = ["pressure", "temperature", "net_flux", "Tstar"]
        if not all(key in profile for key in required_keys):
            print(f"Skipping {profile_file}: Missing one of the required keys.")
            continue

        pressures = profile["pressure"]
        temperatures = profile["temperature"]
        net_fluxes = profile["net_flux"]
        Tstar = profile["Tstar"]

        if Tstar == 0:
            Tstar = 1e-10
            #print(f"Skipping {profile_file}: Tstar is zero.")
            #continue

        tstar_values.append(Tstar)

        # Compute log10(pressure)
        log_pressures = np.log10(pressures)
        log_pressure_values.extend(log_pressures)

        # Normalize temperatures by Tstar
        temperature_ratio = np.array(temperatures) / Tstar
        temperature_ratios.extend(temperature_ratio)

        # Add net flux values for standardization
        net_flux_values.extend(net_fluxes)

    # Calculate global stats for standardization
    stats = {
        "temperature_ratio": {
            "mean": np.mean(temperature_ratios),
            "std": np.std(temperature_ratios),
        },
        "net_flux": {
            "mean": np.mean(net_flux_values),
            "std": np.std(net_flux_values),
        },
        "Tstar": {
            "mean": np.mean(tstar_values),
            "std": np.std(tstar_values),
            "min": min(tstar_values),
            "max": max(tstar_values),
        },
    }

    # Add pressure stats based on the selected normalization method
    if pressure_normalization_method == 'standard':
        stats["log_pressure"] = {
            "mean": np.mean(log_pressure_values),
            "std": np.std(log_pressure_values),
        }
    elif pressure_normalization_method == 'min-max':
        min_log_pressure = np.min(log_pressure_values)
        max_log_pressure = np.max(log_pressure_values)
        stats["log_pressure"] = {
            "min": min_log_pressure,
            "max": max_log_pressure
        }
        #print(f"Pressure (log) Min: {min_log_pressure}, Max: {max_log_pressure}")

    # Print stats for debugging
    #print("\nCalculated Global Statistics:")
    #print(f"Temperature/Tstar Mean: {stats['temperature_ratio']['mean']:.3f}, Std: {stats['temperature_ratio']['std']:.3f}")
    #print(f"Net Flux Mean: {stats['net_flux']['mean']:.3e}, Std: {stats['net_flux']['std']:.3e}")
    #print(f"Tstar Mean: {stats['Tstar']['mean']:.2f}, Std: {stats['Tstar']['std']:.2f}, Min: {stats['Tstar']['min']}, Max: {stats['Tstar']['max']}")

    #if pressure_normalization_method == 'standard':
    #    print(f"Pressure (log) Mean: {stats['log_pressure']['mean']:.3f}, Std: {stats['log_pressure']['std']:.3f}")
    #elif pressure_normalization_method == 'min-max':
    #    print(f"Pressure (log) Min: {stats['log_pressure']['min']}, Max: {stats['log_pressure']['max']}")

    return stats


def normalize_standard(data, mean, std):
    """Standardize data to have mean 0 and standard deviation 1."""
    if std == 0:
        return data - mean
    return (data - mean) / std

def normalize_min_max(data, min_value, max_value):
    """Normalize data to [0, 1] using min and max values."""
    if max_value == min_value:
        return np.zeros_like(data)
    return (data - min_value) / (max_value - min_value)

def process_profiles(input_folder, output_folder, stats, pressure_normalization_method):
    """Process and normalize all profiles in the input folder."""
    profile_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    if not profile_files:
        print("No profile files found to process.")
        return

    # Save normalization metadata
    normalization_metadata = {
        **stats,
        "pressure_normalization_method": pressure_normalization_method,
        "normalization_methods": {
            "pressure": pressure_normalization_method,
            "temperature": "standard",
            "net_flux": "standard",
            "Tstar": "standard",
        }
    }
    metadata_path = os.path.join(output_folder, "normalization_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(normalization_metadata, f, indent=4)
    print(f"\n✔ Normalization metadata saved to: {metadata_path}")

    for profile_file in profile_files:
        input_path = os.path.join(input_folder, profile_file)
        output_path = os.path.join(output_folder, profile_file)

        with open(input_path, "r") as f:
            profile = json.load(f)

        # Check if all required keys are present
        required_keys = ["pressure", "temperature", "net_flux", "Tstar"]
        if not all(key in profile for key in required_keys):
            print(f"Skipping {profile_file}: Missing one of the required keys.")
            continue

        pressures = profile["pressure"]
        temperatures = profile["temperature"]
        net_fluxes = profile["net_flux"]
        Tstar = profile["Tstar"]

        if Tstar == 0:
            Tstar = 1e-10
            #print(f"Skipping {profile_file}: Tstar is zero.")
            #continue

        # Normalize Pressure
        log_pressures = np.log10(pressures)
        if pressure_normalization_method == 'standard':
            mean = stats["log_pressure"]["mean"]
            std = stats["log_pressure"]["std"]
            norm_log_pressures = normalize_standard(log_pressures, mean, std)
        elif pressure_normalization_method == 'min-max':
            min_val = stats["log_pressure"]["min"]
            max_val = stats["log_pressure"]["max"]
            norm_log_pressures = normalize_min_max(log_pressures, min_val, max_val)
        else:
            raise ValueError(f"Unknown pressure normalization method: {pressure_normalization_method}")

        # Normalize Temperature
        temperature_ratios = np.array(temperatures) / Tstar
        mean = stats["temperature_ratio"]["mean"]
        std = stats["temperature_ratio"]["std"]
        norm_temperature_ratios = normalize_standard(temperature_ratios, mean, std)

        # Normalize Net Flux
        mean = stats["net_flux"]["mean"]
        std = stats["net_flux"]["std"]
        norm_net_flux_ratios = normalize_standard(np.array(net_fluxes), mean, std)

        # Normalize Tstar
        mean = stats["Tstar"]["mean"]
        std = stats["Tstar"]["std"]
        norm_Tstar = normalize_standard(Tstar, mean, std)

        # Overwrite original keys with normalized values
        profile["pressure"] = norm_log_pressures.tolist()
        profile["temperature"] = norm_temperature_ratios.tolist()
        profile["net_flux"] = norm_net_flux_ratios.tolist()
        profile["Tstar"] = norm_Tstar if isinstance(norm_Tstar, float) else norm_Tstar.tolist()

        # Save the normalized profile
        with open(output_path, "w") as f:
            json.dump(profile, f, indent=4)

    print(f"✔ Processed and saved normalized profiles to {output_folder}")


