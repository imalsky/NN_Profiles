# visualize.py
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import random
from matplotlib.cm import viridis
from matplotlib.colors import to_rgba
import torch

plt.style.use('science')

def plot_profiles(folder='Data', base_filename='prof', num_profiles=10):
    """
    Plot a subset of PT profiles saved as JSON files in the specified folder.
    
    Parameters:
    - folder (str): The folder where the profile JSON files are saved.
    - base_filename (str): The base filename used when saving the profiles.
    - num_profiles (int): Number of profiles to plot.
    """
    profile_files = glob.glob(os.path.join(folder, f"{base_filename}_*.json"))
    if not profile_files:
        print("No profile files found to plot.")
        return
    
    # Randomly select a subset of profile files
    if num_profiles < len(profile_files):
        profile_files = random.sample(profile_files, num_profiles)  # Randomly pick num_profiles files
    else:
        print("Requested number of profiles exceeds available files. Using all available files.")
    
    # Prepare the viridis colormap for the number of profiles
    colors = [to_rgba(viridis(i / max(1, len(profile_files) - 1))) for i in range(len(profile_files))]
    
    plt.figure(figsize=(8, 10))
    
    for profile_file, color in zip(profile_files, colors):
        with open(profile_file, 'r') as f:
            data = json.load(f)
        pressure = np.array(data['pressure'])  # Should be in bar
        temperature = np.array(data['temperature'])
        
        plt.semilogy(temperature, pressure, color=color)
    
    plt.ylim(1e2, 1e-5)
    plt.xlim(0, 4000)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Pressure (bar)')
    plt.title('Pressure-Temperature Profiles')
    plt.savefig('Figures/pt_profiles.png', dpi=250)
    plt.close()
    print("✔ Visualization saved to Figures/pt_profiles.png")


def plot_fluxes(folder, base_filename, num_profiles):
    """
    Plot a subset of PT profiles saved as JSON files in the specified folder.
    
    Parameters:
    - folder (str): The folder where the profile JSON files are saved.
    - base_filename (str): The base filename used when saving the profiles.
    - num_plots (int): Number of profiles to plot.
    """
    # Get list of profile files
    profile_files = glob.glob(os.path.join(folder, f"{base_filename}_*.json"))
    if not profile_files:
        print("No profile files found to plot.")
        return
    
    # Randomly select a subset of profile files
    if num_profiles < len(profile_files):
        profile_files = random.sample(profile_files, num_profiles)  # Randomly pick num_profiles files
    else:
        print("Requested number of profiles exceeds available files. Using all available files.")
    
    # Prepare the viridis colormap for the number of profiles
    colors = [to_rgba(viridis(i / max(1, len(profile_files) - 1))) for i in range(len(profile_files))]
    
    plt.figure(figsize=(10, 6))
    
    for profile_file, color in zip(profile_files, colors):
        with open(profile_file, 'r') as f:
            data = json.load(f)
        pressure = np.array(data['pressure'])  # Should be in bar
        net_flux = np.array(data['net_flux'])
        
        plt.plot(net_flux, pressure, color=color)
    
    plt.yscale('log')
    plt.ylim(1e2, 1e-5)
    plt.xlim(0, 1e6)

    plt.xlabel(r'Net Flux (W/m$^2$)')
    plt.ylabel('Pressure (bar)')

    plt.savefig('Figures/fluxes.png', dpi=250)
    plt.close()
    print("✔ Visualization saved to Figures/fluxes.png")



def model_predictions(model, test_loader, normalization_metadata, save_path='Figures', device='cpu', N=5):
    """
    Generate and visualize predictions from the model on the test set.

    Parameters:
    - model: The trained model.
    - test_loader: DataLoader for the test set.
    - normalization_metadata: Metadata used for de-normalizing the data.
    - save_path: Directory to save the prediction plots.
    - device: Device to run the model on (e.g., 'cpu' or 'cuda').
    - N: Number of profiles to plot (default: 5).
    """
    os.makedirs(save_path, exist_ok=True)

    # Extract normalization methods
    net_flux_norm_method = normalization_metadata["normalization_methods"].get("net_flux", "standard")
    pressure_norm_method = normalization_metadata["normalization_methods"].get("pressure", "standard")
    Tstar_norm_method = normalization_metadata["normalization_methods"].get("Tstar", "standard")

    # Extract net flux normalization parameters
    if net_flux_norm_method == "standard":
        net_flux_stats = normalization_metadata["net_flux"]
    elif net_flux_norm_method == "ratio_standard":
        net_flux_stats = normalization_metadata["net_flux_ratio"]
    else:
        raise ValueError(f"Unknown net flux normalization method: {net_flux_norm_method}")

    net_flux_mean = net_flux_stats["mean"]
    net_flux_std = net_flux_stats["std"]

    # Extract pressure normalization parameters
    if pressure_norm_method == "standard":
        pressure_mean = normalization_metadata["log_pressure"]["mean"]
        pressure_std = normalization_metadata["log_pressure"]["std"]
    elif pressure_norm_method == "min-max":
        pressure_min = normalization_metadata["log_pressure"]["min"]
        pressure_max = normalization_metadata["log_pressure"]["max"]
    else:
        raise ValueError(f"Unknown pressure normalization method: {pressure_norm_method}")

    # Extract Tstar normalization parameters
    Tstar_mean = normalization_metadata["Tstar"]["mean"]
    Tstar_std = normalization_metadata["Tstar"]["std"]

    # Get Stefan-Boltzmann constant
    sigma_sb = normalization_metadata["stefan_boltzmann_constant"]

    model.eval()
    model.to(device)
    
    num_plots = 0  # Counter for generated plots

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if num_plots >= N:
                break
            inputs, targets, _, _ = batch  # Ignore the extra items
            inputs, targets = inputs.to(device), targets.to(device)

            # Generate predictions
            predictions = model(inputs_main=inputs).squeeze(-1)

            # Move data to CPU and convert to numpy
            predictions = predictions.cpu().numpy()
            targets = targets.cpu().numpy()
            inputs = inputs.cpu().numpy()

            batch_size = predictions.shape[0]

            # Denormalize pressures
            if pressure_norm_method == "standard":
                pressures_norm = inputs[:, :, 0]
                log_pressures = pressures_norm * pressure_std + pressure_mean
            elif pressure_norm_method == "min-max":
                pressures_norm = inputs[:, :, 0]
                log_pressures = pressures_norm * (pressure_max - pressure_min) + pressure_min
            pressures = 10 ** log_pressures  # Convert log_pressure to pressure

            # Determine Tstar index in inputs
            num_features = inputs.shape[2]
            if num_features == 3:  # [pressure, temperature, Tstar]
                Tstar_index = 2
            elif num_features == 2:  # [temperature, Tstar] or [pressure, temperature]
                # Check if Tstar is included
                if "include_Tstar" in normalization_metadata:
                    if normalization_metadata["include_Tstar"]:
                        Tstar_index = 1  # Assuming features are [temperature, Tstar]
                    else:
                        Tstar_index = None
                else:
                    Tstar_index = None
            else:
                Tstar_index = None

            for i in range(batch_size):
                if num_plots >= N:
                    break

                # Denormalize net flux
                if net_flux_norm_method == "standard":
                    # Direct standardization
                    net_flux_pred = predictions[i] * net_flux_std + net_flux_mean
                    net_flux_true = targets[i] * net_flux_std + net_flux_mean
                elif net_flux_norm_method == "ratio_standard":
                    # Denormalize net flux ratios
                    net_flux_ratio_pred = predictions[i] * net_flux_std + net_flux_mean
                    net_flux_ratio_true = targets[i] * net_flux_std + net_flux_mean

                    # Extract normalized Tstar from inputs
                    if Tstar_index is not None:
                        Tstar_norm = inputs[i, 0, Tstar_index]  # First time step, Tstar feature
                        # Denormalize Tstar
                        Tstar = Tstar_norm * Tstar_std + Tstar_mean
                    else:
                        raise ValueError("Tstar is required for 'ratio_standard' net flux normalization but not found in inputs.")

                    F_star = sigma_sb * Tstar ** 4  # Scalar value

                    net_flux_pred = net_flux_ratio_pred * F_star
                    net_flux_true = net_flux_ratio_true * F_star
                else:
                    raise ValueError(f"Unknown net flux normalization method: {net_flux_norm_method}")

                # Fractional Error
                fractional_error = np.abs(net_flux_pred - net_flux_true) / np.abs(net_flux_true + 1e-8)  # Adding epsilon to avoid division by zero

                fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                # First panel: Actual vs Predicted Net Flux
                axes[0].plot(pressures[i], net_flux_true, label="Actual", marker="o", linestyle="-", color="blue")
                axes[0].plot(pressures[i], net_flux_pred, label="Predicted", marker="x", linestyle="--", color="orange")
                axes[0].set_xlabel("Pressure (bar)")
                axes[0].set_ylabel(r"Net Flux (W/m$^2$)")
                axes[0].set_xscale('log')
                axes[0].invert_xaxis()  # Optional: invert x-axis if needed
                axes[0].legend()

                # Second panel: Fractional Error
                axes[1].plot(pressures[i], fractional_error * 100, label="Percent Error", color="Black")
                axes[1].set_xlabel("Pressure (bar)")
                axes[1].set_ylabel("Percent Error")
                axes[1].set_xscale('log')
                axes[1].invert_xaxis()  # Optional: invert x-axis if needed
                axes[1].legend()

                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f"profile_{num_plots + 1}.png"), dpi=250)
                plt.close(fig)

                num_plots += 1

    print(f"Generated {num_plots} prediction plots.")