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



def model_predictions(model, test_loader, normalization_metadata, save_path="Figures", device="cpu", N=5):
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

    # Extract normalization stats
    net_flux_mean = normalization_metadata["net_flux"]["mean"]
    net_flux_std = normalization_metadata["net_flux"]["std"]

    pressure_min = normalization_metadata["pressure"]["min"]
    pressure_max = normalization_metadata["pressure"]["max"]

    model.eval()
    model.to(device)
    
    num_plots = 0  # Counter for generated plots

    with torch.no_grad():
        for inputs, targets in test_loader:
            if num_plots >= N:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            # Generate predictions
            predictions = model(inputs_main=inputs).squeeze(-1)

            # Convert to numpy for plotting
            predictions = predictions.cpu().numpy() * net_flux_std + net_flux_mean  # De-normalize predictions
            targets = targets.cpu().numpy() * net_flux_std + net_flux_mean  # De-normalize targets
            pressures = inputs[:, :, 0].cpu().numpy()  # Extract pressure inputs
            pressures = pressures * (pressure_max - pressure_min) + pressure_min  # De-normalize pressures

            for i in range(predictions.shape[0]):
                if num_plots >= N:
                    break

                # Fractional Error
                fractional_error = np.abs(predictions[i] - targets[i]) / np.abs(targets[i])

                fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                # First panel: Actual vs Predicted Net Flux
                axes[0].plot(10 ** pressures[i], targets[i], label="Actual", marker="o", linestyle="-", color="blue")
                axes[0].plot(10 ** pressures[i], predictions[i], label="Predicted", marker="x", linestyle="--", color="orange")
                axes[0].set_xlabel("Pressure (bar)")
                axes[0].set_ylabel(r"Net Flux (W/m$^2$)")
                axes[0].set_xscale('log')
                axes[0].legend()

                # Second panel: Fractional Error
                axes[1].plot(10 ** pressures[i], fractional_error * 100, label="Percent Error", color="Black")
                axes[1].set_xlabel("Pressure (bar)")
                axes[1].set_ylabel("Percent Error")
                axes[1].set_xscale('log')
                axes[1].legend()

                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f"profile_{num_plots + 1}.png"), dpi=250)
                plt.close(fig)

                num_plots += 1