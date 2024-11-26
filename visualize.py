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



def model_predictions(model, test_loader, save_path="Figures", device="cpu", N=3):
    """
    Generate and visualize predictions from the model on the test set.
    
    Parameters:
    - model: The trained model.
    - test_loader: DataLoader for the test set.
    - save_path: Directory to save the prediction plots.
    - device: Device to run the model on (e.g., 'cpu' or 'cuda').
    - N: Number of profiles to plot (default: 3).
    """
    import os
    import matplotlib.pyplot as plt

    model.eval()
    model.to(device)
    
    os.makedirs(save_path, exist_ok=True)

    num_plots = 0  # Counter for generated plots

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if num_plots >= N:  # Stop if the required number of plots is generated
                break

            inputs, targets = inputs.to(device), targets.to(device)

            # Generate predictions
            predictions = model(inputs_main=inputs).squeeze(-1)

            # Convert to numpy for plotting
            predictions = predictions.cpu().numpy()
            targets = targets.cpu().numpy()

            # Plot predictions vs actual values for each profile
            for j in range(len(predictions)):
                if num_plots >= N:  # Check again inside the batch loop
                    break

                plt.figure(figsize=(8, 6))
                plt.plot(targets[j], label="Actual Net Flux", marker="o", linestyle="-", color="blue")
                plt.plot(predictions[j], label="Predicted Net Flux", marker="x", linestyle="--", color="orange")
                plt.xlabel("Layer Index")
                plt.ylabel("Net Flux")
                plt.title(f"Profile {num_plots + 1}: Actual vs Predicted Net Flux")
                plt.legend()
                plt.savefig(f"{save_path}/profile_{num_plots + 1}.png", dpi=250)
                plt.close()

                num_plots += 1  # Increment the plot counter

                if num_plots >= N:  # Stop if the required number of plots is generated
                    break
