import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

# Set up style for pretty plots
plt.style.use('science')

def plot_profiles(filename='Data/profiles.h5', num_plots=10):
    """Plot a subset of profiles from the HDF5 file and save the plot."""
    
    # Ensure Figures directory exists
    os.makedirs('Figures', exist_ok=True)
    
    # Load data
    with h5py.File(filename, 'r') as hf:
        pressure = hf['P'][:]
        temperature_profiles = hf['temperature'][:]
    
    # Adjust num_plots if it exceeds the available number of profiles
    available_profiles = len(temperature_profiles)
    if num_plots > available_profiles:
        print(f"Warning: Requested {num_plots} profiles, but only {available_profiles} are available.")
        num_plots = available_profiles

    # Select a random subset of profiles to plot
    indices = np.random.choice(available_profiles, num_plots, replace=False)
    subset_profiles = temperature_profiles[indices, :]
    
    # Plot each selected profile
    plt.figure(figsize=(8, 6))
    for profile in subset_profiles:
        plt.semilogy(profile, pressure, alpha=0.7)  # Semi-logarithmic scale for better readability
    
    # Customize plot appearance
    plt.xlim(0, 5000)
    plt.ylim(1e2, 1e-6)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Pressure (bar)')
    plt.title('Sample Temperature-Pressure Profiles')
    
    # Save the plot
    plot_filename = 'Figures/sample_profiles.png'
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    print(f"Plot saved to {plot_filename}")
