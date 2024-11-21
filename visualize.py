# visualize.py
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import json

plt.style.use('science')

def plot_profiles(folder='Data', base_filename='prof', num_plots=10):
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
    
    # Limit to the specified number of profiles
    profile_files = profile_files[:num_plots]
    
    plt.figure(figsize=(8, 10))
    
    for profile_file in profile_files:
        with open(profile_file, 'r') as f:
            data = json.load(f)
        pressure = np.array(data['pressure'])  # Should be in bar
        temperature = np.array(data['temperature'])
        
        plt.semilogy(temperature, pressure)
    
    plt.gca().invert_yaxis()
    plt.xlabel('Temperature (K)')
    plt.ylabel('Pressure (bar)')
    plt.title('Pressure-Temperature Profiles')
    plt.grid(True)
    plt.savefig('Figures/pt_profiles.png')
    plt.close()
    print("✔ Visualization saved to Figures/pt_profiles.png")
