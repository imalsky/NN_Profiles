import os
import json
import numpy as np
from profile_generator import ProfileGenerator
from opacity_structure import initialize_opacity_databases, calculate_opacity_structure
from visualize import plot_profiles

# Ensure required directories exist
os.makedirs('Inputs', exist_ok=True)
os.makedirs('Data', exist_ok=True)
os.makedirs('Figures', exist_ok=True)

# Load pressure range and configuration from parameters.json
with open('Inputs/parameters.json', 'r') as f:
    config = json.load(f)
    pressure_min = config['pressure_range']['min']
    pressure_max = config['pressure_range']['max']
    pressure_points = config['pressure_range']['points']

# Generate the pressure array based on the loaded range
P = np.logspace(np.log10(pressure_min), np.log10(pressure_max), num=pressure_points)

# Function to sample a constant or distribution
def sample_constant_or_distribution(param_config):
    if param_config['dist'] == 'fixed':
        return param_config['value']
    elif param_config['dist'] == 'uniform':
        return np.random.uniform(param_config['low'], param_config['high'])
    elif param_config['dist'] == 'normal':
        return np.random.normal(param_config['mean'], param_config['std'])
    else:
        raise ValueError(f"Unsupported distribution type: {param_config['dist']}")

# Sample planet parameters
grav = sample_constant_or_distribution(config['planet_params']['grav'])
rcp = sample_constant_or_distribution(config['planet_params']['rcp'])
albedo_surf = sample_constant_or_distribution(config['planet_params']['albedo_surf'])
Rp = sample_constant_or_distribution(config['planet_params']['Rp'])





# Main code
if __name__ == '__main__':
    N = 1  # Number of profiles to generate
    
    # Step 1: Generate PT profiles
    print("Generating PT profiles...")
    generator = ProfileGenerator(N, P, config_file='Inputs/parameters.json')
    generator.generate_profiles()
    profiles = generator.get_profiles()  # Get profiles in a format compatible with xk.Atm
    print(f"Generated {len(profiles)} profiles.")
    
    # Step 2: Initialize opacity databases
    print("Initializing opacity databases...")
    k_db, cia_db = initialize_opacity_databases(config_file='Inputs/parameters.json')
    


    # Step 3: Calculate opacity structure
    print("Calculating opacity structure for each profile...")
    atm_objects = calculate_opacity_structure(
        profiles=profiles,  # List of atmospheric profiles
        k_db=k_db,
        cia_db=cia_db,
        grav=grav,
        rcp=rcp,
        albedo_surf=albedo_surf,
        Rp=Rp
    )
    print(f"Processed {len(atm_objects)} profiles successfully.")

    # Example: Access data for the first profile
    if atm_objects:
        first_profile_data = atm_objects[0].data_dict
        print("First profile data:", first_profile_data)
    
    # Step 4: Visualize a subset of profiles
    print("Visualizing a subset of PT profiles...")
    plot_profiles(filename='Data/profiles.h5', num_plots=10)
    print("Visualization saved to Figures/")
