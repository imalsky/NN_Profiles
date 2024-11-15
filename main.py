import os
import json
import numpy as np
from opacity_structure import (
    initialize_opacity_databases,
    set_stellar_spectrum,
    calculate_opacity_structure,
    calculate_heating_rates_and_fluxes
)
from profile_generator import ProfileGenerator
from visualize import plot_profiles

# Ensure required directories exist
os.makedirs('Inputs', exist_ok=True)
os.makedirs('Data', exist_ok=True)
os.makedirs('Figures', exist_ok=True)

# Load configuration
with open('Inputs/parameters.json', 'r') as f:
    config = json.load(f)

# Extract the datapath from the configuration
datapath = config['datapath']

# Generate the pressure array
pressure_range = config['pressure_range']
P = np.logspace(
    np.log10(pressure_range['min']),
    np.log10(pressure_range['max']),
    num=pressure_range['points']
)

# Function to sample constants or distributions
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

# Main function
if __name__ == '__main__':
    print("\n" + "="*70)
    print(f"{'ATMOSPHERIC MODELING PIPELINE':^70}")
    print("="*70)
    
    # Step 1: Generate PT profiles
    print("\n[1] Generating Pressure-Temperature Profiles...")
    generator = ProfileGenerator(
        N=1,  # Number of profiles to generate
        P=P,
        config_file='Inputs/parameters.json'
    )
    generator.generate_profiles()
    profiles = generator.get_profiles()
    print(f"✔ {len(profiles)} PT profile(s) generated successfully.")
    
    # Step 2: Initialize opacity databases
    print("\n[2] Initializing Opacity Databases...")
    k_db, cia_db = initialize_opacity_databases(config_file='Inputs/parameters.json')
    print("✔ Opacity databases initialized.")
    
    # Step 3: Set the stellar spectrum
    print("\n[3] Loading Stellar Spectrum...")
    stellar_spectrum_file = config['model_params'].get('stellar_spectrum_file', 'stellar_spectra/default_spectrum.dat')
    stellar_spectrum = set_stellar_spectrum(datapath=datapath, filename=stellar_spectrum_file)
    print(f"✔ Stellar spectrum loaded successfully.")

    # Step 4: Calculate opacity structure
    print("\n[4] Calculating Opacity Structure for Atmospheric Profiles...")
    rayleigh = config['model_params'].get('rayleigh', True)  # Explicit rayleigh parameter
    atm_objects = calculate_opacity_structure(
        profiles=profiles,
        k_db=k_db,
        cia_db=cia_db,
        grav=grav,
        rcp=rcp,
        albedo_surf=albedo_surf,
        Rp=Rp,
        rayleigh=rayleigh,
        stellar_spectrum=stellar_spectrum  # Pass the loaded stellar spectrum here
    )
    print(f"✔ {len(atm_objects)} profile(s) processed successfully.")

    # Example: Access and display first profile's data
    if atm_objects:
        print("\n[INFO] Example: Data from the First Atmospheric Profile:")
        first_atm = atm_objects[0]
        print(f"  - Pressure (Pa): {10**first_atm.data_dict['pressure'][:5]} ...")
        print(f"  - Temperature (K): {first_atm.data_dict['temperature'][:5]} ...")
    
        # Step 5: Calculate heating rates and fluxes
        print("\n[5] Calculating Heating Rates and Fluxes for the First Profile...")
        wl_range = config['model_params'].get('wavelength_range', [0.1, 50.0])  # Explicit wavelength range
        heat_rates, net_fluxes, TOA_flux = calculate_heating_rates_and_fluxes(first_atm, wl_range=wl_range)
        if heat_rates is not None:
            print(f"✔ Heating rates and fluxes calculated.")
            print(f"  - Sample Heating Rates: {heat_rates[:3]} ...")
            print(f"  - Sample Net Fluxes: {net_fluxes[:3]} ...")
    
    # Step 6: Visualize PT profiles
    print("\n[6] Visualizing a Subset of PT Profiles...")
    plot_profiles(filename='Data/profiles.h5', num_plots=10)
    print("✔ Visualization saved to Figures/pt_profiles.png")
    
    print("\n" + "="*70)
    print(f"{'PIPELINE COMPLETED SUCCESSFULLY':^70}")
    print("="*70)
