import numpy as np
import os
import gc
from calculate_profiles import (
    initialize_opacity_databases,
    set_stellar_spectrum,
    calculate_opacity_structure,
    calculate_heating_rates_and_fluxes,
    save_data
)
from pt_profile_generator import ProfileGenerator
from visualize import plot_profiles
from utils import (
    load_config,
    create_directories,
    sample_constant_or_distribution,
    delete_old_profiles
)

# Load configuration
config = load_config()

# Ensure required directories exist
create_directories('Inputs', 'Data', 'Figures')

# Generate the pressure array
pressure_range = config['pressure_range']
P = np.logspace(
    np.log10(pressure_range['min']),
    np.log10(pressure_range['max']),
    num=pressure_range['points']
)

# Sample planet parameters
grav = sample_constant_or_distribution(config['planet_params']['grav'])
rcp = sample_constant_or_distribution(config['planet_params']['rcp'])
albedo_surf = sample_constant_or_distribution(config['planet_params']['albedo_surf'])
Rp = sample_constant_or_distribution(config['planet_params']['Rp'])

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print(f"{'ATMOSPHERIC MODELING PIPELINE':^70}")
    print("=" * 70)

    # Step 1: Clean up old profiles
    print("\n[1] Deleting old profiles...")
    delete_old_profiles(folder='Data', base_filename='prof')

    # Step 2: Initialize opacity databases
    print("\n[2] Initializing Opacity Databases...")
    k_db, cia_db, species = initialize_opacity_databases(config_file='Inputs/parameters.json')
    print("✔ Opacity databases initialized.")

    # Step 3: Set the stellar spectrum
    print("\n[3] Loading Stellar Spectrum...")
    stellar_spectrum_file = config['model_params'].get('stellar_spectrum_file', 'stellar_spectra/default_spectrum.dat')
    stellar_spectrum = set_stellar_spectrum(datapath=config['datapath'], filename=stellar_spectrum_file)
    print(f"✔ Stellar spectrum loaded successfully.")

    # Step 4: Generate and process PT profiles sequentially
    print("\n[4] Generating and Processing Pressure-Temperature Profiles...")
    generator = ProfileGenerator(
        N=1000,  # Number of profiles to generate
        P=P,
        config_file='Inputs/parameters.json'
    )

    # Generate and process profiles one at a time
    successful_profiles = 0
    max_attempts = generator.N * 2  # To prevent infinite loops
    attempts = 0

    while successful_profiles < generator.N and attempts < max_attempts:
        print(f"\nProcessing profile {successful_profiles + 1} of {generator.N}...")
        profile = generator.generate_single_profile()
        if profile is None:
            print("Failed to generate a valid profile, trying again...")
            attempts += 1
            continue

        # Step 5: Calculate opacity structure for the current profile
        atm = calculate_opacity_structure(
            profile=profile,
            k_db=k_db,
            cia_db=cia_db,
            grav=grav,
            rcp=rcp,
            albedo_surf=albedo_surf,
            Rp=Rp,
            rayleigh=config['model_params'].get('rayleigh', False),
            stellar_spectrum=stellar_spectrum,
            tstar=config['model_params'].get('Tstar', None)
        )
        if atm is None:
            print(f"Skipping profile {successful_profiles + 1} due to errors.")
            attempts += 1
            continue

        # Step 6: Calculate heating rates and fluxes
        heat_rates, net_fluxes, TOA_flux = calculate_heating_rates_and_fluxes(
            atm,
            wl_range=config['model_params'].get('wavelength_range', [0.1, 50.0]),
            rayleigh=config['model_params'].get('rayleigh', False)
        )
        if heat_rates is None:
            print(f"Skipping profile {successful_profiles + 1} due to errors in heating rate calculations.")
            attempts += 1
            continue

        # Step 7: Save data for the current profile
        data_to_save = {
            "pressure": list(10**np.array(atm.data_dict['pressure'])),
            "temperature": list(atm.data_dict['temperature']),
            "Tstar": config['model_params'].get('Tstar', None),
            "flux_up_bol": list(net_fluxes)
        }
        save_data(data_to_save, folder='Data', base_filename='prof')

        # Step 8: Clean up to free memory
        del atm
        gc.collect()

        successful_profiles += 1
        attempts += 1

    # Step 9: Visualize PT profiles (optional)
    print("\n[5] Visualizing PT Profiles...")
    plot_profiles(folder='Data', base_filename='prof', num_plots=min(successful_profiles, 10))

    print("\n" + "=" * 70)
    print(f"{'PIPELINE COMPLETED SUCCESSFULLY':^70}")
    print("=" * 70)
