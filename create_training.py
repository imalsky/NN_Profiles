import numpy as np
import gc
from calculate_opacities import (
    initialize_opacity_databases,
    calculate_opacity_structure
)
from calculate_fluxes import (
    calculate_heating_rates_and_fluxes
)
from pt_profile_generator import ProfileGenerator

from utils import (
    delete_old_profiles,
    save_data
)

def gen_profiles(config, P, grav, rcp, albedo_surf, Rp):
    print("\n" + "=" * 70)
    print(f"{'ATMOSPHERIC MODELING PIPELINE':^70}")
    print("=" * 70)

    # Step 1: Clean up old profiles
    print("\nDeleting old profiles...")
    delete_old_profiles(folder='Data/Profiles', base_filename='prof')

    # Step 2: Initialize opacity databases
    print("\nInitializing Opacity Databases...")
    k_db, cia_db, species = initialize_opacity_databases(config_file='Inputs/parameters.json')
    #print("✔ Opacity databases initialized.")

    # Step 3: Set the stellar spectrum
    #print("\n[3] Loading Stellar Spectrum...")
    #stellar_spectrum_file = config['model_params'].get('stellar_spectrum_file', 'stellar_spectra/default_spectrum.dat')
    #stellar_spectrum = set_stellar_spectrum(datapath=config['datapath'], filename=stellar_spectrum_file)
    #print(f"✔ Stellar spectrum loaded successfully.")
    #print("\n[3] Using a Stellar Blackbody...")

    # Step 4: Generate and process PT profiles sequentially
    # Read N from the configuration
    N_profiles = config.get('simulation_params', {}).get('N', 10)  # Default to 10 if not specified

    generator = ProfileGenerator(
        N=N_profiles,  # Use N from the configuration
        P=P,
        config_file='Inputs/parameters.json'
    )

    # Generate and process profiles one at a time
    successful_profiles = 0
    max_attempts = generator.N * 2  # To prevent infinite loops
    attempts = 0

    #print()
    print("\nCalculating the Profiles...")
    while successful_profiles < generator.N and attempts < max_attempts:
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
            stellar_spectrum=None,
            tstar=config['model_params'].get('Tstar', None)
        )
        if atm is None:
            print(f"Skipping profile {successful_profiles + 1} due to errors.")
            attempts += 1
            continue

        # Step 6: Calculate heating rates and fluxes
        heat_rates, net_fluxes, TOA_flux = calculate_heating_rates_and_fluxes(
            atm,
            wl_range=config['model_params'].get('wavelength_range', [0.3, 50.0]),
            rayleigh=config['model_params'].get('rayleigh', False)
        )
        if heat_rates is None:
            print(f"Skipping profile {successful_profiles + 1} due to errors in heating rate calculations.")
            attempts += 1
            continue

        # Step 7: Save data for the current profile
        data_to_save = {
            "pressure": list(10**np.array(atm.data_dict['pressure'])),  # Assuming log10 pressures
            "temperature": list(atm.data_dict['temperature']),
            "Tstar": config['model_params'].get('Tstar', None),
            "net_flux": list(net_fluxes)
        }
        save_data(data_to_save, folder='Data/Profiles', base_filename='prof')

        # Step 8: Clean up to free memory
        del atm
        gc.collect()

        successful_profiles += 1
        attempts += 1

    print("\n" + "=" * 70)
    print(f"{'PIPELINE COMPLETED SUCCESSFULLY':^70}")
    print("=" * 70)
