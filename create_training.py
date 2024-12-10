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


def gen_profiles(config, P):
    print("\n" + "=" * 70)
    print(f"{'ATMOSPHERIC MODELING PIPELINE':^70}")
    print("=" * 70)

    # Step 1: Clean up old profiles
    print("\nDeleting old profiles...")
    delete_old_profiles(folder='Data/Profiles', base_filename='prof')

    # Step 2: Initialize opacity databases
    print("\nInitializing Opacity Databases...")
    k_db = initialize_opacity_databases(config_file='Inputs/parameters.json')

    # Step 3: Set the stellar spectrum
    print("\nUsing a Stellar Blackbody...")

    # Step 4: Generate and process PT profiles sequentially

    # Initialize the ProfileGenerator
    generator = ProfileGenerator(P=P, config_file='Inputs/parameters.json')

    # Generate and process profiles one at a time
    successful_profiles = 0
    max_attempts = generator.number_of_simulations * 2  # To prevent infinite loops
    attempts = 0

    print("\nCalculating the Profiles...")
    while successful_profiles < generator.number_of_simulations and attempts < max_attempts:
        profile = generator.generate_single_profile()
        if profile is None:
            print("Failed to generate a valid profile, trying again...")
            attempts += 1
            continue

        # Step 5: Calculate opacity structure for the current profile
        atm = calculate_opacity_structure(
            profile=profile,
            k_db=k_db,
            grav=profile.get('grav', 10.0),
            rcp=profile.get('rcp', 0.28),
            albedo_surf=profile.get('albedo_surf', 0.0),
            Rp=profile.get('Rp', 7e6),
            rayleigh=config.get('opacity_params', {}).get('rayleigh', False),
            tstar=profile.get('Tstar', None),
            flux_top_dw=profile.get('flux_surface_down', None)
        )

        if atm is None:
            print(f"Skipping profile {successful_profiles + 1} due to errors.")
            attempts += 1
            continue

        # Step 6: Calculate heating rates and fluxes
        heat_rates, net_fluxes, TOA_flux, flux_up, flux_down= calculate_heating_rates_and_fluxes(
            atm,
            wl_range=config.get('opacity_params', {}).get('wavelength_range', [0.3, 50.0]),
            rayleigh=config.get('opacity_params', {}).get('rayleigh', False))
        if heat_rates is None:
            print(f"Skipping profile {successful_profiles + 1} due to errors in heating rate calculations.")
            attempts += 1
            continue

        # Step 7: Save data for the current profile
        data_to_save = {
            # Assuming log10 pressures
            "pressure": list(10**np.array(atm.data_dict['pressure'])),
            "temperature": list(atm.data_dict['temperature']),
            "Tstar": profile.get('Tstar', None),
            "net_flux": list(net_fluxes),
            "flux_up": list(flux_up),
            "flux_down": list(flux_down),
            "heating_rate": list(heat_rates),
            "orbital_sep": profile.get('orbital_sep', None),
            "flux_surface_down": profile.get('flux_surface_down', None),
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
