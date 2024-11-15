import numpy as np
from opacity_structure import (
    initialize_opacity_databases,
    set_stellar_spectrum,
    calculate_opacity_structure,
    calculate_heating_rates_and_fluxes,
    save_data
)
from profile_generator import ProfileGenerator
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

    # Step 2: Generate PT profiles
    print("\n[2] Generating Pressure-Temperature Profiles...")
    generator = ProfileGenerator(
        N=10,  # Number of profiles to generate
        P=P,
        config_file='Inputs/parameters.json'
    )
    generator.generate_profiles()
    print(f"✔ {len(generator.temperatures)} PT profile(s) generated successfully.")

    # Save profiles to file
    generator.save_profiles(filename='Data/profiles.h5')

    # Step 3: Initialize opacity databases
    print("\n[3] Initializing Opacity Databases...")
    k_db, cia_db, species = initialize_opacity_databases(config_file='Inputs/parameters.json')
    print("✔ Opacity databases initialized.")

    # Step 4: Set the stellar spectrum
    print("\n[4] Loading Stellar Spectrum...")
    stellar_spectrum_file = config['model_params'].get('stellar_spectrum_file', 'stellar_spectra/default_spectrum.dat')
    stellar_spectrum = set_stellar_spectrum(datapath=config['datapath'], filename=stellar_spectrum_file)
    print(f"✔ Stellar spectrum loaded successfully.")

    # Step 5: Calculate opacity structure
    print("\n[5] Calculating Opacity Structure for Atmospheric Profiles...")
    rayleigh = config['model_params'].get('rayleigh', False)  # Explicit rayleigh parameter
    profiles = generator.get_profiles()  # Retrieve generated profiles
    atm_objects = calculate_opacity_structure(
        profiles=profiles,
        k_db=k_db,
        cia_db=cia_db,
        grav=grav,
        rcp=rcp,
        albedo_surf=albedo_surf,
        Rp=Rp,
        rayleigh=rayleigh,
        stellar_spectrum=stellar_spectrum,
        tstar=config['model_params'].get('Tstar', None)
    )
    print(f"✔ {len(atm_objects)} profile(s) processed successfully.")

    # Step 6: Save data for all profiles
    print("\n[6] Saving Data for All Profiles...")
    for i, atm in enumerate(atm_objects):
        print(f"  - Saving data for profile {i + 1}...")
        heat_rates, net_fluxes, TOA_flux = calculate_heating_rates_and_fluxes(
            atm,
            wl_range=config['model_params'].get('wavelength_range', [0.1, 50.0]),
            rayleigh=rayleigh
        )
        if heat_rates is not None:
            data_to_save = {
                "pressure": list(10**np.array(atm.data_dict['pressure'])),
                "temperature": list(atm.data_dict['temperature']),
                "Tstar":config['model_params'].get('Tstar', None),
                "net_fluxes": list(net_fluxes)
            }
            save_data(data_to_save, folder='Data', base_filename=f'prof')

    # Step 7: Visualize PT profiles
    print("\n[7] Visualizing a Subset of PT Profiles...")
    plot_profiles(filename='Data/profiles.h5', num_plots=min(len(generator.temperatures), 10))
    print("✔ Visualization saved to Figures/pt_profiles.png")

    print("\n" + "=" * 70)
    print(f"{'PIPELINE COMPLETED SUCCESSFULLY':^70}")
    print("=" * 70)
