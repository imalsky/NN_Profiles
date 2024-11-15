import os
import json
import exo_k as xk


def initialize_opacity_databases(config_file='Inputs/parameters.json'):
    """
    Initialize k-table and CIA databases for opacity calculations.

    Parameters:
    - config_file (str): Path to the JSON configuration file.

    Returns:
    - k_db (xk.Kdatabase): K-table database.
    - cia_db (xk.CIAdatabase): CIA database.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    datapath = config['datapath']
    k_table_files = config['k_table_files']
    cia_species = config['cia_species']
    
    # Set up paths in exo_k
    xk.Settings().set_mks(True)
    xk.Settings().set_search_path(os.path.join(datapath, 'xsec'), path_type='xtable')
    xk.Settings().set_search_path(os.path.join(datapath, 'corrk_big'), path_type='ktable')
    xk.Settings().set_search_path(os.path.join(datapath, 'cia'), path_type='cia')
    
    # Initialize databases
    k_db = xk.Kdatabase({species: os.path.join(datapath, path) for species, path in k_table_files.items()})
    cia_db = xk.CIAdatabase(molecules=cia_species, mks=True)
    cia_db.sample(k_db.wns)
    
    print("Opacity databases initialized.")
    return k_db, cia_db


def set_stellar_spectrum(datapath, filename):
    """
    Set the stellar spectrum.

    Parameters:
    - datapath (str): Path to the data directory.
    - filename (str): Filename of the stellar spectrum file.

    Returns:
    - stellar_spectrum (xk.Spectrum): Stellar spectrum object.
    """
    spectrum_path = os.path.join(datapath, filename)
    stellar_spectrum = xk.Spectrum(
        filename=spectrum_path,
        spectral_radiance=True,
        input_spectral_unit='nm'
    )
    print(f"Stellar spectrum loaded from {spectrum_path}.")
    return stellar_spectrum


def calculate_opacity_structure(profiles, k_db, cia_db, grav, rcp, albedo_surf, Rp, rayleigh=True, stellar_spectrum=None):
    """
    Calculate opacity structure for atmospheric profiles.

    Parameters:
    - profiles (list): List of atmospheric profiles.
    - k_db (xk.Kdatabase): K-table database.
    - cia_db (xk.CIAdatabase): CIA database.
    - grav (float): Gravitational acceleration in m/s^2.
    - rcp (float): R/cp value (dimensionless).
    - albedo_surf (float): Surface albedo.
    - Rp (float): Planet radius in meters.
    - rayleigh (bool): Whether to include Rayleigh scattering.
    - stellar_spectrum (xk.Spectrum): Stellar spectrum object.

    Returns:
    - atm_objects (list): List of xk.Atm objects with a `data_dict` attribute.
    """
    atm_objects = []

    for i, profile in enumerate(profiles):
        try:
            # Initialize atmosphere with all parameters, including stellar spectrum
            atm = xk.Atm(
                logplay=profile['logplay'],
                tlay=profile['tlay'],
                grav=grav,
                Rp=Rp,
                rcp=rcp,
                albedo_surf=albedo_surf,
                composition=profile['composition'],
                Tstar=None,
                stellar_spectrum=stellar_spectrum,
                k_database=k_db,
                cia_database=cia_db,
                rayleigh=rayleigh
            )
            
            # Compute opacity and emission properties
            atm.setup_emission_caculation()

            # Organize results into a data dictionary
            data_dict = {
                'pressure': profile['logplay'],  # Using input profile's data
                'temperature': profile['tlay'],  # Using input profile's data
                # Add other fields as necessary from available attributes
            }

            # Attach the data dictionary to the atm object
            atm.data_dict = data_dict

            # Append the atm object to the list
            atm_objects.append(atm)
            print(f"Processed profile {i+1} successfully.")

        except Exception as e:
            print(f"Error processing profile {i+1}: {e}")
            continue

    return atm_objects





def calculate_heating_rates_and_fluxes(atm, wl_range=[0.1, 50.0]):
    """
    Calculate heating rates and fluxes for an atmospheric model.

    Parameters:
    - atm (xk.Atm): Atmospheric model object.
    - wl_range (list): Wavelength range for flux calculation (default: [0.1, 50.0] microns).

    Returns:
    - heat_rates (ndarray): Heating rates array.
    - net_fluxes (ndarray): Net flux array.
    - TOA_flux (ndarray): Top-of-atmosphere flux array.
    """
    try:
        # Compute TOA flux
        TOA_flux = atm.emission_spectrum_2stream(
            wl_range=wl_range,
            rayleigh=True
        )
        print(f"TOA flux computed for wavelength range {wl_range}.")

        # Compute heating rates and net fluxes
        heat_rates, net_fluxes = atm.heating_rate()
        print("Heating rates and net fluxes calculated.")
        
        return heat_rates, net_fluxes, TOA_flux

    except Exception as e:
        print(f"Error calculating heating rates and fluxes: {e}")
        return None, None, None
