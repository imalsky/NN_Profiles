import os
import json
import exo_k as xk
import numpy as np


def initialize_opacity_databases(config_file='Inputs/parameters.json'):
    """
    Initialize k-table and CIA databases for opacity calculations.

    Parameters:
    - config_file (str): Path to the JSON configuration file.

    Returns:
    - k_db (xk.Kdatabase): K-table database.
    - cia_db (xk.CIAdatabase): CIA database.
    - species (list): List of species inferred from the k_table_files and cia_species.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    datapath = config['datapath']
    k_table_files = config['k_table_files']
    cia_species = config['cia_species']
    
    # Set up paths in exo_k
    xk.Settings().set_mks(True)
    xk.Settings().set_search_path(os.path.join(datapath, 'xsec'), path_type='xtable')
    xk.Settings().set_search_path(os.path.join(datapath, 'corrk'), path_type='ktable')
    xk.Settings().set_search_path(os.path.join(datapath, 'cia'), path_type='cia')
    
    # Initialize databases
    k_db = xk.Kdatabase({species: os.path.join(datapath, path) for species, path in k_table_files.items()})
    cia_db = xk.CIAdatabase(molecules=cia_species, mks=True)
    cia_db.sample(k_db.wns)
    
    # Infer species list from k_table_files and cia_species
    species = list(k_table_files.keys()) + cia_species

    return k_db, cia_db, species


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

    #print(f"Stellar spectrum loaded from {spectrum_path}.")

    return stellar_spectrum


def calculate_opacity_structure(profile, k_db, cia_db, grav, rcp, albedo_surf, Rp, rayleigh=True, stellar_spectrum=None, tstar=None):
    """
    Calculate opacity structure for a single atmospheric profile.

    Parameters:
    - profile (dict): Atmospheric profile data.
    - k_db (xk.Kdatabase): K-table database.
    - cia_db (xk.CIAdatabase): CIA database.
    - grav (float): Gravitational acceleration in m/s^2.
    - rcp (float): R/cp value (dimensionless).
    - albedo_surf (float): Surface albedo.
    - Rp (float): Planet radius in meters.
    - rayleigh (bool): Whether to include Rayleigh scattering.
    - stellar_spectrum (xk.Spectrum): Stellar spectrum object.
    - tstar (float): Stellar temperature.

    Returns:
    - atm (xk.Atm): Atmospheric model object with a data_dict attribute.
    """    

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
            #Tstar=tstar,
            #stellar_spectrum=stellar_spectrum,
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
        }

        # Attach the data dictionary to the atm object
        atm.data_dict = data_dict

        return atm

    except Exception as e:
        print(f"Error processing profile: {e}")
        return None

