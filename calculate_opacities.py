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

    # Get the user defined params
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Pick your k tables and CIA species
    # datapath = config['datapath']
    # print(datapath)
    print("Using local path, not  specified path for opacities")
    datapath = os.getcwd() + '/Data/Opacities/'

    k_table_files = config['k_table_files']
    cia_species = config['cia_species']

    # Set up paths in exo_k
    xk.Settings().set_mks(True)
    xk.Settings().set_search_path(os.path.join(datapath, 'xsec'), path_type='xtable')
    xk.Settings().set_search_path(os.path.join(datapath, 'corrk'), path_type='ktable')
    xk.Settings().set_search_path(os.path.join(datapath, 'cia'), path_type='cia')

    # Initialize databases
    k_db = xk.Kdatabase({species: os.path.join(datapath, path)
                        for species, path in k_table_files.items()})
    cia_db = xk.CIAdatabase(molecules=cia_species, mks=True)
    cia_db.sample(k_db.wns)

    return k_db, cia_db


def calculate_opacity_structure(profile, k_db, cia_db, grav, rcp, albedo_surf, Rp, rayleigh, tstar, flux_top_dw):
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

    # print(list(profile['logplay']))
    # print(list(profile['tlay']))
    # print(grav)
    # print(Rp)
    # print(rcp)
    # print(albedo_surf)
    # print(tstar)
    # print(flux_top_dw)
    # print(rayleigh)
    # exit()

    try:
        # Initialize atmosphere with all parameters
        # This needs to convert from bar to pa in log
        atm = xk.Atm(
            logplay=profile['logplay'] + 5,
            tlay=profile['tlay'],
            grav=grav,
            Rp=Rp,
            rcp=rcp,
            albedo_surf=albedo_surf,
            composition=profile['composition'],
            Tstar=tstar,
            flux_top_dw=flux_top_dw,
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
