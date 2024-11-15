import os
import json
import numpy as np
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
    xk.Settings().set_search_path(os.path.join(datapath, 'corrk'), path_type='ktable')
    xk.Settings().set_search_path(os.path.join(datapath, 'cia'), path_type='cia')
    
    # Initialize databases
    k_db = xk.Kdatabase({species: os.path.join(datapath, path) for species, path in k_table_files.items()})
    cia_db = xk.CIAdatabase(molecules=cia_species, mks=True)
    cia_db.sample(k_db.wns)
    
    print("Opacity databases initialized.")
    return k_db, cia_db

def calculate_opacity_structure(profiles, k_db, cia_db, grav=9.81, rcp=0.28, albedo_surf=0.0, Rp=6371e3):
    """
    Calculate opacity structure for atmospheric profiles and integrate results into the atm object.

    Parameters:
    - profiles (list): List of atmospheric profiles with keys 'logplay', 'tlay', and 'composition'.
    - k_db (xk.Kdatabase): K-table database.
    - cia_db (xk.CIAdatabase): CIA database.
    - grav (float): Gravitational acceleration in m/s^2.
    - rcp (float): R/cp value (dimensionless).
    - albedo_surf (float): Surface albedo.
    - Rp (float): Planet radius in meters.

    Returns:
    - atm_objects (list): List of xk.Atm objects with calculated data stored in a `data_dict` attribute.
    """
    atm_objects = []

    for i, profile in enumerate(profiles):
        try:
            # Initialize atmosphere
            atm = xk.Atm(
                logplay=profile['logplay'],
                tlay=profile['tlay'],
                grav=grav,
                Rp=Rp,
                rcp=rcp,
                albedo_surf=albedo_surf,
                composition=profile['composition'],
                k_database=k_db,
                cia_database=cia_db,
                rayleigh=True
            )
            
            # Compute opacity and emission properties
            atm.setup_emission_caculation()

            # Organize data into a dictionary
            data_dict = {
                'pressure': 10**atm.logplay,  # Convert log pressure back to linear scale
                'temperature': atm.tlay,
                'tau': atm.tau,  # Multidimensional array (optical depth)
                'dtau': atm.dtau,  # Differential optical depth
                'weights': atm.weights if atm.weights is not None else None
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
