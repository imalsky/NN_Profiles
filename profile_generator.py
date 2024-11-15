import numpy as np
import h5py
import json
import os

class ProfileGenerator:
    def __init__(self, N, P, config_file='Inputs/parameters.json'):
        """
        Parameters:
        - N (int): Number of profiles to generate.
        - P (array): Pressure array in bar.
        - config_file (str): Path to the JSON configuration file with priors and species.
        """
        self.N = N              # Number of profiles to generate
        self.P = P              # Pressure array in bar
        self.load_parameters(config_file)  # Load priors and species from JSON config file
        self.temperatures = []  # List to store temperature profiles
        self.compositions = []  # List to store composition profiles

    def load_parameters(self, config_file):
        """Load priors and species from a JSON configuration file."""
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.priors = config['priors']
        self.species = config['species']

    def sample_parameters(self):
        """Sample parameters based on priors specified."""
        params = {}
        
        # Sample main parameters based on priors
        for key, prior in self.priors.items():
            if prior['dist'] == 'normal':
                value = np.random.normal(prior['mean'], prior['std'])
            elif prior['dist'] == 'uniform':
                value = np.random.uniform(prior['low'], prior['high'])
            else:
                raise ValueError(f"Distribution type {prior['dist']} not supported.")
                
            if 'condition' in prior:
                # Apply the condition specified in the JSON file
                while not eval(f"value {prior['condition']}"):
                    if prior['dist'] == 'normal':
                        value = np.random.normal(prior['mean'], prior['std'])
                    elif prior['dist'] == 'uniform':
                        value = np.random.uniform(prior['low'], prior['high'])
            
            params[key] = value
            if key.startswith('log_'):
                params[key[4:]] = np.exp(value)  # Convert log parameter to linear scale if needed

        # Sample and normalize composition
        comp_values = np.random.dirichlet(np.ones(len(self.species)))
        params['composition'] = dict(zip(self.species, comp_values))
        
        return params

    def compute_profile(self, P, params):
        """Compute a temperature profile using Guillot's model."""
        sqrt3 = np.sqrt(3)
        T_Guillot4 = (3 * params['T_int']**4 / 4) * (2/3 + params['delta'] * P) + \
                     (3 * params['T_eq']**4 / 4) * (2/3 + 1 / (params['gamma'] * sqrt3) +
                     (params['gamma'] / sqrt3 - 1 / (params['gamma'] * sqrt3)) * np.exp(-params['gamma'] * params['delta'] * sqrt3 * P))
        T_Guillot = T_Guillot4**0.25
        T_final = T_Guillot * (1 - params['alpha'] / (1 + P / params['P_trans']))
        return T_final

    def generate_profiles(self):
        """Generate profiles based on sampled parameters and discard profiles with temperatures exceeding 5000 K."""
        for _ in range(self.N):
            params = self.sample_parameters()
            T_profile = self.compute_profile(self.P, params)
            
            # Discard profile if any temperature exceeds 5000 K
            if np.any(T_profile > 5000):
                continue

            self.temperatures.append(T_profile)
            self.compositions.append(params['composition'])
        
        # Convert to arrays for compatibility
        self.temperatures = np.array(self.temperatures)

    def get_profiles(self):
        """Return generated profiles in a format compatible with xk.Atm."""
        profiles = []
        log_P = np.log10(self.P)  # Convert pressure to log scale
        for i in range(len(self.temperatures)):
            profiles.append({
                'logplay': log_P,
                'tlay': self.temperatures[i],
                'composition': self.compositions[i]
            })
        return profiles

    def save_profiles(self, filename='Data/profiles.h5'):
        """Save profiles and pressure data to an HDF5 file."""
        os.makedirs('Data', exist_ok=True)  # Ensure Data directory exists
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('P', data=self.P)
            hf.create_dataset('temperature', data=self.temperatures, compression='gzip', compression_opts=9)
            comp_group = hf.create_group('composition')
            for i, comp in enumerate(self.compositions):
                grp = comp_group.create_group(str(i))
                for species, value in comp.items():
                    grp.create_dataset(species, data=value)
        print(f"Data saved to {filename} with compression.")
