import numpy as np
import json

class ProfileGenerator:
    def __init__(self, N, P, config_file='Inputs/parameters.json'):
        """
        Parameters:
        - N (int): Number of profiles to generate.
        - P (array): Pressure array in bar.
        - config_file (str): Path to the JSON configuration file with priors and composition.
        """
        self.N = N              # Number of profiles to generate
        self.P = P              # Pressure array in bar
        self.load_parameters(config_file)  # Load priors and composition from JSON config file

    def load_parameters(self, config_file):
        """Load priors and fixed composition from a JSON configuration file."""
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.priors = config['priors']

        # Load fixed composition from JSON
        self.fixed_composition = config.get('composition', {})

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

        # Use the fixed composition from the configuration
        params['composition'] = self.fixed_composition
        
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

    def generate_single_profile(self):
        """
        Generate a single profile based on sampled parameters, ensuring temperatures do not exceed 5000 K.
        Returns:
        - profile (dict): Atmospheric profile data compatible with xk.Atm.
        """
        max_attempts = 10  # Prevent infinite loops
        attempts = 0
        while attempts < max_attempts:
            params = self.sample_parameters()
            T_profile = self.compute_profile(self.P, params)
            
            # Discard profile if any temperature exceeds 5000 K
            if np.any(T_profile > 5000):
                attempts += 1
                continue

            # Prepare profile data
            log_P = np.log10(self.P)  # Convert pressure to log scale

            profile = {
                'logplay': log_P,
                'tlay': T_profile,
                'composition': params['composition']
            }

            return profile

        # If failed to generate a valid profile after max_attempts
        print("Failed to generate a valid profile within maximum attempts.")
        return None
