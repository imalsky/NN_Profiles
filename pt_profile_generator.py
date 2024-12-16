import numpy as np
import json

class ProfileGenerator:
    def __init__(self, P, config_file='Inputs/parameters.json'):
        """
        Parameters:
        - P (array): Pressure array in bar.
        - config_file (str): Path to the JSON configuration file with variables.
        """
        self.number_of_simulations = None
        self.variables = {}
        self.P = P  # Pressure array in bar
        self.load_parameters(config_file)  # Load variables from JSON config file
        self.N = self.number_of_simulations  # Number of profiles to generate

    def load_parameters(self, config_file):
        """Load variables from a JSON configuration file."""
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Extract variables with 'dist' field
        for key, value in config.items():
            if isinstance(value, dict) and 'dist' in value:
                self.variables[key] = value

        # Load number of simulations
        self.number_of_simulations = config.get('number_of_simulations', 10)

    def sample_parameters(self):
        """Sample parameters based on variables specified."""
        params = {}

        # Sample main parameters based on variables
        for key, prior in self.variables.items():
            if prior['dist'] == 'normal':
                value = np.random.normal(prior['mean'], prior['std'])
            elif prior['dist'] == 'uniform':
                value = np.random.uniform(prior['low'], prior['high'])
            elif prior['dist'] == 'fixed':
                value = prior['value']
            else:
                raise ValueError(f"Distribution type {prior['dist']} not supported.")

            if 'condition' in prior:
                # Apply the condition specified in the JSON file
                if prior['dist'] == 'fixed':
                    # Fixed values won't change; check condition once
                    if not eval(f"value {prior['condition']}"):
                        raise ValueError(f"Fixed value {value} for parameter '{key}' does not satisfy condition '{prior['condition']}'.")
                else:
                    while not eval(f"value {prior['condition']}"):
                        if prior['dist'] == 'normal':
                            value = np.random.normal(prior['mean'], prior['std'])
                        elif prior['dist'] == 'uniform':
                            value = np.random.uniform(prior['low'], prior['high'])

            # Handle log parameters
            if key.startswith('log_'):
                linear_key = key[4:]
                linear_value = 10 ** value
                params[linear_key] = linear_value  # Convert log parameter to linear scale

                # Store both log and linear values
                params[key] = value
            else:
                params[key] = value


        return params

    def compute_profile_six_param(self, P, params):
        """Compute a temperature profile using Guillot's model."""
        delta = params.get('delta')
        gamma = params.get('gamma')
        T_int = params.get('T_int')
        T_irr = params.get('T_irr')
        P_trans = params.get('P_trans')
        alpha = params.get('alpha')
        mu = params.get('mu')
        mu = mu + 1e-8
        Rstar = params.get('stellar_radius')
        albedo = params.get('albedo_surf')
        heat_redistribution = params.get('heat_redistribution')
        Tstar = params.get('Tstar')

        Line_2013_beta = ((1 - albedo) / heat_redistribution) ** (0.25)

        T_Guillot4 = (
                     (3 * T_int**4 / 4) * (2/3 + delta * P) +
                     (3 * T_irr**4 * mu / 4) * (2/3 + (mu/gamma) + (gamma / (3 * mu) - (mu/gamma)) * np.exp(-gamma * delta * P / mu))
                     )


        T_Guillot = T_Guillot4**0.25
        T_final = T_Guillot * (1 - alpha / (1 + P / P_trans))

        # Casting the orbital sep as an int gets rid of some floating point precision problems
        #orbital_sep = int(Tstar ** 2 * Rstar / (T_irr ** 2))
        orbital_sep = (Rstar * Line_2013_beta ** 2 / 2) * (Tstar / T_irr) ** 2
 
        # Check this, it doesn't matter a ton, but I'm not sure that I'm getting my heat redistrubution right
        flux_surface_down = (1 * 5.6703e-8 * T_irr ** 4.0) * (1 - albedo) / heat_redistribution
        
        #print(mu, albedo, heat_redistribution, mu * 5.67e-8 * T_irr ** 4 * 1)
        #print(f'{5.67e-8 * T_irr ** 4:.3e}. Tirr: {T_irr:0.2e}')
        #print(f'albedo: {albedo:.2e}, mu: {mu:.2e}, flux_down: {flux_surface_down:.2e}, f: {heat_redistribution:.2e}')


        smoothing = True
        if smoothing:
            beta = 1.5
            s = 0.5 * (1 + np.tanh(beta * (np.log10(P) - params.get('log_homogenous_zone'))))
            T_final = T_final + (params.get('homogenous_temperature') - T_final) * s

        return T_final, orbital_sep, flux_surface_down


    def generate_single_profile(self):
        """
        Generate a single profile based on sampled parameters, ensuring temperatures do not exceed 5000 K.
        Returns:
        - profile (dict): Atmospheric profile data with all parameters and computed values included at the top level.
        """
        max_attempts = 10  # Prevent infinite loops
        attempts = 0
        while attempts < max_attempts:
            # Sample parameters
            params = self.sample_parameters()

            # Compute profile and derived quantities
            T_profile, orbital_sep, flux_surface_down = self.compute_profile_six_param(self.P, params)

            # Discard profile if any temperature exceeds 5000 K
            if np.any(T_profile > 5000):
                attempts += 1
                continue

            # Prepare profile data
            log_P = np.log10(self.P)  # Convert pressure to log scale

            # Add computed values to params
            params['logplay'] = log_P
            params['tlay'] = T_profile
            params['orbital_sep'] = orbital_sep
            params['flux_surface_down'] = flux_surface_down

            # Tstar is already in params if it's a sampled parameter
            if 'Tstar' not in params:
                raise ValueError("Tstar not found in parameters. Ensure it is included in your sampling.")

            # Now, params contains all parameters and computed values
            profile = params

            return profile

        # If failed to generate a valid profile after max_attempts
        print("Failed to generate a valid profile within maximum attempts.")
        return None

