import numpy as np

def sample_constant_or_distribution(param_config):
    """Sample a constant value or a distribution based on the configuration."""
    if param_config['dist'] == 'fixed':
        return param_config['value']
    elif param_config['dist'] == 'uniform':
        return np.random.uniform(param_config['low'], param_config['high'])
    elif param_config['dist'] == 'normal':
        return np.random.normal(param_config['mean'], param_config['std'])
    else:
        raise ValueError(f"Unsupported distribution type: {param_config['dist']}")

def generate_pressure_array(pressure_range):
    """Generate a pressure array based on the specified range."""
    return np.logspace(
        np.log10(pressure_range['min']),
        np.log10(pressure_range['max']),
        num=pressure_range['points']
    )
