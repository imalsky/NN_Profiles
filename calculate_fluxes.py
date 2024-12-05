def calculate_heating_rates_and_fluxes(atm, wl_range, rayleigh):
    """
    Calculate heating rates and fluxes for an atmospheric model.

    Parameters:
    - atm (xk.Atm): Atmospheric model object.
    - wl_range (list): Wavelength range for flux calculation (default: [0.1, 50.0] microns).
    - rayleigh (bool): Whether to include Rayleigh scattering in calculations.

    Returns:
    - heat_rates (ndarray): Heating rates array.
    - net_fluxes (ndarray): Net flux array.
    - TOA_flux (ndarray): Top-of-atmosphere flux array.
    """
    try:
        # Compute TOA flux
        # This is with the toon two stream method
        # Eventually I'll prob replace it with a DISORT method
        TOA_flux = atm.emission_spectrum_2stream(integral=True,wl_range=wl_range,rayleigh=rayleigh)
        heat_rates, net_fluxes = atm.heating_rate()

        # Spectrally integrate upward and downward fluxes
        flux_up_integrated = atm.spectral_integration(atm.flux_up_nu)
        flux_down_integrated = atm.spectral_integration(atm.flux_down_nu)

        return heat_rates, net_fluxes, TOA_flux, flux_up_integrated, flux_down_integrated

    except Exception as e:
        print(f"Error calculating heating rates and fluxes: {e}")
        return None, None, None
