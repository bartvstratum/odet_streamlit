import numpy as np

# Constants
Rd  = 287.04
Rv  = 461.5
cp  = 1005.0
Lv  = 2.5e6
p0  = 1e5
eps = Rd / Rv
e0  = 611.2
a   = 17.67
b   = 243.5
T0  = 273.15

# Thermo functions.
def esat(T):
    """
    Compute saturation vapor pressure using the Bolton (1980) formula.

    Parameters:
    ----------
    T : float or np.ndarray
        Temperature in Kelvin.

    Returns:
    -------
    float or np.ndarray
        Saturation vapor pressure in Pa.
    """
    Tc = T - T0
    return e0 * np.exp(a * Tc / (Tc + b))


def qsat(T, p):
    """
    Compute saturation specific humidity from temperature and pressure.

    Parameters:
    ----------
    T : float or np.ndarray
        Temperature in Kelvin.
    p : float or np.ndarray
        Pressure in Pa.

    Returns:
    -------
    float or np.ndarray
        Saturation specific humidity in kg/kg.
    """
    es = esat(T)
    return eps * es / (p - (1.0 - eps) * es)


def esat_from_q(q, p):
    """
    Compute saturation vapor pressure from specific humidity and pressure.

    Parameters:
    ----------
    q : float or np.ndarray
        Specific humidity in kg/kg.
    p : float or np.ndarray
        Pressure in Pa.

    Returns:
    -------
    float or np.ndarray
        Saturation vapor pressure in Pa.
    """
    return q * p / (eps + (1.0 - eps) * q)


def dewpoint(q, p):
    """
    Compute dew-point temperature from specific humidity and pressure,
    by inverting the Bolton (1980) saturation vapor pressure formula.

    Parameters:
    ----------
    q : float or np.ndarray
        Specific humidity in kg/kg.
    p : float or np.ndarray
        Pressure in Pa.

    Returns:
    -------
    float or np.ndarray
        Dew-point temperature in Kelvin.
    """
    es  = esat_from_q(q, p)
    lnr = np.log(es / e0)
    Tc  = b * lnr / (a - lnr)
    return Tc + T0


def exner(p):
    """
    Compute the Exner function pi = (p/p0)**(Rd/cp).

    Parameters:
    ----------
    p : float or np.ndarray
        Pressure in Pa.

    Returns:
    -------
    float or np.ndarray
        Dimensionless Exner function.
    """
    return (p / p0) ** (Rd / cp)


def dTdp(T, p):
    """
    Compute the lapse rate dT/dp for a saturated parcel.

    Derived by combining the first law of thermodynamics for a saturated
    parcel (cp*dT - R*T/p*dp + Lv*dqs = 0) with the Clausius-Clapeyron
    equation to express dqs in terms of dT.

    References:
    ----------
    - Emanuel, K. A. (1994). Atmospheric Convection, Oxford University Press.

    Parameters:
    ----------
    T : float or np.ndarray
        Temperature in Kelvin.
    p : float or np.ndarray
        Pressure in Pa.

    Returns:
    -------
    float or np.ndarray
        Temperature tendency with respect to pressure in K/Pa.
    """
    qs = qsat(T, p)
    return (T / p) * (Rd + Lv * qs / (Rd * T)) \
                    / (cp + Lv**2 * qs / (Rv * T**2))


def calc_moist_adiabat(T_start, p):
    """
    Integrate the pseudoadiabatic lapse rate upward in pressure using
    the MicroHH RK3 scheme (Williamson 1980, low-storage).

    Parameters:
    ----------
    T_start : np.ndarray, shape (n_lines,)
        Starting temperatures at the lowest pressure level in Kelvin.
    p : np.ndarray, shape (ktot,)
        Pressure levels in Pa (decreasing, i.e. bottom to top).

    Returns:
    -------
    np.ndarray, shape (ktot, n_lines)
        Temperature along moist adiabats in Kelvin.
    """
    cA = np.array([0., -5./9., -153./128.])
    cB = np.array([1./3., 15./16., 8./15.])

    ktot = p.size
    T_out = np.empty((ktot, T_start.size))
    T_out[0, :] = T_start

    for k in range(1, ktot):
        dp = p[k] - p[k-1]
        T = T_out[k-1, :].copy()
        Tt = 0.0

        for s in range(3):
            Tt = cA[s] * Tt + dTdp(T, p[k-1])
            T = T + cB[s] * dp * Tt

        T_out[k, :] = T

    return T_out


def find_lcl(T_sfc, Td_sfc, p_sfc, tol=5):
    """
    Find the Lifting Condensation Level (LCL) using bisection.

    Searches for the pressure where the dry adiabat temperature
    equals the dewpoint at constant mixing ratio.

    Parameters:
    ----------
    T_sfc : float
        Surface temperature in Kelvin.
    Td_sfc : float
        Surface dew-point temperature in Kelvin.
    p_sfc : float
        Surface pressure in Pa.
    tol : float
        Convergence tolerance in Pa (default 1 Pa).

    Returns:
    -------
    p_lcl : float
        LCL pressure in Pa.
    T_lcl : float
        LCL temperature in Kelvin.
    """
    theta_sfc = T_sfc / exner(p_sfc)
    q_sfc = qsat(Td_sfc, p_sfc)

    def residual(p):
        return theta_sfc * exner(p) - dewpoint(q_sfc, p)

    p_lo, p_hi = 500e2, p_sfc
    while (p_hi - p_lo) > tol:
        p_mid = 0.5 * (p_lo + p_hi)
        if residual(p_mid) > 0:
            p_hi = p_mid
        else:
            p_lo = p_mid

    p_lcl = 0.5 * (p_lo + p_hi)
    T_lcl = theta_sfc * exner(p_lcl)

    return p_lcl, T_lcl


def calc_non_entraining_parcel(T_sfc, Td_sfc, p_sfc, p):
    """
    Compute the three parcel ascent lines: isohume, dry adiabat,
    and moist adiabat.

    Parameters:
    ----------
    T_sfc : float
        Surface temperature in Kelvin.
    Td_sfc : float
        Surface dew-point temperature in Kelvin.
    p_sfc : float
        Surface pressure in Pa.
    p : np.ndarray
        Pressure levels in Pa (decreasing, i.e. bottom to top).

    Returns:
    -------
    dict with keys:
        'T_isohume', 'p_isohume' : Dew-point line (constant mixing ratio), surface to LCL.
        'T_dry', 'p_dry'         : Dry adiabat, surface to LCL.
        'T_moist', 'p_moist'     : Moist adiabat, LCL upward.
    """
    theta_sfc = T_sfc / exner(p_sfc)
    q_sfc = qsat(Td_sfc, p_sfc)
    p_lcl, T_lcl = find_lcl(T_sfc, Td_sfc, p_sfc)

    # Below LCL: isohume and dry adiabat.
    dry_idx = np.where(p >= p_lcl)[0]
    p_dry = np.append(p[dry_idx], p_lcl)
    T_dry = theta_sfc * exner(p_dry)

    p_isohume = p_dry.copy()
    T_isohume = dewpoint(q_sfc, p_isohume)

    # Above LCL: moist adiabat.
    moist_idx = np.where(p < p_lcl)[0]
    p_moist = np.concatenate(([p_lcl], p[moist_idx]))
    T_moist = calc_moist_adiabat(np.array([T_lcl]), p_moist)[:, 0]

    return dict(
        T_isohume=T_isohume,
        p_isohume=p_isohume,
        T_dry=T_dry,
        p_dry=p_dry,
        T_moist=T_moist,
        p_moist=p_moist,
    )