import numpy as np
import thermo as thrm
from microhhpy.thermo import sat_adjust


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
        Convergence tolerance in Pa (default 5 Pa).

    Returns:
    -------
    p_lcl : float
        LCL pressure in Pa.
    T_lcl : float
        LCL temperature in Kelvin.
    """
    theta_sfc = T_sfc / thrm.exner(p_sfc)
    q_sfc = thrm.qsat(Td_sfc, p_sfc)

    def residual(p):
        return theta_sfc * thrm.exner(p) - thrm.dewpoint(q_sfc, p)

    p_lo, p_hi = 500e2, p_sfc
    while (p_hi - p_lo) > tol:
        p_mid = 0.5 * (p_lo + p_hi)
        if residual(p_mid) > 0:
            p_hi = p_mid
        else:
            p_lo = p_mid

    p_lcl = 0.5 * (p_lo + p_hi)
    T_lcl = theta_sfc * thrm.exner(p_lcl)

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
    theta_sfc = T_sfc / thrm.exner(p_sfc)
    q_sfc = thrm.qsat(Td_sfc, p_sfc)
    p_lcl, T_lcl = find_lcl(T_sfc, Td_sfc, p_sfc)

    # Below LCL: isohume and dry adiabat.
    dry_idx = np.where(p >= p_lcl)[0]
    p_dry = np.append(p[dry_idx], p_lcl)
    T_dry = theta_sfc * thrm.exner(p_dry)

    p_isohume = p_dry.copy()
    T_isohume = thrm.dewpoint(q_sfc, p_isohume)

    # Above LCL: moist adiabat.
    moist_idx = np.where(p < p_lcl)[0]
    p_moist = np.concatenate(([p_lcl], p[moist_idx]))
    T_moist = thrm.calc_moist_adiabat(np.array([T_lcl]), p_moist)[:, 0]

    return dict(
        T_isohume=T_isohume,
        p_isohume=p_isohume,
        T_dry=T_dry,
        p_dry=p_dry,
        T_moist=T_moist,
        p_moist=p_moist,
    )


def calc_entraining_parcel(
        z_env,
        theta_env,
        thetav_env,
        qt_env,
        p_env,
        dtheta_plume_s,
        dq_plume_s,
        area_plume_s,
        fire_multiplier=1,
        a_w=1.0,
        b_w=0.2,
        fac_ent=0.8,
        beta=0.4,
        dz=1,
        z_max=5000,
        float_type=np.float32):
    """
    Compute an entraining parcel ascent through a given environment.

    Uses a Morton entrainment formulation and forward Euler time integration.
    Based loosely on Rio et al. (2010).

    Parameters
    ----------
    z_env : np.ndarray
        Environmental height above ground level [m].
    theta_env : np.ndarray
        Environmental potential temperature [K].
    thetav_env : np.ndarray
        Environmental virtual potential temperature [K].
    qt_env : np.ndarray
        Environmental total-water specific humidity [kg/kg].
    p_env : np.ndarray
        Environmental pressure [Pa].
    dtheta_plume_s : float
        Surface parcel potential temperature excess [K].
    dq_plume_s : float
        Surface parcel moisture excess [kg/kg].
    area_plume_s : float
        Initial parcel area [m^2].
    fire_multiplier : float, optional
        Scales the surface parcel perturbations (default 1).
    a_w : float, optional
        Buoyancy scaling factor in the vertical velocity equation (default 1.0).
    b_w : float, optional
        Drag scaling factor in the vertical velocity equation (default 0.2).
    fac_ent : float, optional
        Multiplier on the Morton entrainment rate (default 0.8).
    beta : float, optional
        Ratio of fractional entrainment to detrainment (default 0.4).
    dz : float, optional
        Vertical grid spacing [m] (default 1).
    z_max : float, optional
        Maximum height of the output grid [m] (default 5000).
    float_type : dtype, optional
        Numpy float type for internal arrays (default np.float32).

    Returns
    -------
    dict with parcel properties.
    """

    z = np.arange(0, z_max, dz)

    """
    Interpolate environment to parcel grid.
    """
    theta_env = np.interp(z, z_env, theta_env).astype(float_type)
    thetav_env = np.interp(z, z_env, thetav_env).astype(float_type)
    qt_env = np.interp(z, z_env, qt_env).astype(float_type)
    p_env = np.interp(z, z_env, p_env).astype(float_type)
    exner_env = thrm.exner(p_env)
    rho_env = p_env / (thrm.Rd * exner_env * thetav_env)

    """
    Parcel properties.
    """
    theta_plume = np.zeros_like(z, dtype=float_type)
    qt_plume = np.zeros_like(z, dtype=float_type)
    thetav_plume = np.zeros_like(z, dtype=float_type)
    T_plume = np.zeros_like(z, dtype=float_type)
    Tv_plume = np.zeros_like(z, dtype=float_type)
    area_plume = np.zeros_like(z, dtype=float_type)
    w_plume = np.zeros_like(z, dtype=float_type)
    mass_flux_plume = np.zeros_like(z, dtype=float_type)
    entrainment_plume = np.zeros_like(z, dtype=float_type)
    detrainment_plume = np.zeros_like(z, dtype=float_type)
    type_plume = np.zeros_like(z, dtype=np.int8)

    """
    Initial conditions.
    """
    theta_plume[0] = theta_env[0] + fire_multiplier * dtheta_plume_s
    qt_plume[0] = qt_env[0] + fire_multiplier * dq_plume_s

    T_plume[0], ql, qi, qs = sat_adjust(theta_plume[0], qt_plume[0], p_env[0], use_ice=False)
    thetav_plume[0] = thrm.virtual_temp(theta_plume[0], qt_plume[0], ql, qi)
    Tv_plume[0] = thrm.virtual_temp(T_plume[0], qt_plume[0], ql, qi)

    area_plume[0] = area_plume_s
    w_plume[0] = 0.1
    mass_flux_plume[0] = rho_env[0] * area_plume[0] * w_plume[0]

    """
    Entrainment settings.
    """
    epsi = fac_ent * beta / np.sqrt(area_plume[0])  # Morton formulation
    delt = epsi / beta

    entrainment_plume[0] = epsi * mass_flux_plume[0]
    detrainment_plume[0] = 0.0

    """
    Launch parcel!
    """
    for i in range(1, len(z)):
        mass_flux_plume[i] = mass_flux_plume[i-1] + (entrainment_plume[i-1] - detrainment_plume[i-1]) * dz
        theta_plume[i] = theta_plume[i-1] - entrainment_plume[i-1] * (theta_plume[i-1] - theta_env[i-1]) / mass_flux_plume[i-1] * dz
        qt_plume[i] = qt_plume[i-1] - entrainment_plume[i-1] * (qt_plume[i-1] - qt_env[i-1]) / mass_flux_plume[i-1] * dz

        T_plume[i], ql, qi, qs = sat_adjust(theta_plume[i], qt_plume[i], p_env[i], use_ice=False)
        thetav_plume[i] = thrm.virtual_temp(theta_plume[i], qt_plume[i], ql, qi)
        Tv_plume[i] = thrm.virtual_temp(T_plume[i], qt_plume[i], ql, qi)

        if ql > 0 or qi > 0:
            type_plume[i] = 1

        buoy_m = thrm.g / thetav_env[i-1] * (thetav_plume[i-1] - thetav_env[i-1])
        w_plume[i] = (max(0, w_plume[i-1]**2 + 2*(a_w * buoy_m - b_w * epsi * w_plume[i-1]**2) * dz))**0.5

        entrainment_plume[i] = epsi * mass_flux_plume[i]
        detrainment_plume[i] = delt * mass_flux_plume[i]

        w_eps = 1e-6
        area_plume[i] = mass_flux_plume[i] / (rho_env[i] * (w_plume[i] + w_eps))

        if (area_plume[i] <= 0) or (w_plume[i] < w_eps):
            break

    return dict(
        T=T_plume[:i],
        Tv=Tv_plume[:i],
        Td=thrm.dewpoint(qt_plume[:i], p_env[:i]),
        theta=theta_plume[:i],
        thetav=thetav_plume[:i],
        qt=qt_plume[:i],
        area=area_plume[:i],
        w=w_plume[:i],
        mass_flux=mass_flux_plume[:i],
        entrainment=entrainment_plume[:i],
        detrainment=detrainment_plume[:i],
        type=type_plume[:i],
        z=z[:i],
        p=p_env[:i],
    )
