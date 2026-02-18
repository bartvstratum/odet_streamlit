from retry_requests import retry
from datetime import datetime

import pandas as pd
import numpy as np
import requests_cache

import openmeteo_requests
import thermo as thrm


def get_meteo(lat, lon, model, pressure_lev_vars, pressure_levs, single_lev_vars, start=None, end=None, forecast_days=None):

    if (start is not None or end is not None) and forecast_days is not None:
        raise Exception('Provide either `start` + `end` for historical data, or `forecast_days` for forecasts.')

    if forecast_days is not None:
        url = 'https://api.open-meteo.com/v1/forecast'
        forecast = True
    elif start is not None and end is not None:
        url = 'https://historical-forecast-api.open-meteo.com/v1/forecast'
        forecast = False

    n_press_vars = len(pressure_lev_vars)
    n_press_levs = len(pressure_levs)

    # Populate single list of variables.
    variables = []
    for var in pressure_lev_vars:
        for lev in pressure_levs:
            variables.append(f'{var}_{lev}hPa')
    variables += single_lev_vars

    # Setup the Open-Meteo API client with cache and retry on error.
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": variables,
        "models": [model],
        "wind_speed_unit": "ms",
    }

    if forecast:
        params['forecast_days'] = forecast_days
    else:
        params['start_date'] = start
        params['end_date'] = end

    response = openmeteo.weather_api(url, params=params)[0]
    hourly = response.Hourly()

    # Poll first response to get number of times.
    n_times = hourly.Variables(0).ValuesAsNumpy().size

    # Gather data in 2D (time, level) and 1D (time) arrays.
    data = {}
    for var in pressure_lev_vars:
        data[var] = np.zeros((n_times, n_press_levs), dtype=np.float32)
    
    for var in single_lev_vars:
        data[var] = np.zeros(n_times, dtype=np.float32)

    for i,var in enumerate(pressure_lev_vars):
        for j in range(n_press_levs):
            ij = j + i*n_press_levs
            data[var][:,j] = hourly.Variables(ij).ValuesAsNumpy()

    ij0 = ij+1
    for i,var in enumerate(single_lev_vars):
        data[var][:] = hourly.Variables(ij0+i).ValuesAsNumpy()

    # Conversions.
    for key in data.keys():
        if 'temperature' in key or 'dew_point' in key:
            data[key] += 273.15

    return data


def get_sounding(lat, lon, model, date_str):
    pressure_levs = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30]
    pressure_lev_vars = ['temperature', 'relative_humidity', 'wind_speed', 'wind_direction', 'geopotential_height']
    single_lev_vars = ['temperature_2m', 'dew_point_2m', 'precipitation', 'rain', 'showers', 'surface_pressure', 'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m']

    meteo = get_meteo(lat, lon, model, pressure_lev_vars, pressure_levs, single_lev_vars, start=date_str, end=date_str)

    p = np.array(pressure_levs) * 100.0

    # Derive dewpoint from relative humidity.
    rh = meteo['relative_humidity'] / 100.0
    q = rh * thrm.qsat(meteo['temperature'], p[np.newaxis, :])

    # Add to data dictionary.
    meteo['dew_point'] = thrm.dewpoint(q, p[np.newaxis, :])
    meteo['specific_humidity'] = q
    meteo['p'] = p
    meteo['n_times'] = meteo['temperature'].shape[0]
    meteo['pressure_levs'] = pressure_levs

    return meteo