import numpy as np
import pandas as pd

def parse_sounding(sounding_csv):
    """
    Read single sounding CSV into a Pandas dataframe and calculate
    derived properties (potential temperature, specific humidity,
    dew point, wind components).
    """
    df = pd.read_csv(sounding_csv, parse_dates=['timestamp'], index_col=['timestamp'])
    df['temperature'] += thrm.T0                                     # °C → K
    df['exner'] = thrm.exner(df['pressure'])
    df['theta'] = df['temperature'] / df['exner']
    es = thrm.esat(df['temperature'])
    e = df['relative_humidity'] / 100 * es
    df['qt'] = e * 0.622 / df['pressure']
    df['Td'] = thrm.dewpoint(df['qt'], df['pressure'])
    wind_dir_rad = np.deg2rad(df['heading'])
    df['u'] = df['speed'] * np.sin(wind_dir_rad)
    df['v'] = df['speed'] * np.cos(wind_dir_rad)
    return df
