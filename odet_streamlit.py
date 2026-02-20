import streamlit as st
import numpy as np
import pandas as pd
from datetime import date

import open_meteo
import parcel as prcl
import skewT as skt
import thermo as thrm
from helpers import parse_sounding

@st.cache_resource
def get_skewt_lines():
    stl = skt.SkewT_lines()
    stl.calc()
    return stl

st.html("""
    <style>
        .stMainBlockContainer {
            max-width:80rem;
        }
    </style>
    """
)

# Page config.
st.set_page_config(page_title='ODET sounding analysis', layout='wide')
st.title('🌳🔥🌲 | ODET sounding analysis')

# Load cases.
cases = pd.read_csv('cases.csv', parse_dates=['date'])

# Sidebar inputs.
with st.sidebar:
    st.header('Case')
    case_names = ['Custom'] + cases['name'].tolist()
    selected_case = st.selectbox('Select case', case_names, index=case_names.index('Pont de Vilomara'))

    # Reset parcel controls when case changes.
    if 'prev_case' not in st.session_state:
        st.session_state.prev_case = selected_case
    if selected_case != st.session_state.prev_case:
        st.session_state.prev_case = selected_case
        st.session_state.launch_parcel = False
        st.session_state.deltaT = 0.0
        st.session_state.deltaTd = 0.0

    if selected_case != 'Custom':
        row = cases[cases['name'] == selected_case].iloc[0]
        default_lat = float(row['lat'])
        default_lon = float(row['lon'])
        default_date = row['date'].date()
    else:
        default_lat = 51.986
        default_lon = 5.666
        default_date = date.today()

    st.header('Location & date')
    lat = st.number_input('Latitude (°N)', value=default_lat, min_value=-90.0, max_value=90.0, step=0.1, format='%.2f')
    lon = st.number_input('Longitude (°E)', value=default_lon, min_value=-180.0, max_value=180.0, step=0.1, format='%.2f')
    sel_date = st.date_input('Date', value=default_date)

    st.header('Model')
    models = {
        'best_match': 'Best match',
        'ecmwf_ifs025': 'ECMWF IFS 9 km',
        'ecmwf_aifs025_single': 'ECMWF AIFS',
        'icon_seamless': 'DWD ICON seamless',
        'metno_seamless': 'MET Nordic',
        'gfs_seamless': 'NOAA GFS seamless',
        'gem_seamless': 'CWS GEM seamless',
        'meteofrance_seamless': 'MeteoFrance seamless',
        'ukmo_seamless': 'UKMO seamless',
    }
    model_keys = list(models.keys())
    model = st.selectbox('Model', model_keys, index=0, format_func=lambda k: models[k])

    fetch = st.button('Fetch & plot', type='primary', width='stretch')

    st.header('Parcel control')
    launch_parcel = st.checkbox('Launch parcel', key='launch_parcel', value=False)
    if launch_parcel:
        parcel_type = st.radio('Parcel type', ['Non-entraining', 'Entraining'], horizontal=True)
        deltaT = st.slider('ΔT (K)', min_value=-5.0, max_value=20.0, value=0.0, step=0.5, key='deltaT')
        deltaq = st.slider('Δq (g/kg)', min_value=0.0, max_value=10.0, value=0.0, step=0.1, key='deltaq') * 1e-3
        if parcel_type == 'Entraining':
            area_plume = st.slider('Fire area (km²)', min_value=0.1, max_value=10.0, value=0.3, step=0.1, key='area_plume') * 1e6

    st.header('Sounding')
    uploaded_file = st.file_uploader('Upload sounding CSV', type='csv')

# Fetch model data.
if 'meteo' not in st.session_state:
    st.session_state.meteo = None

if fetch:
    with st.spinner('Fetching sounding data...'):
        date_str = sel_date.strftime('%Y-%m-%d')
        st.session_state.meteo = open_meteo.get_sounding(lat, lon, model, date_str)
        st.session_state.model = model

# Parse uploaded sounding.
sounding_df = None
if uploaded_file is not None:
    sounding_df = parse_sounding(uploaded_file)

# Main panel.
has_meteo = st.session_state.meteo is not None
has_sounding = sounding_df is not None

if has_meteo or has_sounding:
    if not launch_parcel:
        deltaT = 0.0
        deltaTd = 0.0

    if has_meteo:
        meteo = st.session_state.meteo
        n_times = meteo.sizes['time']
        _, col_slider, _ = st.columns([1, 2, 1])
        with col_slider:
            t = st.slider('Hour (UTC)', min_value=0, max_value=n_times - 1, value=min(12, n_times - 1))
        T = meteo['T'].isel(time=t).values
        Td = meteo['Td'].isel(time=t).values
        p = meteo['p'].values

    # Build title.
    if has_meteo:
        model_label = models.get(st.session_state.model, st.session_state.model)
        title = f'{model_label} | {sel_date} {t:02d}:00 UTC | {lat:.2f}°N {lon:.2f}°E'
    else:
        ts = sounding_df.index[0]
        title = f'Sounding | {ts.strftime("%Y-%m-%d %H:%M")} UTC'

    # Plot skew-T.
    skew = skt.SkewT_plotly(get_skewt_lines())
    skew.plot(title=title)

    if has_meteo:
        skew.plot_sounding(T, p, name='T (model)', color='red')
        skew.plot_sounding(Td, p, name='Td (model)', color='blue')

    if has_sounding:
        skew.plot_sounding(sounding_df['temperature'].values, sounding_df['pressure'].values,
                           name='T (obs)', color='red', dash='4px,2px')
        skew.plot_sounding(sounding_df['Td'].values, sounding_df['pressure'].values,
                           name='Td (obs)', color='blue', dash='4px,2px')

    if launch_parcel:
        if has_meteo:
            T_sfc, Td_sfc, p_sfc = T[0], Td[0], p[0]
            p_fine = np.geomspace(p[0], p[-1], 128)
        else:
            T_sfc  = sounding_df['temperature'].iloc[0]
            Td_sfc = sounding_df['Td'].iloc[0]
            p_sfc  = sounding_df['pressure'].iloc[0]
            p_fine = np.geomspace(p_sfc, 100e2, 128)

        if parcel_type == 'Non-entraining':
            q_sfc = thrm.qsat(Td_sfc, p_sfc)
            Td_sfc_new = thrm.dewpoint(q_sfc + deltaq, p_sfc)
            parcel = prcl.calc_non_entraining_parcel(T_sfc + deltaT, Td_sfc_new, p_sfc, p_fine)
            skew.plot_non_entraining_parcel(parcel)
        else:
            plume = prcl.calc_entraining_parcel(
                meteo['z_agl'][t,:].values,
                meteo['theta'][t,:].values,
                meteo['thetav'][t,:].values,
                meteo['qt'][t,:].values,
                p,
                dtheta_plume_s=deltaT,
                dq_plume_s=deltaq,
                area_plume_s=area_plume,
            )
            skew.plot_entraining_parcel(plume)

    st.plotly_chart(skew.fig, width='stretch')
else:
    st.info('Set location and time in the sidebar, then click **Fetch & plot**, or upload a sounding CSV.')
