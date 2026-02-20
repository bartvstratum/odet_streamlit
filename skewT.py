import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

import thermo as thrm


def skew_transform(T, p, skew_factor):
    """
    Apply skew-T log-p transform.

    Parameters:
    ----------
    T : float or np.ndarray
        Temperature in Kelvin.
    p : float or np.ndarray
        Pressure in Pa.

    Returns:
    -------
    float or np.ndarray
        Skewed temperature in degrees Celsius.
    """
    Tc = T - thrm.T0
    p_hpa = p / 100.0
    return Tc + skew_factor * (np.log(1000.0) - np.log(p_hpa))


class SkewT_lines:
    def __init__(self, skew_factor=35, ktot=64):
        """
        Calculate static lines of a skew-T diagram.

        Parameters:
        ----------
        skew_factor : float or int
            Skewness factor of diagram.
        ktot : int
            Number of vertical levels in curved lines.

        Returns:
        -------
        None
        """

        self.ktot = ktot
        self.p1 = np.geomspace(105_000, 10_000, self.ktot)  # Full pressure grid.
        self.p2 = np.geomspace(105_000, 50_000, self.ktot)  # Low pressure grid for theta and mixing ratio.

        # Linear version for straight lines.
        self.p1_lin = np.array([105_000, 10_000])
        self.p2_lin = np.array([105_000, 50_000])

        # Start points (temperature in Celcus at 1000 hPa) of static lines.
        self.x0_isotherms      = np.arange(-120, 40.01, 10)
        self.x0_dry_adiabats   = np.arange( -40, 50.01, 10)
        #self.x0_moist_adiabats = np.arange( -0,  25.01, 5)
        self.x0_moist_adiabats = np.array([0, 5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5])
        self.x0_isohumes       = np.arange( -30, 30.01, 10)

        self.skew_factor = skew_factor


    def calc(self):
        """
        Calculate default (static) lines as f(T,p).
        """

        # Isotherms: lines of constant absolute temperature.
        x = self.x0_isotherms + thrm.T0
        T = np.broadcast_to(x[np.newaxis, :], (self.p1_lin.size, len(x)))
        self.isotherms = skew_transform(T, self.p1_lin[:, np.newaxis], self.skew_factor)

        # Dry adiabats: lines of constant potential temperature.
        x = self.x0_dry_adiabats + thrm.T0
        T = x[np.newaxis, :] * thrm.exner(self.p2[:, np.newaxis])
        self.dry_adiabats = skew_transform(T, self.p2[:, np.newaxis], self.skew_factor)

        # Moist adiabats: lines of constant saturated potential temperature.
        x = self.x0_moist_adiabats + thrm.T0
        T = thrm.calc_moist_adiabat(x, self.p1)
        self.moist_adiabats = skew_transform(T, self.p1[:, np.newaxis], self.skew_factor)

        # Isohumes: lines of constant specific humidity.
        x = self.x0_isohumes + thrm.T0
        q_0 = thrm.qsat(x, thrm.p0)
        T = thrm.dewpoint(q_0[np.newaxis, :], self.p2_lin[:, np.newaxis])
        self.isohumes = skew_transform(T, self.p2_lin[:, np.newaxis], self.skew_factor)


class SkewT_mpl:
    def __init__(self, skewt_lines, mode='color'):
        """
        Plot skew-T diagram with Matplotlib.
        """
        self.stl = skewt_lines

        self.p_ticks = np.array([1000, 950, 900, 850, 800, 700, 600, 500, 400, 300, 200, 100]) * 100.0
        self.ylim = (1050e2, 100e2)
        self.xlim = (-40, 50)

        if mode == 'simple':
            self.cT = '0.8'          # Isotherms
            self.cth = '0.8'         # Dry adiabats
            self.cths = '0.8'        # Moist adiabats
            self.cr = '0.8'          # Mixing ratio
            self.lw = 0.5            # Line width
            self.ls = '--'           # Line style
        elif mode == 'color':
            self.cT = '0.7'
            self.cth = 'tab:red'
            self.cths = '0.7'
            self.cr = 'tab:blue'
            self.lw = 0.5
            self.ls = '--'
        else:
            raise Exception('Invalid color mode.')


    def plot(self, figsize=(6,6)):
        """
        Create base plot with default static lines.
        """
        stl = self.stl

        plt.figure(figsize=figsize, layout='constrained')

        # Default lines diagram.
        plt.plot(stl.isotherms,      stl.p1_lin, color=self.cT,   linewidth=self.lw, linestyle=self.ls)
        plt.plot(stl.isohumes,       stl.p2_lin, color=self.cr,   linewidth=self.lw, linestyle=self.ls)
        plt.plot(stl.dry_adiabats,   stl.p2,     color=self.cth,  linewidth=self.lw, linestyle=self.ls)
        plt.plot(stl.moist_adiabats, stl.p1,     color=self.cths, linewidth=self.lw, linestyle=self.ls)

        # Finish diagram.
        plt.yscale('log')
        plt.gca().invert_yaxis()
        plt.yticks(self.p_ticks, (self.p_ticks / 100).astype(int))
        plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())
        plt.grid(axis='y', linewidth=0.5, color='0.5')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.ylabel('Pressure (hPa)')
        plt.xlabel('Temperature (°C)')


    def plot_sounding(self, T, p, ax=None, *args, **kwargs):
        """
        Plot observed or modelled sounding.

        Parameters:
        ----------
        T : ndarray
            Temperature (K)
        p : ndarray
            Pressure (Pa)
        ax : matplotlib axes, optional
            Axes to plot on. Defaults to current axes.
        *args, **kwargs :
            Passed to ax.plot().

        Returns:
        -------
        None
        """

        if ax is None:
            ax = plt.gca()

        ax.plot(skew_transform(T,  p, self.stl.skew_factor), p, *args, **kwargs)
        ax.axhline(p[0], color='k', linewidth=0.5)


    def plot_non_entraining_parcel(self, parcel, ax=None, *args, **kwargs):
        """
        Plot non-entraining parcel

        Parameters:
        ----------
        parcel : dict
            Dict with parcel properties.
        ax : matplotlib axes, optional
            Axes to plot on. Defaults to current axes.
        *args, **kwargs :
            Passed to ax.plot().

        Returns:
        -------
        None
        """

        if ax is None:
            ax = plt.gca()

        ax.plot(skew_transform(parcel['T_isohume'], parcel['p_isohume'], self.stl.skew_factor), parcel['p_isohume'], *args, **kwargs)
        ax.plot(skew_transform(parcel['T_dry'],     parcel['p_dry'],     self.stl.skew_factor), parcel['p_dry'],     *args, **kwargs)
        ax.plot(skew_transform(parcel['T_moist'],   parcel['p_moist'],   self.stl.skew_factor), parcel['p_moist'],   *args, **kwargs)


class SkewT_plotly:
    def __init__(self, skewt_lines):
        self.stl = skewt_lines
        self.p_ticks = np.array([1000, 950, 900, 850, 800, 700, 600, 500, 400, 300, 200, 100]) * 100.0
        self.ylim = (1050e2, 100e2)
        self.xlim = (-40, 50)

    def plot(self, title=''):
        stl = self.stl
        fig = go.Figure()

        line_kw = dict(showlegend=False, hoverinfo='skip')

        # Isotherms.
        for i in range(stl.isotherms.shape[1]):
            fig.add_trace(go.Scatter(
                x=stl.isotherms[:, i], y=stl.p1_lin,
                mode='lines', line=dict(color='rgba(179,179,179,0.7)', width=1.2, dash='4px,2px'), **line_kw))

        # Isohumes.
        for i in range(stl.isohumes.shape[1]):
            fig.add_trace(go.Scatter(
                x=stl.isohumes[:, i], y=stl.p2_lin,
                mode='lines', line=dict(color='rgba(31,119,180,0.7)', width=1.2, dash='4px,2px'), **line_kw))

        # Dry adiabats.
        for i in range(stl.dry_adiabats.shape[1]):
            fig.add_trace(go.Scatter(
                x=stl.dry_adiabats[:, i], y=stl.p2,
                mode='lines', line=dict(color='rgba(214,39,40,0.7)', width=1.2, dash='4px,2px'), **line_kw))

        # Moist adiabats.
        for i in range(stl.moist_adiabats.shape[1]):
            fig.add_trace(go.Scatter(
                x=stl.moist_adiabats[:, i], y=stl.p1,
                mode='lines', line=dict(color='rgba(179,179,179,0.7)', width=1.2, dash='4px,2px'), **line_kw))

        fig.update_yaxes(
            type='log', range=[np.log10(self.ylim[0]), np.log10(self.ylim[1])],
            tickvals=self.p_ticks,
            ticktext=(self.p_ticks / 100).astype(int),
            title='Pressure (hPa)',
            showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.3)',
            minor=dict(showgrid=False),
        )
        fig.update_xaxes(
            range=self.xlim,
            title='Temperature (°C)',
        )
        fig.update_layout(
            title=dict(text=f'<span style="font-weight:normal">{title}</span>', x=0.5, xanchor='center'),
            height=900, width=550,
            margin=dict(l=60, r=20, t=60, b=60),
            plot_bgcolor='white',
        )

        self.fig = fig
        return fig

    def plot_sounding(self, T, p, name='', color='red', width=2, dash=None):
        x = skew_transform(T, p, self.stl.skew_factor)
        line_kw = dict(color=color, width=width)
        if dash is not None:
            line_kw['dash'] = dash
        self.fig.add_trace(go.Scatter(
            x=x, y=p, mode='lines', name=name,
            line=line_kw,
        ))
        return self.fig

    def plot_non_entraining_parcel(self, parcel, name='Parcel', color='black', width=2, dash='4px,2px'):
        sf = self.stl.skew_factor
        for i, (key_T, key_p) in enumerate([('T_isohume', 'p_isohume'), ('T_dry', 'p_dry'), ('T_moist', 'p_moist')]):
            self.fig.add_trace(go.Scatter(
                x=skew_transform(parcel[key_T], parcel[key_p], sf), y=parcel[key_p],
                mode='lines', name=name, line=dict(color=color, width=width, dash=dash),
                showlegend=i == 0,
            ))
        return self.fig

    def plot_entraining_parcel(self, plume, name='Plume', color='black', width=2, dash='4px,2px'):
        sf = self.stl.skew_factor
        line_kw = dict(color=color, width=width, dash=dash)
        self.fig.add_trace(go.Scatter(
            x=skew_transform(plume['T'], plume['p'], sf), y=plume['p'],
            mode='lines', name=name, line=line_kw,
        ))
        below_lcl = plume['type'] == 0
        self.fig.add_trace(go.Scatter(
            x=skew_transform(plume['Td'][below_lcl], plume['p'][below_lcl], sf), y=plume['p'][below_lcl],
            mode='lines', name=name, line=line_kw,
            showlegend=False,
        ))
        return self.fig