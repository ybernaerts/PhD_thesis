import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns
import numpy as np

from scipy import integrate

import ephys_extractor as efex
import ephys_features as ft

from ephys_utils import get_time_voltage_current_currindex0, extract_spike_features, get_cell_features

def prel_inspection(data, axis = None):
    """ Plots the voltage traces for some current steps meant for quality checks and preliminary inspection.
    Parameters
    ----------
    data : data full of voltage (V) and time (s) for a particular cell
    axis : axis you'd like to plot information on (optional, None by default)
        
    Returns
    -------
    ax : figure object
    
    """
    time, voltage, current, curr_index_0 = get_time_voltage_current_currindex0(data)
    if axis:
        ax = axis
    else: f, ax = plt.subplots(figsize = (10, 10))
    grey_colors = np.array([[0, 0, 0], [49, 79, 79], [105, 105, 105], [112, 138, 144], [119, 136, 153], [190, 190, 190], \
                   [211, 211, 211]]) / 256
    for i in np.arange(0, voltage.shape[1], 1):
        if time[-1] < 0.9:
            ax.plot(time, voltage, color=grey_colors[np.random.randint(0,6)])
        else:
            ax.plot(time[:ft.find_time_index(time, 0.9)], voltage[:ft.find_time_index(time, 0.9), i], \
                    color = grey_colors[np.random.randint(0, 6)])
    ax.set_title('All traces', fontsize = 20)
    ax.set_xlabel('Time (s)', fontsize = 17)
    ax.set_ylabel('Membrane voltage (mV)', fontsize = 17)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    return ax


def three_informative_traces(data, el_num = 2, current_step = 20, start = 0.1, end = 0.7, axis = None, per_type = False):
    """ Plots the voltage traces for lowest hyperpolarisation trace, the first trace that shows spikes and the highest
    frequency trace, based on raw electrophysiological recording.
    ----------
    data : data full of voltage (V) and time (s) for a particular cell
    el_num : integer, from which electrode number has been measured (optional, 2 by default)
    current_step : float, which current step (pA) has been used between consecutive experiments (optional, 20 by default)
    start : start of stimulation interval (s, optional)
    end : end of stimulation interval (s, optional)
    axis : axis you'd like to plot information on (optional, None by default)
    per_type : plot less info if True (optional, False by default)
        
    Returns
    -------
    ax : figure object
    
    """
    time, voltage, current, curr_index_0 = get_time_voltage_current_currindex0(data)
    filter_ = 10
    if (1/time[1]-time[0]) < 20e3:
        filter_ = (1/time[1]-time[0])/(1e3*2)-0.5
    df, df_related_features = extract_spike_features(time, current, voltage, fil = filter_)
    # Cell_Features = get_cell_features(df, df_related_features, curr_index_0)
    # plt.close()
    
    # First current magnitude (starting from the lowest absolute current stimulation magnitude value) where we find more than
    # one spikes
    index_df =  np.where(df.loc[0]['fast_trough_i'].values[~np.isnan(df.loc[0]['fast_trough_i'].values)] > 0)[0][0]
    current_first = np.where(current == df.loc[0]['fast_trough_i'].values[index_df])[0][0]
    # The max amount of spikes in 600 ms of the trace showing the max amount of spikes in 600 ms
    max_freq = np.max(df_related_features['spike_count'].values)
    # Take the first trace showing this many spikes if there are many
    current_max_freq = np.flatnonzero(df_related_features['spike_count'].values >= max_freq)[0]

    if axis:
        ax = axis
    else: f, ax = plt.subplots(figsize = (10, 10))
    
    grey_colors = np.array([[0, 0, 0], [49, 79, 79], [105, 105, 105]]) / 256
    ax.plot(time*1000, voltage[:, 0], color = grey_colors[0], linewidth = 2.5)
    ax.plot(time*1000, voltage[:, current_first], color = grey_colors[1], linewidth = 2.5)
    ax.plot(time[ft.find_time_index(time, 0.1):ft.find_time_index(time, 0.7)]*1000, \
            voltage[ft.find_time_index(time, 0.1):ft.find_time_index(time, 0.7), current_max_freq] + 100, \
            color = grey_colors[2], linewidth = 2.5)
    ax.set_ylim([-150, 210])

    if not per_type:
        ax.set_title('Three traces', fontsize = 20)
        ax.set_xlabel('Time (s)', fontsize = 17)
        ax.set_ylabel('Membrane voltage (mV)', fontsize = 17)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    else:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        sns.despine(ax = ax, left = True, bottom = True)
        #plt.xticks([])
        #plt.yticks([])
    return ax

def plot_info_first_peak(data, el_num = 2, current_step = 20, \
                         start = 0.1, end = 0.7, axis = None):
    """ Displays the 1st AP that has been generated, with AP width, amplitude, AHP and ADP annotations, based on raw electrophysiological recording.
    Parameters
    ----------
    data : data full of voltage (V) and time (s) for a particular cell
    el_num : integer, from which electrode number has been measured (optional, 2 by default)
    current_step : float, which current step (pA) has been used between consecutive experiments (optional, 20 by default)
    start : start of the stimulation (s) in the voltage trace (optional, default 0.1)
    end : end of the stimulation (s) in the voltage trace (optional, default 0.7)
    axis : axis you'd like to plot information on (optional, None by default)
    
    Returns
    -------
    ax : figure object
    """
    
    time, voltage, current, curr_index_0 = get_time_voltage_current_currindex0(data)
    filter_ = 10
    if (1/time[1]-time[0]) < 20e3:
        filter_ = (1/time[1]-time[0])/(1e3*2)-0.5
    df, df_related_features = extract_spike_features(time, current, voltage)
    Cell_Features = get_cell_features(df, df_related_features, time, current, voltage, curr_index_0)
    plt.close()
    
    # First current magnitude (starting from the lowest absolute current stimulation magnitude value) where we find more than
    # three spikes
    
    current_first = np.flatnonzero(np.logical_and(df_related_features['spike_count'].values >=1, \
                                                  df_related_features['current'].values > 0))[0]


    # Actual current there
    current_first_magn = current[current_first]
    # Amount of spikes there
    spike_count = df_related_features['spike_count'].values[current_first]


    # Find start and end indexes in df for all the spikes in that particular train
    index_start_df = np.flatnonzero(df['threshold_i'].values >= \
                        current_first_magn)[0]

    # Get threshold, peak, upstroke and downstroke indexes
    thresh_index = np.array(df['threshold_index'].values[index_start_df], dtype = int)
    upstroke_index = np.array(df['upstroke_index'].values[index_start_df], dtype = int)
    peak_index = np.array(df['peak_index'].values[index_start_df], dtype = int)
    downstroke_index = np.array(df['downstroke_index'].values[index_start_df], dtype = int)
    fast_trough_index = np.array(df['fast_trough_index'].values[index_start_df], dtype = int)
    slow_trough_index = np.array(df['slow_trough_index'].values[index_start_df], dtype = int)
    adp_index = np.array(df['adp_index'].values[index_start_df], dtype = int)
    #slow_trough_index = np.array(slow_trough_index[~np.isnan(slow_trough_index)], dtype = int)

    
    start_index = thresh_index - 50
    if slow_trough_index.size & (adp_index > 0):
        end_index = adp_index + 50 # Plot 50 time indices after the adp index
    else: end_index = fast_trough_index + 50 # Plot 50 time indices after the fast trough index (should always exist)
    
    # When first spike is clipped
    if df['clipped'].values[index_start_df]:
        return
    if axis:
        ax = axis
    else: f, ax = plt.subplots(figsize = (10, 5))
    sns.set_context(rc={'lines.markeredgewidth': 1})
    ax.plot(time[start_index : end_index]*1000, voltage[start_index : end_index, current_first], color = 'orange', \
                lw=4, label = None)

    ax.plot(time[thresh_index]*1000, voltage[thresh_index, current_first], 'b.', ms = 15, label = 'AP threshold')
    ax.plot(time[upstroke_index]*1000, voltage[upstroke_index, current_first], 'r.', ms = 15, label = 'AP upstroke')
    ax.plot(time[peak_index]*1000, voltage[peak_index, current_first], 'g.', ms = 15, label = 'AP peak')
    ax.plot(time[downstroke_index]*1000, voltage[downstroke_index, current_first], 'k.', ms = 15, label = 'AP downstroke')
    ax.plot(time[fast_trough_index]*1000, voltage[fast_trough_index, current_first], 'y.', ms = 15, label = 'AP fast trough')
    if slow_trough_index.size & (adp_index > 0):
        ax.plot(time[adp_index]*1000, voltage[adp_index, current_first], 'c.', ms = 15, label = 'ADP')
        #ax.plot(time[slow_trough_index], voltage[slow_trough_index, current_first], 'm.', ms = 15, label = \
        #       'AP slow trough\n(if applicable)')
    ax.legend(fontsize = 17, frameon=False, loc = 'upper right')
    ax.set_xlabel('Time (s)', fontsize = 17)
    ax.set_ylabel('Membrane voltage (mV)', fontsize = 17)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    ax.set_title('First AP', fontsize = 19)

    # Nice annotations
    
    # For the AP amplitude
    ax.annotate('', xy = (time[peak_index]*1000, voltage[peak_index, current_first]), \
                xycoords = 'data', xytext = (time[peak_index]*1000, voltage[thresh_index, current_first]), \
                textcoords = 'data', arrowprops = {'arrowstyle': '<->', 'ec': 'grey', \
                                                   'connectionstyle': 'arc3', 'lw': 2, 'shrinkB': 0})
    
    ax.plot(time[peak_index]*1000, voltage[thresh_index, current_first], marker = '_', color = 'black', ms = 100)
    ax.plot(time[peak_index]*1000, voltage[peak_index, current_first], marker = '_', color = 'black', ms = 100)

    # For the AP width
    width_level = (voltage[peak_index, current_first] - voltage[thresh_index, current_first])/2 + \
                   voltage[thresh_index, current_first]
    width_start_index = peak_index - np.flatnonzero(voltage[peak_index : thresh_index:-1, current_first] <= width_level)[0]
    width_end_index = peak_index + np.flatnonzero(voltage[peak_index: fast_trough_index, current_first] <=width_level)[0]
    ax.plot(time[width_start_index]*1000, voltage[width_end_index, current_first], '|', color = 'black', ms = 100)
    ax.plot(time[width_end_index]*1000, voltage[width_end_index, current_first], '|', color = 'black', ms = 100)

    # The width itself is calculated based on t[width_end_index] - t[width_start_index], but the voltages might be different
    # at the respective indices, thus we choose the arrow to go from v[width_end_index] to v[width_end_index] to make
    # it horizontal (interpretability of the figure improves)
    ax.annotate('', xy = (time[width_start_index]*1000, voltage[width_end_index, current_first]), \
                xycoords = 'data', xytext = (time[width_end_index]*1000, voltage[width_end_index, current_first]), \
                textcoords = 'data', arrowprops = {'arrowstyle': '<->', 'connectionstyle': 'arc3',\
                                                   'lw': 2, 'ec': 'grey', 'shrinkA': 0})
    ax.annotate('AP width', xy = (time[width_start_index]*1000, \
                 width_level - 5), xycoords='data', xytext = (5, -15), textcoords = 'offset points', fontsize = 19)

    # We still need to annotate the AP amplitude based on the width_level!
    ax.annotate('AP amplitude', xy = (time[peak_index]*1000-1.8, width_level + 30), \
                 xycoords = 'data', xytext = (5, 0), textcoords = 'offset points', fontsize = 19)
    
    # For the AHP
    ax.plot(time[fast_trough_index]*1000, voltage[thresh_index, current_first], marker = "_", color = 'black', ms = 100)
    ax.plot(time[fast_trough_index]*1000, voltage[fast_trough_index, current_first], marker = "_", color = 'black', ms = 100)
    ax.annotate('', xy = (time[fast_trough_index], voltage[thresh_index, current_first]), \
                xycoords = 'data', xytext = (time[fast_trough_index]*1000, voltage[fast_trough_index, current_first]), \
                textcoords = 'data', arrowprops = {'arrowstyle': '<->','connectionstyle': 'arc3', \
                                                   'lw': 2, 'ec': 'grey', 'shrinkB': 0})
    fast_trough_level = (voltage[thresh_index, current_first] - voltage[fast_trough_index, current_first])/2 + \
                   voltage[fast_trough_index, current_first]
    ax.annotate('AHP', xy = (time[fast_trough_index], fast_trough_level), \
                xycoords = 'data', xytext = (10, -5), textcoords = 'offset points', fontsize = 19)
    
    # For a possible ADP
    if slow_trough_index.size & (adp_index > 0):
        ax.plot(time[adp_index]*1000, voltage[adp_index, current_first], marker = "_", color = 'black', ms = 100)
        ax.plot(time[adp_index]*1000, voltage[fast_trough_index, current_first], marker = "_", color = 'black', ms = 100)
        ax.annotate('', xy = (time[adp_index]*1000, voltage[fast_trough_index, current_first]), \
                xycoords = 'data', xytext = (time[adp_index]*1000, voltage[adp_index, current_first]), \
                textcoords = 'data', arrowprops = {'arrowstyle': '<->', 'connectionstyle': 'arc3', \
                                                   'lw': 2, 'ec': 'b', 'shrinkB': 0})
        adp_level = (voltage[adp_index, current_first] - voltage[fast_trough_index, current_first])/2 + \
                   voltage[fast_trough_index, current_first]
        ax.annotate('ADP', xy = (time[adp_index]*1000, adp_level), \
                xycoords = 'data', xytext = (10, -5), textcoords = 'offset points', fontsize = 19)
    
    return ax

def plot_max_spikes_trace(data, el_num = 2, current_step = 20, start = 0.1, end = 0.7, axis = None):
    """ Displays maximum firing trace, based on raw electrophysiological recording.
    
    Parameters
    ----------
    data : data full of voltage (V) and time (s) for a particular cell
    el_num : integer, from which electrode number has been measured (optional, 2 by default)
    current_step : float, which current step (pA) has been used between consecutive experiments (optional, 20 by default)
    start : start of the stimulation (s) in the voltage trace (optional, default 0.1)
    end : end of the stimulation (s) in the voltage trace (optional, default 0.7)
    axis : axis you'd like to plot information on (optional, None by default)
    
    Returns
    -------
    ax : figure object
    """
    
    time, voltage, current, curr_index_0 = get_time_voltage_current_currindex0(data)
    filter_ = 10
    if (1/time[1]-time[0]) < 20e3:
        filter_ = (1/time[1]-time[0])/(1e3*2)-0.5
    df, df_related_features = extract_spike_features(time, current, voltage)
    Cell_Features = get_cell_features(df, df_related_features, time, current, voltage, curr_index_0)
    plt.close()


    # The max amount of spikes in 600 ms of the trace showing the max amount of spikes in 600 ms
    max_freq = np.max(df_related_features['spike_count'].values)
    # Take the first trace showing this many spikes if there are many
    current_max_freq = np.flatnonzero(df_related_features['spike_count'].values >= max_freq)[0]
    
    if axis:
        ax = axis
    else: f, ax = plt.subplots(figsize = (15, 3))
    sns.set_context(rc={'lines.markeredgewidth': 1})
    
    
    if time[-1] < 0.9:
        ax.plot(time*1000, voltage[:,current_max_freq], color='green', lw=4, label='Highest frequency trace = {:.2f} Hz'.format(max_freq))
    else:
        ax.plot(time[:ft.find_time_index(time, 0.9)]*1000, voltage[:ft.find_time_index(time, 0.9), current_max_freq], \
                    color = 'green', lw=4, label='Highest frequency trace = {:.2f} Hz'.format(max_freq))
    ax.set_xlabel('Time (s)', fontsize = 17)
    ax.set_ylabel('Membrane voltage (mV)', fontsize = 17)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    ax.legend(loc='upper right', fontsize = 17)
    return ax

def plot_lowest_trace(data, el_num = 2, current_step = 20, start = 0.1, end = 0.7, axis = None):
    """ Displays lowest hyperpolarization trace with sag and rebound annotations, based on raw electrophysiological recording.
    Parameters
    ----------
    data : data full of voltage (V) and time (s) for a particular cell
    el_num : integer, from which electrode number has been measured (optional, 2 by default)
    current_step : float, which current step (pA) has been used between consecutive experiments (optional, 20 by default)
    start : start of the stimulation (s) in the voltage trace (optional, default 0.1)
    end : end of the stimulation (s) in the voltage trace (optional, default 0.7)
    fil : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    axis : axis you'd like to plot information on (optional, None by default)
    
    Returns
    -------
    ax : figure object
    """
    
    from matplotlib.patches import Polygon
    
    time,voltage, current, curr_index_0 = get_time_voltage_current_currindex0(data)
    
    filter_ = 10
    
    if (1/time[1]-time[0]) < 20e3:
        filter_ = (1/time[1]-time[0])/(1e3*2)-0.5
    df, df_related_features = extract_spike_features(time, current, voltage, fil = filter_)
    Cell_Features = get_cell_features(df, df_related_features, time, current, voltage, curr_index_0)
    plt.close()
    
    if axis:
        ax = axis
    else: f, ax = plt.subplots(figsize = (10, 10))
    if time[-1] < 0.9:
        ax.plot(time*1000, voltage[:,0], color='blue', lw=4, label=None)
    else:
        ax.plot(time[:ft.find_time_index(time, 0.9)]*1000, voltage[:ft.find_time_index(time, 0.9), 0], \
                    color = 'blue', lw=4, label=None)
    ax.set_xlabel('Time (s)', fontsize = 17)
    ax.set_ylabel('Membrane voltage (mV)', fontsize = 17)
    ax.set_title('Lowest trace', fontsize = 20)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    baseline_interval = 0.1 # To calculate the SS voltage
    
    
    #v_baseline = EphysObject._get_baseline_voltage()
    v_baseline = ft.average_voltage(voltage[:, 0], time, start = start - 0.1, end = start)
    
    start_index = ft.find_time_index(time, start)
    end_index = ft.find_time_index(time, end)
    index_rebound = end_index + 0.05 # Just a first initial value
    
    if np.flatnonzero(voltage[end_index:, 0] > v_baseline).size == 0: # So perfectly zero here means
                                                                      # it did not reach it
        rebound = 0
    else:
        index_rebound = end_index + np.flatnonzero(voltage[end_index:, 0] > v_baseline)[0]
        if not (time[index_rebound] > (end + 0.2)): # We definitely have 100 ms left to calculate the rebound
            rebound = ft.average_voltage(voltage[index_rebound:index_rebound + ft.find_time_index(time, 0.15), 0], \
                                 time[index_rebound:index_rebound + ft.find_time_index(time, 0.15)]) - v_baseline
        else:                                       # Work with whatever time is left
            rebound = ft.average_voltage(voltage[index_rebound:, 0], \
                                 time[index_rebound:]) - v_baseline


    #v_peak, peak_index = EphysObject.voltage_deflection("min")
    
    peak_index = start_index + np.argmin(voltage[start_index:end_index, 0])
    ax.plot(time[peak_index]*1000, voltage[peak_index, 0], '.', c = [0, 0, 0], markersize = 15, label = 'Sag trough')
    v_steady = ft.average_voltage(voltage[:, 0], time, start = end - baseline_interval, end=end)

    # First time SS is reached after stimulus onset
    first_index = start_index + np.flatnonzero(voltage[start_index:peak_index, 0] < v_steady)[0]
    # First time SS is reached after the max voltage deflection downwards in the sag
    if np.flatnonzero(voltage[peak_index:end_index, 0] > v_steady).size == 0: 
        second_index = end_index
    else:
        second_index = peak_index + np.flatnonzero(voltage[peak_index:end_index, 0] > v_steady)[0]
    # Time_to_SS is the time difference between these two time points
    time_to_SS = time[second_index] - time[first_index]
    # Integration_to_SS is the integration area of the voltage between these two time points
    integration_to_SS = -integrate.cumtrapz(voltage[first_index:second_index, 0], time[first_index:second_index])[-1]
    
    # Now let's add nice annotations
    # First up the time and integration to reach the SS
    ax.plot(time[first_index]*1000, voltage[first_index, 0], '|', markersize = 30, color = [0, 0, 0])
    ax.plot(time[second_index]*1000, voltage[second_index, 0], '|', markersize = 30, color = [0, 0, 0])
    ax.annotate('', xy = (time[first_index]*1000, voltage[first_index, 0]), \
            xycoords = 'data', xytext = (time[second_index]*1000, voltage[second_index, 0]), \
            textcoords = 'data', arrowprops = {'arrowstyle': '<->', 'connectionstyle': 'arc3', \
                                               'lw': 2, 'ec': 'grey', 'shrinkA': 0})
    ax.annotate('Sag time', xy=((time[first_index] + time_to_SS/2)*1000, v_steady), \
             xycoords='data', xytext=(5, 5), textcoords='offset points', fontsize = 19)
    a, b = time[first_index]*1000, time[second_index]*1000
    verts = [(a, voltage[first_index, 0]), *zip(time[first_index:second_index]*1000, voltage[first_index:second_index, 0]), \
             (b, voltage[second_index, 0])] 
    poly = Polygon(verts, facecolor = '0.9', edgecolor = '0.5')
    ax.add_patch(poly)
     
    # Now the rebound
    if rebound != 0:
        end_index_for_rebound = index_rebound + ft.find_time_index(time, 0.15)
        if (time[index_rebound] > (0.9 - 0.15)):
            end_index_for_rebound = ft.find_time_index(time, np.max(time)) # Plot till the end (simply what you have left)
        ax.plot(time[index_rebound]*1000, voltage[index_rebound, 0], '|',
                            markersize = 10, color = [0, 0, 0])
        ax.plot(time[end_index_for_rebound]*1000, voltage[end_index_for_rebound, 0], '|', \
                            markersize = 10, color = [0, 0, 0])
        if time[index_rebound] == time[end_index_for_rebound]:
            return ax
        ax.plot([time[index_rebound]*1000,  time[end_index_for_rebound]*1000], \
                [ft.average_voltage(voltage[index_rebound:end_index_for_rebound, 0], \
                                 time[index_rebound:end_index_for_rebound]), \
                 ft.average_voltage(voltage[index_rebound:end_index_for_rebound, 0], \
                                 time[index_rebound:end_index_for_rebound])], '-',  \
                 color = [0, 0, 0])
        ax.plot([time[index_rebound]*1000, time[end_index_for_rebound]*1000], [v_baseline, v_baseline], '-', \
                 color = [0, 0, 0])
        ax.annotate('', xy = ((time[index_rebound] + (time[end_index_for_rebound] - time[index_rebound])/2)*1000, \
                              v_baseline), \
            xycoords = 'data', xytext = ((time[index_rebound] + (time[end_index_for_rebound] - time[index_rebound])/2)*1000, \
                                         v_baseline + rebound), \
            textcoords = 'data', arrowprops = {'arrowstyle': '<->', 'connectionstyle': 'arc3', \
                                               'lw': 2, 'ec': 'grey', 'shrinkB': 0})
        ax.annotate('Rebound', xy=((time[index_rebound] + (time[end_index_for_rebound] - time[index_rebound])/2)*1000, \
                                   v_baseline + rebound/2), \
             xycoords='data', xytext=(0, 10), textcoords='offset points', fontsize = 19)
    plt.legend(fontsize = 17)
    return ax




def adjust_spines(ax, spines, spine_pos=5, color='k', linewidth=None, smart_bounds=True):
    """Convenience function to adjust plot axis spines."""

    # If no spines are given, make everything invisible
    if spines is None:
        ax.axis('off')
        return

    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', spine_pos))  # outward by x points
            #spine.set_smart_bounds(smart_bounds)
            spine.set_color(color)
            if linewidth is not None:
                spine.set_linewidth = linewidth
        else:
            spine.set_visible(False)  # make spine invisible
            # spine.set_color('none')  # this will interfere w constrained plot layout

    # Turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # No visible yaxis ticks and tick labels
        # ax.yaxis.set_visible(False)  # hides whole axis, incl. ax label
        # ax.yaxis.set_ticks([])  # for shared axes, this would delete ticks for all
        plt.setp(ax.get_yticklabels(), visible=False)  # hides ticklabels but not ticks
        plt.setp(ax.yaxis.get_ticklines(), color='none')  # changes tick color to none
        # ax.tick_params(axis='y', colors='none')  # (same as above) changes tick color to none

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # No visible xaxis ticks and tick labels
        # ax.xaxis.set_visible(False)  # hides whole axis, incl. ax label
        # ax.xaxis.set_ticks([])  # for shared axes, this would  delete ticks for all
        plt.setp(ax.get_xticklabels(), visible=False)  # hides ticklabels but not ticks
        plt.setp(ax.xaxis.get_ticklines(), color='none')  # changes tick color to none


def latent_space_ephys(model, latent, X, Y, Y_column_index, features, alpha = 1, triangle_max_len=50,
                       fontsize=13, axis = None):
    '''
    Parameters
    ----------
    model: keras deep bottleneck neural network regression model
    latent: latent space projections
    X: 2D numpy array, normalized transcriptomic data
    Y: 2D numpy array, normalized ephys data
    Y_column_index: column index in Y, correspoding to certain feature
    features: list of all ephys features (Y_column_index should correspond to correct feature in this list!)
    alpha: transparancy for contours (default = 0.5)
    triangle_max_len: # triangles with too long edges (poorly constrained by data) (default=50)
    fontsize: fontsize of title (default: 13)
    axis: axis to plot one (default: None)
    
    Returns
    -------
    ax: figure objects; latent space with gene activation contours
    '''
    # Create mappings from the model to get latent activations from genes and ephys predictions from genes and
    # latent activations
    ephys_prediction = model.predict(X)

    if axis:
        ax = axis
    else:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize = (6, 6))      
    
    # produces triangles from latent coordinates
    triang = tri.Triangulation(latent[:,0], latent[:,1])
    
    # extract coordinates of each triangle
    x1=latent[:,0][triang.triangles][:,0]
    x2=latent[:,0][triang.triangles][:,1]
    x3=latent[:,0][triang.triangles][:,2]
    y1=latent[:,1][triang.triangles][:,0]
    y2=latent[:,1][triang.triangles][:,1]
    y3=latent[:,1][triang.triangles][:,2]
    
    # calculate the area of each triangle
    #A=1/2 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    
    # calculate edges
    edges=np.concatenate((np.sqrt((x2-x1)**2+(y2-y1)**2)[:,np.newaxis],
                np.sqrt((x3-x1)**2+(y3-y1)**2)[:,np.newaxis],
                np.sqrt((x2-x3)**2+(y2-y3)**2)[:,np.newaxis]), axis=1)
    
    # triangles with an edge longer than the 50th biggest are masked. These are triangles poorly constrained by data
    triang.set_mask(np.max(edges, axis=1)>np.max(edges, axis=1)[np.argsort(np.max(edges, axis=1))][-triangle_max_len])
    
    ax.tricontourf(triang, ephys_prediction[:, Y_column_index], cmap='inferno',
                   levels=np.linspace(-1,1,40), extend='both')
    ax.set_xlim([np.min(latent[:, 0]), np.max(latent[:, 0])])
    ax.set_ylim([np.min(latent[:, 1]), np.max(latent[:, 1])])
    #ax.set_aspect('equal', adjustable='box')
    ax.set_title(features[Y_column_index], fontsize=fontsize, y=0.97)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    return ax

        
def latent_space_genes(model, latent, Z, X, X_column_index, geneNames, alpha = 1, triangle_max_len=50,
                       fontsize=13, axis = None):
    '''
    Parameters
    ----------
    model: keras decoder network (from latent space to selected genes)
    latent: latent space
    Z: projections (could be same as latent)
    X: 2D numpy array, normalized transcriptomic data (should be a selected genes i.e. reduced size matrix)
    X_column_index: column index in X, correspoding to certain gene
    geneNames: list of gene names (X_column_index should correspond to correct gene in this list!)
    alpha: transparancy for contours (default = 0.5)
    triangle_max_len: # triangles with too long edges (poorly constrained by data) (default=50)
    fontsize: fontsize of title (default: 13)
    axis: axis to plot one (default: None)
    
    Returns
    -------
    ax: figure objects; latent space with gene activation contours
    '''
    gene_prediction=model.predict(latent)
    
    # Create figure
    if axis:
        ax = axis
    else:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize = (6, 6))    
    
    # produces triangles from latent coordinates
    triang = tri.Triangulation(Z[:,0], Z[:,1])
    
    # extract coordinates of each triangle
    x1=Z[:,0][triang.triangles][:,0]
    x2=Z[:,0][triang.triangles][:,1]
    x3=Z[:,0][triang.triangles][:,2]
    y1=Z[:,1][triang.triangles][:,0]
    y2=Z[:,1][triang.triangles][:,1]
    y3=Z[:,1][triang.triangles][:,2]
    
    # calculate the area of each triangle
    # A=1/2 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    
    
    # calculate edges
    edges=np.concatenate((np.sqrt((x2-x1)**2+(y2-y1)**2)[:,np.newaxis],
                np.sqrt((x3-x1)**2+(y3-y1)**2)[:,np.newaxis],
                np.sqrt((x2-x3)**2+(y2-y3)**2)[:,np.newaxis]), axis=1)
    
    # triangles with an edge longer than the 50th biggest are masked. These are triangles poorly constrained by data
    triang.set_mask(np.max(edges, axis=1)>np.max(edges, axis=1)[np.argsort(np.max(edges, axis=1))][-triangle_max_len])
    ax.tricontourf(triang, gene_prediction[:, X_column_index], cmap='inferno',
                   levels=np.linspace(-1,1,40), extend='both')
    ax.set_xlim([np.min(Z[:, 0]), np.max(Z[:, 0])])
    ax.set_ylim([np.min(Z[:, 1]), np.max(Z[:, 1])])
    #ax.set_aspect('equal', adjustable='box')
    ax.set_title(geneNames[X_column_index], fontsize=fontsize, y=0.97)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    return ax


def create_axes(fig):
    # return axes to construct a less big figure
    
    if fig is None:
        fig = plt.figure(figsize=(9, 6))
    else: fig = fig
    
    width = 0.1 # width of every small heatmap plot
    height = 0.135 # height of every small heatmap plot
    
    b_ax_latent=plt.axes([.33,0.5,0.31,0.45])
    b_ax_genes_1=plt.axes([0,0.82,width,height])
    b_ax_genes_2=plt.axes([0.11,0.82,width,height])
    b_ax_genes_3=plt.axes([0.22,0.82,width,height])
    
    b_ax_genes_4=plt.axes([0,0.66,width,height])
    b_ax_genes_5=plt.axes([0.11,0.66,width,height])
    b_ax_genes_6=plt.axes([0.22,0.66,width,height])
    
    b_ax_genes_7=plt.axes([0,0.5,width,height])
    b_ax_genes_8=plt.axes([0.11,0.5,width,height])
    b_ax_genes_9=plt.axes([0.22,0.5,width,height])
    
    b_ax_ephys_1=plt.axes([0.66,0.82, width,height])
    b_ax_ephys_2=plt.axes([0.77,0.82, width,height])
    b_ax_ephys_3=plt.axes([0.88,0.82, width,height])
    
    b_ax_ephys_4=plt.axes([0.66,0.66, width,height])
    b_ax_ephys_5=plt.axes([0.77,0.66, width,height])
    b_ax_ephys_6=plt.axes([0.88,0.66, width,height])
    
    b_ax_ephys_7=plt.axes([0.66,0.5, width,height])
    b_ax_ephys_8=plt.axes([0.77,0.5, width,height])
    b_ax_ephys_9=plt.axes([0.88,0.5, width,height])
    
    ax_latent=plt.axes([.33,0,0.31,0.45])    
    ax_genes_1=plt.axes([0,0.32,width,height])
    ax_genes_2=plt.axes([0.11,0.32,width,height])
    ax_genes_3=plt.axes([0.22,0.32,width,height])
    
    ax_genes_4=plt.axes([0,0.16,width,height])
    ax_genes_5=plt.axes([0.11,0.16,width,height])
    ax_genes_6=plt.axes([0.22,0.16,width,height])
    
    ax_genes_7=plt.axes([0,0,width,height])
    ax_genes_8=plt.axes([0.11,0,width,height])
    ax_genes_9=plt.axes([0.22,0,width,height])
    
    ax_ephys_1=plt.axes([0.66,0.32, width,height])
    ax_ephys_2=plt.axes([0.77,0.32, width,height])
    ax_ephys_3=plt.axes([0.88,0.32, width,height])
    
    ax_ephys_4=plt.axes([0.66,0.16, width,height])
    ax_ephys_5=plt.axes([0.77,0.16, width,height])
    ax_ephys_6=plt.axes([0.88,0.16, width,height])
    
    ax_ephys_7=plt.axes([0.66, 0, width,height])
    ax_ephys_8=plt.axes([0.77, 0, width,height])
    ax_ephys_9=plt.axes([0.88, 0, width,height])
    
    
    return [b_ax_latent, b_ax_genes_1, b_ax_genes_2, b_ax_genes_3, b_ax_genes_4, b_ax_genes_5, b_ax_genes_6, \
            b_ax_genes_7, b_ax_genes_8, b_ax_genes_9, b_ax_ephys_1, b_ax_ephys_2, b_ax_ephys_3, b_ax_ephys_4, \
            b_ax_ephys_5, b_ax_ephys_6, b_ax_ephys_7, b_ax_ephys_8, b_ax_ephys_9, \
            ax_latent, ax_genes_1, ax_genes_2, ax_genes_3, ax_genes_4, ax_genes_5, ax_genes_6, \
            ax_genes_7, ax_genes_8, ax_genes_9, ax_ephys_1, ax_ephys_2, ax_ephys_3, ax_ephys_4, \
            ax_ephys_5, ax_ephys_6, ax_ephys_7, ax_ephys_8, ax_ephys_9
           ]

def create_less_axes(fig):
    # return axes to construct figure
    
    if fig is None:
        fig = plt.figure(figsize=(16, 16/3)) # width/height ratio of 3
    else: fig = fig
    
    ax_latent=plt.axes([.33,0,0.33,0.99])
    ax_genes_1=plt.axes([0,0.66,0.11,0.3])
    ax_genes_2=plt.axes([0.11,0.66,0.11,0.3])
    ax_genes_3=plt.axes([0.22,0.66,0.11,0.3])
    ax_genes_4=plt.axes([0,0.33,0.11,0.3])
    ax_genes_5=plt.axes([0.11,0.33,0.11,0.3])
    ax_genes_6=plt.axes([0.22,0.33,0.11,0.3])
    ax_genes_7=plt.axes([0,0,0.11,0.3])
    ax_genes_8=plt.axes([0.11,0,0.11,0.3])
    ax_genes_9=plt.axes([0.22,0,0.11,0.3])
    ax_ephys_1=plt.axes([0.66,0.66, 0.11, 0.3])
    ax_ephys_2=plt.axes([0.77,0.66, 0.11, 0.3])
    ax_ephys_3=plt.axes([0.88,0.66, 0.11, 0.3])
    ax_ephys_4=plt.axes([0.66,0.33, 0.11, 0.3])
    ax_ephys_5=plt.axes([0.77,0.33, 0.11, 0.3])
    ax_ephys_6=plt.axes([0.88,0.33, 0.11, 0.3])
    ax_ephys_7=plt.axes([0.66,0, 0.11, 0.3])
    ax_ephys_8=plt.axes([0.77,0, 0.11, 0.3])
    ax_ephys_9=plt.axes([0.88,0, 0.11, 0.3])
    
    return [ax_latent, ax_genes_1, ax_genes_2, ax_genes_3, ax_genes_4, ax_genes_5, ax_genes_6, ax_genes_7, ax_genes_8, \
           ax_genes_9, ax_ephys_1, ax_ephys_2, ax_ephys_3, ax_ephys_4, ax_ephys_5, ax_ephys_6, ax_ephys_7, ax_ephys_8, \
           ax_ephys_9]


