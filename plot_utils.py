import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import integrate

from ephys_utils import get_time_voltage_current_currindex0, extract_spike_features, get_cell_features
import ephys_extractor as efex
import ephys_features as ft

def prel_inspection(data, axis = None):
    """ Plots the voltage traces for some current steps ment for quality checks and preliminary inspection
    Parameters
    ----------
    data : data full of voltage (V) and time (s) for a particular cell
    axis : axis you'd like to plot information on (optional, None by default)
        
    Returns
    -------
    ax : figure object
    
    """
    
    (
    time,
    voltage,
    current,
    curr_index_0,
    ) = get_time_voltage_current_currindex0(data)
    
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
    """ Plots the voltage traces for the steepest hyperpolarisation trace, the first trace that shows spikes and the highest
    frequency trace.
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
    (
    time,
    voltage,
    current,
    curr_index_0,
    ) = get_time_voltage_current_currindex0(data)
    
    filter_ = 10

    if axis:
        ax = axis
    else: f, ax = plt.subplots(figsize = (10, 10))
    
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
    
    grey_colors = np.array([[0, 0, 0], [49, 79, 79], [105, 105, 105]]) / 256
    ax.plot(time, voltage[:, 0], color = grey_colors[0], linewidth = 2.5)
    ax.plot(time, voltage[:, current_first], color = grey_colors[1], linewidth = 2.5)
    ax.plot(time[ft.find_time_index(time, 0.1):ft.find_time_index(time, 0.7)], \
            voltage[ft.find_time_index(time, 0.1):ft.find_time_index(time, 0.7), current_max_freq] + 100, \
            color = grey_colors[2], linewidth = 2.5)
    ax.set_ylim([-150, 210])
    
    #if np.sum(df_related_features['spike_count'] == max_freq) % 2 != 0: # The median will be a Fano_factor and cv
    #                                                                        # in the DataFrame
    #            fano_factor = np.nanmedian(df_related_features.loc[df_related_features['spike_count'] == max_freq, \
    #                                                               'fano_factor'])
    #else: # Just take the first trace showing the max # number of spikes in 600 ms
    #            fano_factor = df_related_features.loc[df_related_features['spike_count'] == max_freq, \
    #                                                  'fano_factor'].values[0]

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

def plot_info(data, el_num = 2, current_step = 20, start = 0.1, end = 0.7, axis = None):
    """ Analyses a specific cell and returns a figure object with annotations on a particular
    trace of how certain features have been calculated.
    
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
    
    (
    time,
    voltage,
    current,
    curr_index_0,
    ) = get_time_voltage_current_currindex0(data)
    
    filter_ = 10
    
    if (1/time[1]-time[0]) < 20e3:
        filter_ = (1/time[1]-time[0])/(1e3*2)-0.5
    df, df_related_features = extract_spike_features(time, current, voltage, fil = filter_)
    Cell_Features = get_cell_features(df, df_related_features, time, current, voltage, curr_index_0)
    plt.close()
    
    if not np.any(df_related_features['spike_count'].values > 3):
        return
    # First current magnitude (starting from the lowest absolute current stimulation magnitude value) where we find more than
    # three spikes
    current_first = np.flatnonzero(df_related_features['spike_count'].values >=3)[0]
    if axis:
        ax = axis
    else: f, ax = plt.subplots(figsize = (10, 10))
    
    if time[-1] < 0.9:
        ax.plot(time, voltage[:,current_first], color=np.array([0,0,0]))
    else:
        ax.plot(time[:ft.find_time_index(time, 0.9)], voltage[:ft.find_time_index(time, 0.9), current_first], \
                    color = np.array([0,0,0]))
    # Actual current there
    current_first_magn = current[current_first]
    # Amount of spikes there
    spike_count = df_related_features['spike_count'].values[current_first]
    
    
    # Find start and end indexes in df for all the spikes in that particular train
    index_start_df = np.flatnonzero(df['threshold_i'].values >= current_first_magn)[0]
    index_end_df = np.flatnonzero(df['threshold_i'].values >= current_first_magn)[0] + spike_count
    # When last spike is clipped
    if df['clipped'].values[index_end_df-1]:
        index_end_df = np.flatnonzero(df['threshold_i'].values >= \
                                        current_first_magn)[0] + (spike_count - 1)
    # When the spike before the last spike is also clipped
    if df['clipped'].values[index_end_df-1]:
        index_end_df = np.flatnonzero(df['threshold_i'].values >= \
                                        current_first_magn)[0] + (spike_count - 2)
    
    # Get threshold, peak, upstroke and downstroke indexes
    thresh_index = np.array(df['threshold_index'].values[index_start_df : index_end_df], dtype = int)
    upstroke_index = np.array(df['upstroke_index'].values[index_start_df : index_end_df], dtype = int)
    peak_index = np.array(df['peak_index'].values[index_start_df : index_end_df], dtype = int)
    downstroke_index = np.array(df['downstroke_index'].values[index_start_df : index_end_df], dtype = int)
    fast_trough_index = np.array(df['fast_trough_index'].values[index_start_df : index_end_df], dtype = int)
    fast_trough_index = np.array(fast_trough_index[~np.isnan(fast_trough_index)], dtype = int)
    slow_trough_index = df['slow_trough_index'].values[index_start_df : index_end_df]
    slow_trough_index = np.array(slow_trough_index[~np.isnan(slow_trough_index)], dtype = int)
    
    start_index = (np.abs(time - start/10)).argmin() # Find closest index where the injection current starts (quite a bit before now)
    end_index = (np.abs(time - end)).argmin() # Find closest index where the injection current ends
    
    #ax.plot(time[np.abs(time - start/2).argmin()], Cell_Features['Rm (mV)'], 'C0.', ms = 15, label = None)
    ax.plot(time[thresh_index], voltage[thresh_index, current_first], 'b.', ms = 15, label = 'AP threshold')
    ax.plot(time[upstroke_index], voltage[upstroke_index, current_first], 'r.', ms = 15, label = 'AP upstroke')
    ax.plot(time[peak_index], voltage[peak_index, current_first], 'g.', ms = 15, label = 'AP peak')
    ax.plot(time[downstroke_index], voltage[downstroke_index, current_first], 'k.', ms = 15, label = 'AP downstroke')
    ax.plot(time[fast_trough_index], voltage[fast_trough_index, current_first], 'y.', ms = 15, label = 'AP fast trough')
    ax.plot(time[slow_trough_index], voltage[slow_trough_index, current_first], 'm.', ms = 15, \
            label = 'AP slow trough\n(if applicable)')
    ax.set_xlim([time[start_index], time[end_index]])
    ax.set_xlabel('Time (s)', fontsize = 17)
    ax.set_ylabel('Membrane voltage (mV)', fontsize = 17)
    ax.set_title('First trace showing at least three APs', fontsize = 20)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    #ax.legend(['Resting Vm', 'Threshold', 'Upstroke', 'Peak', 'Downstroke', 'Fast trough', \
    #           'Slow trough (if applicable)'], fontsize = 15, loc='upper left', bbox_to_anchor=(1, 1))
    ax.legend(fontsize = 15, loc = 'upper right')
    
    # Nice annotations
    if len(thresh_index) > 2 :
        ax.plot(time[thresh_index[0:3]], voltage[thresh_index[0:3], current_first], '|', color = 'black', ms = 200)
        ax.annotate('', xy = (time[thresh_index[0]], voltage[thresh_index[0], current_first] - 10), \
                xycoords = 'data', xytext = (time[thresh_index[1]], voltage[thresh_index[0], current_first] - 10), \
                textcoords = 'data', arrowprops = {'arrowstyle': '<->', 'connectionstyle': 'arc3', \
                                                   'lw': 2, 'ec': 'grey', 'shrinkA': 0})
        ax.annotate('', xy = (time[thresh_index[1]], voltage[thresh_index[0], current_first] - 10), \
                xycoords = 'data', xytext = (time[thresh_index[2]], voltage[thresh_index[0], current_first] - 10), \
                textcoords = 'data', arrowprops = {'arrowstyle': '<->', 'connectionstyle': 'arc3', \
                                                   'lw': 2, 'ec': 'grey', 'shrinkA': 0})
        ax.annotate('ISI adapt. index = 2nd ISI / 1st ISI', xy=(time[thresh_index[1]], voltage[thresh_index[1], current_first]-30), \
                 xycoords='data', xytext=(5, 0), textcoords='offset points', fontsize = 15)
    return ax

def plot_info_first_peak(data, el_num = 2, current_step = 20, \
                         start = 0.1, end = 0.7, axis = None):
    """ Analyses a specific cell and returns a figure object with annotations on the first
    peak of a particular trace of how certain features have been calculated.
    
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
    
    (
    time,
    voltage,
    current,
    curr_index_0,
    ) = get_time_voltage_current_currindex0(data)
    
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
                label = None, lw=4)

    ax.plot(time[thresh_index]*1000, voltage[thresh_index, current_first], 'b.', ms = 15, label = 'AP threshold')
    ax.plot(time[upstroke_index]*1000, voltage[upstroke_index, current_first], 'r.', ms = 15, label = 'AP upstroke')
    ax.plot(time[peak_index]*1000, voltage[peak_index, current_first], 'g.', ms = 15, label = 'AP peak')
    ax.plot(time[downstroke_index]*1000, voltage[downstroke_index, current_first], 'k.', ms = 15, label = 'AP downstroke')
    ax.plot(time[fast_trough_index]*1000, voltage[fast_trough_index, current_first], 'y.', ms = 15, label = 'AP fast trough')
    # if slow_trough_index.size & (adp_index > 0):
    #     ax.plot(time[adp_index], voltage[adp_index, current_first], 'c.', ms = 15, label = 'ADP')

    ax.legend(fontsize = 17, frameon=False, loc = (.7,.4))
    # ax.set_xlabel('Time (s)', fontsize = 17)
    # ax.set_ylabel('Membrane voltage (mV)', fontsize = 17)
    # ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    # ax.set_title('First AP', fontsize = 20)

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
                 width_level - 5), xycoords='data', xytext = (5, -15), textcoords = 'offset points', fontsize = 17)

    # We still need to annotate the AP amplitude based on the width_level!
    ax.annotate('AP amplitude', xy = (time[peak_index]*1000, width_level + 30), \
                 xycoords = 'data', xytext = (5, 0), textcoords = 'offset points', fontsize = 17)
    
    # For the AHP
    ax.plot(time[fast_trough_index]*1000, voltage[thresh_index, current_first], marker = "_", color = 'black', ms = 100)
    ax.plot(time[fast_trough_index]*1000, voltage[fast_trough_index, current_first], marker = "_", color = 'black', ms = 100)
    ax.annotate('', xy = (time[fast_trough_index]*1000, voltage[thresh_index, current_first]), \
                xycoords = 'data', xytext = (time[fast_trough_index]*1000, voltage[fast_trough_index, current_first]), \
                textcoords = 'data', arrowprops = {'arrowstyle': '<->','connectionstyle': 'arc3', \
                                                   'lw': 2, 'ec': 'grey', 'shrinkB': 0})
    fast_trough_level = (voltage[thresh_index, current_first] - voltage[fast_trough_index, current_first])/2 + \
                   voltage[fast_trough_index, current_first]
    ax.annotate('AHP', xy = (time[fast_trough_index]*1000, fast_trough_level), \
                xycoords = 'data', xytext = (10, -5), textcoords = 'offset points', fontsize = 17)
    
    ax.set_xlabel('Time (ms)', fontsize = 17)
    ax.set_ylabel('Membrane voltage (mV)', fontsize = 17)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    # # For a possible ADP
    # if slow_trough_index.size & (adp_index > 0):
    #     ax.plot(time[adp_index], voltage[adp_index, current_first], marker = "_", color = 'black', ms = 100)
    #     ax.plot(time[adp_index], voltage[fast_trough_index, current_first], marker = "_", color = 'black', ms = 100)
    #     ax.annotate('', xy = (time[adp_index], voltage[fast_trough_index, current_first]), \
    #             xycoords = 'data', xytext = (time[adp_index], voltage[adp_index, current_first]), \
    #             textcoords = 'data', arrowprops = {'arrowstyle': '<->', 'connectionstyle': 'arc3', \
    #                                                'lw': 2, 'ec': 'b', 'shrinkB': 0})
    #     adp_level = (voltage[adp_index, current_first] - voltage[fast_trough_index, current_first])/2 + \
    #                voltage[fast_trough_index, current_first]
    #     ax.annotate('ADP', xy = (time[adp_index], adp_level), \
    #             xycoords = 'data', xytext = (10, -5), textcoords = 'offset points', fontsize = 15)
    
    return ax

def plot_max_spikes_trace(data, el_num = 2, current_step = 20, start = 0.1, end = 0.7, axis = None):
    """ Analyses a specific a cell and returns a figure object with the trace for which
    the features ISI FF, ISI CV, AP amp FF, AP amp CV, max # number of spikes in 600 ms and the Spike frequency adaptation
    are normally calculated (only in the very specific case of more traces showing the same max # of spikes in 600 ms can
    the trace for which the feature is being calculated differ among the features).
    
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
    
    (
    time,
    voltage,
    current,
    curr_index_0,
    ) = get_time_voltage_current_currindex0(data)
    
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
    #sns.set_context(rc={'lines.markeredgewidth': 1})
    
    
    if time[-1] < 0.9:
        ax.plot(time*1000, voltage[:,current_max_freq], color='green', lw=4)
    else:
        ax.plot(time[:ft.find_time_index(time, 0.9)]*1000, voltage[:ft.find_time_index(time, 0.9), current_max_freq], \
                    color = 'green', lw=4)
    
    ax.annotate('Highest firing rate = {:.2f} Hz'.format(max_freq/0.6), (750,10), fontsize=17)
    
    ax.set_xlabel('Time (ms)', fontsize = 17)
    ax.set_ylabel('Membrane voltage (mV)', fontsize = 17)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    # ax.set_title('Highest frequency trace', fontsize = 20)
    return ax

def plot_lowest_trace(data, el_num = 2, current_step = 20, start = 0.1, end = 0.7, axis = None):
    """ Analyses a specific a cell and returns a figure object with annotations on a particular
    trace of how certain features have been calculated.
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
    
    (
    time,
    voltage,
    current,
    curr_index_0,
    ) = get_time_voltage_current_currindex0(data)
    
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
        ax.plot(time*1000, voltage[:,0], color='blue', label=None,lw=4)
    else:
        ax.plot(time[:ft.find_time_index(time, 0.9)]*1000, voltage[:ft.find_time_index(time, 0.9), 0], \
                    color = 'blue', label=None, lw=4)
    # ax.set_xlabel('Time (s)', fontsize = 17)
    # ax.set_ylabel('Membrane voltage (mV)', fontsize = 17)
    # ax.set_title('Lowest trace', fontsize = 20)
    # ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    
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
    ax.annotate('Sag time', xy=(time[first_index]*1000 + time_to_SS/2*1000, v_steady), \
             xycoords='data', xytext=(5, 5), textcoords='offset points', fontsize = 17)
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
        ax.annotate('', xy = (time[index_rebound]*1000 + (time[end_index_for_rebound] - time[index_rebound])/2*1000, \
                              v_baseline), \
            xycoords = 'data', xytext = (time[index_rebound]*1000 + (time[end_index_for_rebound] - time[index_rebound])/2*1000, \
                                         v_baseline + rebound), \
            textcoords = 'data', arrowprops = {'arrowstyle': '<->', 'connectionstyle': 'arc3', \
                                               'lw': 2, 'ec': 'grey', 'shrinkB': 0})
        ax.annotate('Rebound', xy=(time[index_rebound]*1000 + (time[end_index_for_rebound] - time[index_rebound])/2*1000, \
                                   (v_baseline + rebound/2)-8), \
             xycoords='data', xytext=(0, 2), textcoords='offset points', fontsize = 17)
    
    ax.legend(frameon=False, fontsize=17, loc=(.7,.15))
    
    ax.set_xlabel('Time (ms)', fontsize = 17)
    ax.set_ylabel('Membrane voltage (mV)', fontsize = 17)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    return ax

def plot_w_style(data, el_num = 2, current_step = 20, start = 0.1, end = 0.7, axes = None):
    """Analyses a specific cell and returns axes with traces and annotations.
    
    Parameters
    ----------
    data : data full of voltage (V) and time (s) for a particular cell
    el_num : integer, from which electrode number has been measured (optional, 2 by default)
    start : start of the stimulation (s) in the voltage trace (optional, default 0.1)
    end : end of the stimulation (s) in the voltage trace (optional, default 0.7)
    current_step : float, which current step (pA) has been used between consecutive experiments (optional, 20 by default)
    
    Returns
    -------
    axes : figure objects
    
    """
    (
    time,
    voltage,
    current,
    curr_index_0,
    ) = get_time_voltage_current_currindex0(data)
    
    filter_ = 10
    
    if (1/time[1]-time[0]) < 20e3:
        filter_ = (1/time[1]-time[0])/(1e3*2)-0.5
    
    df, df_related_features = extract_spike_features(time, current, voltage, start = start, end = end, fil = filter_)
    
    sns.set_style('ticks')
    sns.set_context('paper', font_scale=.85)
    
    if axes:
        axes = axes
    else:
        fig = plt.figure(figsize = (20, 18))
        ax1 = plt.subplot2grid((4, 2), (0, 0))
        ax2 = plt.subplot2grid((4, 2), (0, 1))
        ax3 = plt.subplot2grid((4, 2), (1, 0))
        ax4 = plt.subplot2grid((4, 2), (1, 1))
        ax5 = plt.subplot2grid((4, 2), (2, 0))
        ax6 = plt.subplot2grid((4, 2), (2, 1))
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    
    prel_inspection(data, axis = axes[0])
    plot_info(data, el_num = el_num, current_step = current_step, start = start, end = end, axis = axes[1])
    plot_info_first_peak(data, el_num = el_num, current_step = current_step, start = start, end = end, axis = axes[2])
    Cell_Features = get_cell_features(df, df_related_features, time, current, voltage, \
                                      curr_index_0 = curr_index_0, axis = axes[3])
    plot_max_spikes_trace(data, axis = axes[4])
    plot_lowest_trace(data, axis = axes[5])
    plt.tight_layout()
    
    for axis in axes:
        sns.despine(ax = axis)
    
    sns.set_context(rc={'lines.markeredgewidth': 3})
    
    plt.text(-0.05, 1.04, 'a', transform = plt.gcf().get_axes()[0].transAxes, fontsize=25, fontweight='bold')
    plt.text(-0.05, 1.04, 'b', transform = plt.gcf().get_axes()[1].transAxes, fontsize=25, fontweight='bold')
    plt.text(-0.05, 1.04, 'c', transform = plt.gcf().get_axes()[2].transAxes, fontsize=25, fontweight='bold')
    plt.text(-0.05, 1.04, 'd', transform = plt.gcf().get_axes()[3].transAxes, fontsize=25, fontweight='bold')
    plt.text(-0.05, 1.04, 'e', transform = plt.gcf().get_axes()[4].transAxes, fontsize=25, fontweight='bold')
    plt.text(-0.05, 1.04, 'f', transform = plt.gcf().get_axes()[5].transAxes, fontsize=25, fontweight='bold')
    
    return axes