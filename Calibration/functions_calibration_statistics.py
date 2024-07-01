

# IMPORTING LIBRARIES
import collections
import copy
import numpy as np
import os
import params_calibration_statistics
import pickle
import xarray as xr

import matplotlib.pyplot as plt
import scipy.signal as signal

from ecogconf import ECoGConfig
from h5eeg import H5EEGFile



from scipy.io import loadmat


############################ UPLOADING GLOBAL PARAMETERS #########################
def uploading_global_parameters():
    """
    DESCRIPTION:
    Importing all of the parameters which will become global to this script.
    
    GLOBAL PARAMETERS:
    calib_state_val: [int]; The state value from where to extract the appropriate calibration data.
    car:             [bool (True/False)] Whether or not CAR filtering will be performed.
    car_channels:    [dictionary (key string (patient ID); Value: list > list > strings (channels))]; The sublists of
                     channels that are CAR filtered together.
    date:            [string]; The date from which the calibration statistics will be computed.
    dir_saving:      [string]; Directory where calibration statistics are saved.
    elim_channels:   [dictionary (key: string (patient ID); Value: list > strings (bad channels))]; The list of bad or 
                     non-neural channels to be exlucded from further analysis.
    exper_name:      [string]; Name of the experiment from which the calibration statistics will be computed.
    file_extension:  [string (hdf5/mat)]; The data file extension of the data.
    patient_id:      [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    sampling_rate:   [int (samples/s)]; Sampling rate at which the data was recorded.
    state_name:      [string]; BCI2000 state name that contains relevant state information.
    sxx_shift:       [int (units: ms)]; Length of time by which sliding window (sxx_window) shifts along the time domain.
    sxx_window:      [int (units: ms)]; Time length of the window that computes the frequency power.
    """
    
    # COMPUTATION:
    
    # Making each of the following parameters global variables.
    global calib_state_val
    global car
    global car_channels
    global date
    global dir_saving
    global elim_channels
    global exper_name
    global file_extension
    global patient_id
    global sampling_rate
    global state_name
    global sxx_shift
    global sxx_window
    
    # Importing the parameters. 
    calib_state_val = params_calibration_statistics.calib_state_val
    car             = params_calibration_statistics.car
    car_channels    = params_calibration_statistics.car_channels
    date            = params_calibration_statistics.date
    dir_saving      = params_calibration_statistics.dir_saving
    elim_channels   = params_calibration_statistics.elim_channels
    exper_name      = params_calibration_statistics.exper_name
    file_extension  = params_calibration_statistics.file_extension
    patient_id      = params_calibration_statistics.patient_id
    sampling_rate   = params_calibration_statistics.sampling_rate
    state_name      = params_calibration_statistics.state_name
    sxx_shift       = params_calibration_statistics.sxx_shift
    sxx_window      = params_calibration_statistics.sxx_window
    
    # PRINTING THE GLOBAL PARAMETERS.
    print('GLOBAL PARAMETERS TO functions_calibration_statistics.py SCRIPT')
    print('CALIB STATE VAL:   ', calib_state_val)
    print('CAR FILTERING:     ', car)
    print('CAR CHANNELS:      ', car_channels)
    print('DATE:              ', date)
    print('DIRECTORY SAVING:  ', dir_saving )
    print('EXCLUDED CHANNELS: ', elim_channels)
    print('EXPERIMENT NAME:   ', exper_name)
    print('FILE EXTENSION:    ', file_extension)
    print('PATIENT ID:        ', patient_id)
    print('SAMPLING RATE:     ', sampling_rate, 'Sa/s')
    print('STATE NAME:        ', state_name)
    print('SPECTRAL SHIFT:    ', sxx_shift, 'ms')
    print('SPECTRAL WINDOW:   ', sxx_window, 'ms')
    
# Immediately uploading global parameters.
uploading_global_parameters()




########################################## FUNCTIONS ##########################################

def calib_signals_relevant(calib_cont_dict):
    """
    DESCRIPTION:
    Only the signals and states where the state array equals the calibration state value are extracted.
    
    INPUT VARIABLES:
    calib_info:  [dictionary (Key/Value pairs below)];
        signals: [xarray (channels, time samples) > floats (units: microvolts)]; Array of continuous voltage signals. 
                 Time dimension is in units of seconds.
        states:  [xarray (time samples,) > ints (0 or 1)]; Array of states at each time sample. Time dimension is in
                 units of seconds.
                 
    GLOBAL PARAMETERS:
    calib_state_val: [int]; The state value from where to extract the appropriate calibration data.

    OUTPUT VARIABLES:
    calib_info:  [dictionary (Key/Value pairs below)];
        signals: [xarray (channels, time samples) > floats (units: microvolts)]; Array of continuous voltage signals 
                 from only the relevant time samples for the calibration tasks. Time dimension is in units of seconds.
        states:  [xarray (time samples,) > ints (0 or 1)]; Array of states from only the relevant time samples for the 
                 calibration tasks. Time dimension is in units of seconds.
    """
    
    # COMPUTATION:
    
    # Iterating across all tasks in the calibration data dictionary.
        
    # Extracting the calibration signals and states from the current task.
    signals = calib_cont_dict['signals']
    states  = calib_cont_dict['states']
    
    # Find indices where the states array only equals the calibration state value.
    state_val_inds = states == calib_state_val
    
    # Extracting only signals and states of the state value indices.
    calib_cont_dict['states']  = states[state_val_inds]
    calib_cont_dict['signals'] = signals[:,state_val_inds]
    
    return calib_cont_dict




def car_filter(calib_info):   
    """
    DESCRIPTION:
    The signals from the included channels will be extracted and referenced to their common average at each time point
    (CAR filtering: for each time point, subtracting the mean of all signals from each signal). The experimenter may 
    choose to CAR specific subset of channels or to CAR all the channels together. If the experimenter wishes to CAR 
    specific subsets of channels, each subset of channels should be written as a sublist, within a larger nested list.
    For example: [[ch1, ch2, ch3], [ch8, ch10, ch13]].
    
    INPUT VARIABLES:
    calib_info:  [dictionary (Key/Value pairs below)];
        signals: [xarray (channels, time samples) > floats (units: microvolts)]; Array of continuous voltage signals
                 from only the relevant time samples for the calibration tasks. Time dimension is in units of seconds.
        states:  [xarray (time samples,) > ints (0 or 1)]; Array of states from only the relevant time samples for the 
                 calibration tasks. Time dimension is in units of seconds.
        
    GLOBAL PARAMETERS:
    car:          [bool (True/False)]; Whether or not CAR filtering will be performed.
    car_channels: [dictionary (key string (patient ID); Value: list > list > strings (channels))]; The sublists of
                  channels that are CAR filtered together.
    patient_id:   [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
        
    OUTPUT VARIABLES:
    calib_info: Same as above, only the signals data may or may not be CAR filtered.
    """
 
    # If CAR filtering will happen.
    if car:
       
        # Extracting the signals from the current task block.
        signals = calib_info['signals']

        # Extracting all of the channel and time coordinates from the signals xarray.
        chs_include = list(signals.channel.values) 
        t_seconds   = signals.time.values
        
        # Extracting the signal values from the signals xarray for speed.
        signals = signals.values

        # Loading the sets of channels over which to independently apply CAR.
        specific_car_chs = car_channels[patient_id]

        # Ensuring that the CAR channels are actually in the included channels list. If a particular channel is not in
        # the included channels list, it will be removed from the CAR channel list.
        n_car_sets           = len(specific_car_chs)
        specific_car_chs_inc = [None]*n_car_sets
        for (n, this_car_set) in enumerate(specific_car_chs):
            this_car_set_verify     = [x for x in this_car_set if x in chs_include]
            specific_car_chs_inc[n] = this_car_set_verify
        specific_car_chs = specific_car_chs_inc

        # Extracting the total number of time samples in the signals array.   
        n_samples = signals.shape[1]

        # Initializing the CAR-ed signals array.
        signals_car = copy.deepcopy(signals)

        # If subsets of channels will be CAR-ed.
        if specific_car_chs:
            
            # Computing the number of specific CAR channel groups.
            n_sublists = len(specific_car_chs)

            # Iterating across each group.
            for s in range(n_sublists):

                # Extracting the specific CAR channels from the current group.
                these_specific_car_chs = specific_car_chs[s]   

                # Computing the number of channels to CAR from the current group.
                n_these_specific_car_chs = len(these_specific_car_chs)

                # Only perform CAR on the current group of channels if that group contains more than one channel.
                # If there is only one channel, the resulting CAR-ed activity will be equal to 0.
                if n_these_specific_car_chs > 1:

                    # Extracting the specific channel indices which will be CAR-ed.
                    car_ch_inds = [chs_include.index(ch) for ch in these_specific_car_chs]

                    # Iterating across all time samples.
                    for n in range(n_samples):

                        # CAR-ing only the specific channels in this particular subgroup.
                        signals_car[(car_ch_inds, n)] = signals[(car_ch_inds, n)] - np.mean(signals[(car_ch_inds, n)])

        # If all channels will be CAR-ed together.
        else:
            
            # Iterating through all time samples to CAR all channels.
            for n in range(n_samples):
                signals_car[:, n] = signals[:, n] - np.mean(signals[:, n])

        # Converting the CAR-ed signals back into an xarray.
        signals_car_xr = xr.DataArray(signals_car, 
                                      coords={'channel': chs_include, 'time': t_seconds}, 
                                      dims=["channel", "time"])

        # Updating the data dictionary with the CAR-ed signals.
        calib_info['signals'] = signals_car_xr  
            
    # If no CAR filtering will be applied.
    else:
        pass
    
    return calib_info




# def car_filter(signals):
#     """
#     DESCRIPTION:
#     The signals from the included channels will be extracted and referenced to their common average at each time point (CAR filtering: subtracting the mean of all signals 
#     from each signal at each time point). The user may choose to CAR specific subset of channels or to CAR all the channels. If the user wishes to CAR specific subsets of
#     channels, each subset of channels should be written as a sublist, within a larger nested list. For example: [[ch1, ch2, ch3],[ch8, ch10, ch13]].

#     INPUT VARIABLES:
#     signals: [array (samples x chs) > ints (units: microvolts)]; Array of raw time signals.

#     OUTPUT VARIABLES:
#     signals_car:      [array (N samples x N included chs) > floats (units: microvolts)]; The rerefenced common average voltage at each time point from each included channel.
#     specific_car_chs: [list > list > strings (chs)]; Experimenter-specified input channels over which to CAR. Each sublist refers to a subset of channels over which
#                       CAR will be specifically applied. Channels not included in these sublists will CAR-ed together.
#     """
    
#     # COMPUTATION:
    
#     # Extracting the included channels and date parameters and the specific car channel groups.
#     chs_include      = parameters_baseline_stats_generator.chs_include
#     specific_car_chs = parameters_baseline_stats_generator.car_channels
        
#     # Ensuring that the CAR channels are actually in the included channels list. If a particular channel is not in the included channels list, it will be removed from 
#     # the CAR channel list.
#     n_car_sets           = len(specific_car_chs)
#     specific_car_chs_inc = [None]*n_car_sets
#     for (n, this_car_set) in enumerate(specific_car_chs):
#         this_car_set_verify     = [x for x in this_car_set if x in chs_include]
#         specific_car_chs_inc[n] = this_car_set_verify
#     specific_car_chs = specific_car_chs_inc
    
#     # Extracting the total number of time samples in the signals array.   
#     n_samples = signals.shape[0]
    
#     # Initializing the CAR-ed signals array.
#     signals_car = copy.deepcopy(signals)
        
#     # If the user wishes to CAR only a specific subset of channels.
#     if specific_car_chs:
        
#         # Computing the number of specific CAR channel groups.
#         n_sublists = len(specific_car_chs)
        
#         # Iterating across each group.
#         for s in range(n_sublists):
            
#             # Extracting the specific CAR channels from the current group.
#             these_specific_car_chs = specific_car_chs[s]   
            
#             # Computing the number of channels to CAR from the current group.
#             N_these_specific_car_chs = len(these_specific_car_chs)
                        
#             # Only perform CAR on the current group of channels if that group contains more than one channel. If there is only one channel, the resulting CAR-ed
#             # activity will be equal to 0.
#             if N_these_specific_car_chs > 1:
                
#                 # Extracting the specific channel indices which will be CAR-ed.
#                 car_ch_inds = [chs_include.index(ch) for ch in these_specific_car_chs]
                                
#                 # Iterating across all time samples.
#                 for n in range(n_samples):
                    
#                     # CAR-ing only the specific channels in this particular subgroup.
#                     signals_car[(n, car_ch_inds)] = signals[(n, car_ch_inds)] - np.mean(signals[(n, car_ch_inds)])

#     # If the user wishes to CAR all channels.
#     else:
#         # Iterating through all time samples to CAR all channels.
#         for n in range(n_samples):
#             signals_car[n, :] = signals[n, :] - np.mean(signals[n, :])

#     return signals_car, specific_car_chs





def channel_selector(eeglabels):
    """
    DESCRIPTION:
    This function extracts all the channels to include in further analysis and excludes the experimenter-specified 
    channels for exclusion.

    INPUT VARIABLES:
    eeglabels: [array > strings (eeg channel names)]: EEG channels extracted from the .hdf5 or .mat file.
    
    GLOBAL PARAMETERS:
    elim_channels: [dictionary (key: string (patient ID); Value: list > strings (bad channels))]; The list of bad or 
                   non-neural channels to be exlucded from further analysis.
    patient_id:    [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    
    OUTPUT VARIABLES:
    chs_exclude: [list > strings]; The list of channels to be excluded in further analysis.
    chs_include: [list > strings]; The list of channels to be included in further analysis.
    """
    
    # COMPUTATION:
    
    # Extracting the additional channels for elimination.
    chs_exclude = elim_channels[patient_id]
            
    # Based on the bad channels, computing the included and excluded channels.
    chs_include = [n for n in eeglabels if n not in chs_exclude]
    
    # PRINTING:
    print('\nEXCLUDED CHANNELS:')
    print(chs_exclude)
    print('\nINCLUDED CHANNELS:')
    print(chs_include)
    
    return chs_exclude, chs_include





def computing_baseline_statistics(f_sxx, sxx):
    """
    DESCRIPTION:
    Computing the baseline mean and standard deviation for each frequency across each channel.
    
    INPUT VARIABLES:
    f_sxx: [array (1 x freqs) > floats (units: Hz)] ; Array of frequency values calculated by the spectrogram functionality. 
    sxx:   [array (freqs x samples x chs) > floats]; Array of power frequency values at each time point of the multi-dimensional (channel) signals.
    
    OUTPUT VARIABLES:
    base_sxx_mean:  [array (freqs x chs) > floats]; Array of mean values across time samples for each channel and frequency band.
    base_sxx_stdev: [array (freqs x chs) > floats]; Array of standard deviation values across time samples for each channel and frequency band.
    """
    
    # COMPUTATION:
    
    # Extracting the included channels and date parameters.
    chs_include = parameters_baseline_stats_generator.chs_include
    date        = parameters_baseline_stats_generator.date
    
    # Computing the mean and standard deviation.
    base_sxx_mean = np.mean(sxx, axis=1)
    base_sxx_stdev = np.std(sxx, axis=1)
    
    # Computing the number of included channels.
    n_chs = len(chs_include)
    
    # Plotting
    fig, ax = plt.subplots()
    #c = ax.pcolormesh(np.arange(len(selected_chs)), f_sxx[25:65], np.squeeze(base_sxx_means[0][25:65,:]), cmap = 'bwr', vmin=-3, vmax=-1)
    c = ax.pcolormesh(np.arange(n_chs), f_sxx[0:50], np.squeeze(base_sxx_mean[0:50,:]), cmap = 'bwr', vmin=-3, vmax=3)
    ax.set_title(date)
    ax.set_xlabel('Channels')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xticks(np.arange(n_chs))
    ax.set_xticklabels(chs_include, rotation = 90, fontsize=5)
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    
    return base_sxx_mean, base_sxx_stdev



def computing_calibration_statistics(calib_sxx):
    """
    DESCRIPTION:
    Computing the calibration mean and standard deviation for each frequency across each channel.
    
    INPUT VARIABLES:
    calib_sxx:       [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_signals: [xarray (channels, frequency bins, time samples) > floats (units: V^2/Hz)]; Spectral power of the 
                     continuous voltage signals.
        sxx_states:  [xarray (time samples,) > ints (0 or 1)]; States array downsampled to match time resolution of the
                     signal spectral power.
                     
    OUTPUT VARIABLES:
    calibration_sxx_mean   [xarray (channels, frequency bins) > floats (units: V^2/Hz)]; Calibration mean of each 
                           channel and frequency bin across time.
    calibration_sxx_stdev: [xarray (channels, frequency bins) > floats (units: V^2/Hz)]; Calibration standard deviation
                           of each channel and frequency bin across time.
    """
    
    # COMPUTATION:
    
    # Extracting the spectral signals from the dictionary.
    sxx_signals = calib_sxx['sxx_signals']

    # Computing the mean and standard deviation.
    calibration_sxx_mean  = np.mean(sxx_signals, axis=2)    
    calibration_sxx_stdev = np.std(sxx_signals, axis=2)
    
    # Extracting the channels and frequency bins.
    chs_include = sxx_signals.channel.values
    f_sxx       = sxx_signals.frequency.values
    
    # Computing the total number of channels. 
    n_chs_include = len(chs_include)
    
    print(chs_include)
    
    # PLOTTING:
    fig, ax = plt.subplots(figsize=(20,5))
    c = ax.pcolormesh(np.arange(n_chs_include), f_sxx[0:50], calibration_sxx_mean[:,0:50].transpose(), cmap = 'bwr',\
                      vmin=-3, vmax=3)
    ax.set_title(date)
    ax.set_xlabel('Channels')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xticks(np.arange(n_chs_include))
    ax.set_xticklabels(chs_include, rotation = 90, fontsize=10)
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    
    return calibration_sxx_mean, calibration_sxx_stdev



def data_upload(chs_include, eeglabels):
    """
    DESCRIPTION:
    For the calibration experiment, the signals and states at the time resolution of the continuous sampling rate are 
    uploaded into a data dictionary.
    
    INPUT VARIABLES:
    chs_include: [list > strings]; The list of channels to be included in further analysis.
    eeglabels:  [list > strings (eeg channel names)]: EEG channels extracted from the .hdf5 or .mat file.
    
    GLOBAL PARAMETERS:
    date:            [string]; The date from which the calibration statistics will be computed.
    exper_name:      [string]; Name of the experiment from which the calibration statistics will be computed.
    file_extension:  [string (hdf5/mat)]; The data file extension of the data.
    patient_id:      [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    sampling_rate:   [int (samples/s)]; Sampling rate at which the data was recorded.
    state_name:      [string]; BCI2000 state name that contains relevant state information.
    
    OUTPUT VARIABLES:    
    calib_info:  [dictionary (Key/Value pairs below)];
        signals: [xarray (channels, time samples) > floats (units: microvolts)]; Array of continuous voltage signals.
                 Time dimension is in units of seconds.
        states:  [xarray (time samples,) > ints (0 or 1)]; Array of states at each time sample. Time dimension is in
                 units of seconds.
    """
    
    # COMPUTATION:
    
    # Initializing a dictionary with the signals and state arrays recorded at the continuous sampling rate.
    data_cont_dict = collections.defaultdict(dict)
    
    # Creating the file pathway from where to upload the data.
    path = '/mnt/shared/ecog/' + patient_id + '/' + file_extension + '/' + date + '/' + exper_name +  '.' +\
           file_extension

    # Uploading the eeg signals and states, depending on whether they come from a .hdf5 or .mat file.
    if file_extension == 'hdf5':

        # Uploading the .hdf5 data.
        h5file      = H5EEGFile(path)
        eeg         = h5file.group.eeg(); 
        eeg_signals = eeg.dataset[:]

        # Uploading the .hdf5 states.
        aux    = h5file.group.aux();
        states = aux[:, state_name]

    if file_extension == 'mat':

        # Uploading the .mat data.
        matlab_data = loadmat(path, simplify_cells=True)
        eeg_signals = matlab_data['signal']

        # Uploading the .mat states.
        states = matlab_data['states'][state_name]

    # Multiplying the EEG signals by a gain that's done downstream of the .dat collection in BCI2000.
    eeg_signals = eeg_signals * 0.25

    # Computing the total number of channels and time samples.
    n_samples  = eeg_signals.shape[0]
    n_channels = len(chs_include)

    # Creating the time array. Note that the first time sample is not 0, as the first recorded signal sample does not
    # correspond to a 0th time.
    t_seconds = (np.arange(n_samples) + 1)/sampling_rate    

    # Converting the states array into an xarray.    
    states = xr.DataArray(states,
                          coords={'time': t_seconds},
                          dims=["time"])

    # Initializing an xarray for the signals.
    signals = xr.DataArray(np.zeros((n_channels, n_samples)), 
                           coords={'channel': chs_include, 'time': t_seconds}, 
                           dims=["channel", "time"])

    # Populating the signals xarray with the eeg signals.
    for ch_name in chs_include:

        # Extracting the appropriate channel index from the original eeglabels list and populating the
        # signals xarray with the channel's activity.
        eeg_ind              = eeglabels.index(ch_name)            
        signals.loc[ch_name] = eeg_signals[:,eeg_ind]

    # Populating the data dictionary with the signals and stimuli arrays.
    data_cont_dict['signals'] = signals
    data_cont_dict['states']  = states
            
    return data_cont_dict
    
    
    
    

def data_extraction(eeglabels, path, stimulus_marker):
    """
    DESCRIPTION:
    The neural signals as well as state signals are extracted here. The time vector is also defined in terms of seconds and samples.
    
    INPUT VARIABLES:
    eeglabels:       [array > strings (eeg channel names)]: EEG channels extracted from the .hdf5 or .mat file.
    path_list:       [list > strings (pathways)]; The list of file pathways from which we will extract our dataset(s).
    stimulus_marker: [string]; The type of stimulus marker the experimenter wishes to extract.
    
    OUTPUT VARIABLES:
    sampling_rate: [int (samples/s)]; Sampling rate at which the data was recorded.
    signals:       [array (samples x chs) > floats (units: microvolts)]; Array of raw time signals.
    stimuli:       [array (samples x 1) > ints]; Array of presented stimulus values at each time stamp.
    t_samples:     [array (samples x 1) > ints (units: samples)]; Array of the samples at each time point.
    t_seconds:     [array (samples x 1) > floats (units: s)]; Array of the time stamps at each time point. 
    """
    # COMPUTATION:
    
    # Extracting the included channels and file extension parameters.
    chs_include    = parameters_baseline_stats_generator.chs_include
    file_extension = parameters_baseline_stats_generator.file_extension

    # Extracting the total number of channels (eeg + analog).
    n_chs_all = len(eeglabels)

    # Extracting the eeg signals for the current date, depending on whether they come from an .hdf5 or .mat file.
    if file_extension == 'hdf5':

        # Extracting the current date's .hdf5 data.
        h5file      = H5EEGFile(path)
        eeg         = h5file.group.eeg(); 
        eeg_signals = eeg.dataset[:]

        # Extracting the current session's .hdf5 stimuli.
        aux = h5file.group.aux();
        stimuli = aux[:, stimulus_marker]

    if file_extension == 'mat':

        # Extracting the current date's .mat data.
        matlab_data = loadmat(path, simplify_cells=True)
        eeg_signals = matlab_data['signal']

        # Extracting the current date's .mat stimuli.
        stimuli = matlab_data['states'][stimulus_marker]
        
    # Converting the stimuli_pre to integers.
    stimuli = stimuli.astype(int)
    
    # Multiplying our eeg signals by a gain that's done downstream of the .dat collection in BCI2000.
    eeg_signals = eeg_signals * 0.25

    # Initializing the signals list and channel index.
    n_chs_include = len(chs_include)
    signals_list  = [None] * n_chs_include

    # Iterating across each name in the included channels list.
    for (ch_ind, ch_name) in enumerate(chs_include):

        # Extracting the appropriate channel index from the original eeglabels list.
        eeg_ind = np.argwhere(eeglabels == ch_name)[0][0]
        
        # Assigining the signals from the mat file to the appropriate index in the signals list.
        signals_list[ch_ind] = eeg_signals[:,eeg_ind]

    # Converting the signals_list to the proper array format.
    signals = np.squeeze(np.array(signals_list).transpose())
    
    # If the signals don't have enough dimensions (only one channel).
    if n_chs_include == 1:
        signals = np.expand_dims(signals,1) 
    
    # Extracting the sampling rate.
    if file_extension == 'hdf5':
        sampling_rate = int(eeg.get_rate())
    if file_extension == 'mat':
        sampling_rate = int(matlab_data['parameters']['SamplingRate']['NumericValue'])     
        
        # Create the eeg Object witht the (with .get_rate attribute).
        class MyEEG:
            def __init__(self, sampling_rate):
                self.sampling_rate = sampling_rate
            def get_rate(self):
                return self.sampling_rate

        # Create instance of the class MyEEG to form eeg.get_rate()\
        eeg = MyEEG(sampling_rate)
        
    print('Sampling rate (sa/s):')
    print(sampling_rate)

    # Creating the time signals in units of seconds and samples
    n_samples = len(stimuli)
    t_samples = np.arange(0,n_samples)
    t_seconds = t_samples/eeg.get_rate()
    
    # PLOTTING STIMULI:
    fig = plt.figure(figsize=(20,5))
    plt.plot(t_seconds, stimuli)
    plt.xlabel('Time (s)')
    plt.ylabel('Stimulus Value')
    plt.title('Stimulus Over Time')
        
    
    return sampling_rate, signals, stimuli, t_samples, t_seconds





def extract_calibration_data(signals, stimuli, t_samples, t_seconds):
    """
    DESCRIPTION:
    Extracting the arrays of signals, time stamps and samples only during the calibration stimulus.
    
    INPUT VARIABLES:
    signals:   [array (samples x chs) > floats (units: microvolts)]; Array of raw time signals.
    stimuli:   [array (samples x 1) > ints]; Array of presented stimulus values at each time stamp.
    t_samples: [array (samples x 1) > ints (units: samples)]; Array of the samples at each time point.
    t_seconds: [array (samples x 1) > floats (units: s)]; Array of the time stamps at each time point. 

    OUTPUT VARIABLES:
    signals:   [array (samples x chs) > floats (units: microvolts)]; Array of raw time signals only during the calibration stimulus.
    t_samples: [array (samples x 1) > ints (units: samples)]; Array of the samples at each time point only during the calibration stimulus.
    t_seconds: [array (samples x 1) > floats (units: s)]; Array of the time stamps at each time point only during the calibration stimulus.
    """
    # Extracting the calibration stimulus.
    calibration_stimulus = parameters_baseline_stats_generator.calibration_stimulus
    
    # Extracting the arrays of signals, time stamps and samples only during the calibration stimulus.
    signals   = signals[stimuli == calibration_stimulus,:]
    t_samples = t_samples[stimuli == calibration_stimulus]
    t_seconds = t_seconds[stimuli == calibration_stimulus]
    
    return t_samples, t_seconds, signals





# def extract_data_pathway():
#     """
#     DESCRIPTION:
#     Using the parameter file, creating the date file pathway.
        
#     OUTPUT:
#     path: [string (pathways)]; The file pathway from which we will extract our dataset.
#     """
    
#     # COMPUTATION:

#     # Extracting the parameters for creating the file pathway.
#     date           = parameters_baseline_stats_generator.date
#     file_extension = parameters_baseline_stats_generator.file_extension
#     patient_id     = parameters_baseline_stats_generator.patient_id
#     task           = parameters_baseline_stats_generator.task
    
#     # Creating the path for the date and task for calibration.
#     path = '/mnt/shared/ecog/' + patient_id + '/' + file_extension + '/' + date + '/' + task + '.' + file_extension
    
#     print('Pathway:')
#     print(path)
    
#     return path





# def import_electrode_information(path):
#     """
#     DESCRIPTION:
#     The eeglabels and auxlabels lists will be populated with eeg channel names and auxilliary channel names respectively. These lists are created differently 
#     based on whether the data is extracted from a .hdf5 file or a .mat file.

#     INPUT VARIABLES:
#     path: [strings (file pathway)]; The file pathways from which we will extract our dataset(s).

#     OUTPUT VARIABLES:
#     auxlabels: [array > strings (aux channel names)]: Auxilliary channels extracted from the .hdf5 or .mat file.
#     eeglabels: [array > strings (eeg channel names)]: EEG channels extracted from the .hdf5 or .mat file.
#     """
#     # COMPUTATION:
    
#     # Defining the ecog object.
#     ecog = ECoGConfig()
    
#     # Extracting the file extension parameter.
#     file_extension = parameters_baseline_stats_generator.file_extension

#     # Extracting the channel and auxilliary labels from the .hdf5 file.
#     if file_extension == 'hdf5':

#         h5file    = H5EEGFile(path) # reads all patient data after calling it in.
#         eeg       = h5file.group.eeg()
#         aux       = h5file.group.aux()
#         eeglabels = eeg.get_labels(); 
#         auxlabels = aux.get_labels()
        
#     # Extracting the channel and auxilliary labels from the .mat file.
#     if file_extension == 'mat':

#         matlab_file0 = loadmat(path, simplify_cells=True)
#         eeglabels    = matlab_file0['parameters']['ChannelNames']['Value']    
#         auxlabels    = list(matlab_file0['states'].keys())

#     print('EEG LABELS:')
#     print(eeglabels)
    
#     print('\nAUX LABELS:')
#     print(auxlabels)
    
#     return auxlabels, eeglabels 


def import_electrode_information():
    """
    DESCRIPTION:
    The eeglabels and auxlabels lists will be populated with eeg channel names and auxilliary channel names 
    respectively. These lists are created differently based on whether the data is extracted from a .hdf5 file or a .mat
    file.
    
    GLOBAL PARAMETERS:
    date:           [string]; The date from which the calibration statistics will be computed.
    exper_name:     [string]; Name of the experiment from which the calibration statistics will be computed.
    file_extension: [string (hdf5/mat)]; The data file extension of the data.
    patient_id:     [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    
    OUTPUT VARIABLES:
    auxlabels: [array > strings (aux channel names)]: Auxilliary channels extracted from the .hdf5 or .mat file.
    eeglabels: [list > strings (eeg channel names)]: EEG channels extracted from the .hdf5 or .mat file.
    """
    
    # COMPUTATION:
    
    # Creating the path from which the calibration data will be extracted.
    path = '/mnt/shared/ecog/' + patient_id + '/' + file_extension + '/' + date + '/' + exper_name + '.' +\
           file_extension
    
    # Extracting the channel and auxilliary labels from the .hdf5 file.
    if file_extension == 'hdf5':
        h5file    = H5EEGFile(path)
        eeg       = h5file.group.eeg()
        aux       = h5file.group.aux()
        eeglabels = eeg.get_labels().tolist(); 
        auxlabels = aux.get_labels()
        
    # Extracting the channel and auxilliary labels from the .mat file.
    if file_extension == 'mat':
        matlab_file0 = loadmat(path, simplify_cells=True)
        eeglabels    = matlab_file0['parameters']['ChannelNames']['Value'].tolist()   
        auxlabels    = list(matlab_file0['states'].keys())

    # PRINTING
    print('EEG LABELS: \n', eeglabels); print('\n')
    print('AUX LABELS: \n', auxlabels)
    
    return auxlabels, eeglabels



def plotting_states(calib_info):
    """
    DESCRIPTION:
    Plotting the stimuli of an experimenter-specified task such as to observe how many state values there are. This
    will inform the state mapping.
    
    INPUT VARIABLES:
    calib_info:  [dictionary (Key/Value pairs below)];
        signals: [xarray (channels, time samples) > floats (units: microvolts)]; Array of continuous voltage signals. 
                 Time dimension is in units of seconds.
        states:  [xarray (time samples,) > ints (0 or 1)]; Array of states at each time sample. Time dimension is in 
                 units of seconds.
    """
    
    # COMPUTATION:
    
    # Extracting the time points and states for the calibration task.
    states = calib_info['states']
    time   = states.time

    # PLOTTING
    
    # Plotting the states from the calibration task.
    fig = plt.figure(figsize=(20,5))
    plt.plot(time, states)
    plt.xlabel('Time (s)')
    plt.ylabel('State Values')
    plt.title('Calibration States')

    return None





# def spectrogram_generator(sampling_rate, signals):
#     """
#     DESCRIPTION: 
#     This function takes a 2D array of time signals from multiple channels (time samples X channels) and uses a sliding window to compute the frequency 
#     power at each time point.  
    
#     INPUTS:
#     sampling_rate: [int (samples/s)]; Sampling rate at which the data was recorded.
#     shift:         [int (units: ms)]; Length of time by which sliding window shifts along the time domain.
#     signals:       [array (samples x chs) > floats]; Array of raw time signals.
#     window_length: [int (units: ms)]; Time length of the window that computes the frequency power.
    
#     OUTPUTS:
#     f_sxx: [array (1 x freqs) > floats (units: Hz)] ; Array of frequency values calculated by the spectrogram functionality. 
#     sxx:   [array (freqs x samples x chs) > floats]; Array of power frequency values at each time point of the multi-dimensional (channel) signals.
#     t_sxx: [array (1 x samples) > floats (units: s)]; Array of spectrogram time samples.
#     """

#     # COMPUTATION:
    
#     # Extracting the spectrogram parameters.
#     shift         = parameters_baseline_stats_generator.shift
#     window_length = parameters_baseline_stats_generator.window_length
    
#     # Computing the constant (units: samples/ms) which is used to correct for signal sampling rates that differ from 1000 samples/s.
#     factor = int(sampling_rate/1000)  
    
#     # Resizing the spectral window and shift to account for the sampling rate.
#     window_length = window_length * factor
#     shift         = shift * factor
        
#     # Computing the spectrogram input parameters.
#     f_s        = 1e3*factor            # [units: samples/s] Sampling rate for sliding window Fourier transform.
#     my_overlap = window_length - shift # [units: ms]; Overlap between sliding windows.
#     my_nperseg = window_length         # Number of samples in one sliding window
    
#     # Computing the spectrogram of the signals over all channels. 
#     f_sxx, t_sxx, sxx = signal.spectrogram(signals, fs = f_s, noverlap = my_overlap, nperseg = my_nperseg, axis = 0)
        
#     # Re-orienting the spectrogram. 
#     sxx = np.moveaxis(sxx, -2, -1)

#     # Logging the spectrograms. 
#     sxx = np.log10(sxx)

#     # Replacing potential -infinity values or NANs (from log10) with 0.
#     sxx[np.isneginf(sxx)] = 0
#     sxx[np.isnan(sxx)]    = 0
    
#     return f_sxx, t_sxx, sxx


def save_statistics(calib_sxx_mean, calib_sxx_stdev):
    """
    DESCRIPTION:
    Saving the calibration statistics.
    
    INPUT VARIABLES:
    calib_sxx_mean   [xarray (channels, frequency bins) > floats (units: V^2/Hz)]; Calibration mean of each channel and 
                     frequency bin across time.
    calib_sxx_stdev: [xarray (channels, frequency bins) > floats (units: V^2/Hz)]; Calibration standard deviation of
                     each channel and frequency bin across time.
    
    GLOBAL PARAMETERS:
    date:       [string]; The date from which the calibration statistics will be computed.
    dir_saving: [string]; Directory where calibration statistics are saved.    
    """
    
    dir_saving = '/mnt/shared/danprocessing/Projects/PseudoOnlineTests_for_RTCoG/Parms_for_RTCoG/' + patient_id +\
                 '/Calibration_Statistics/'
                 
    # Computing the foldername containing the calibration statistics
    foldername = date + '_calibration_statistics'
    
    # Creating the dictionary of calibration statistics to be saved.
    calib_statistics = {'calib_sxx_mean': calib_sxx_mean, 'calib_sxx_stdev': calib_sxx_stdev}

    # SAVING:
    
    # Creating the pathway where the contents of the dictionary will be saved.
    path = dir_saving + foldername
    
    # If the path doesn't exist, create it.
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Iterating across each item in the dictionary.
    for pair in calib_statistics.items():
        
        # Extracting the key and value of the current item. 
        key   = pair[0]
        value = pair[1]
        
        # Saving the value under the name of the key in the appropriate path.
        with open(path + '/' + key, 'wb') as (fp): pickle.dump(value, fp)

    return None





def spectrogram_generator(calib_info):
    """
    DESCRIPTION:
    Creating the spectrogram for each channel across all tasks. Also downsampling the states array to match the 
    spectrogram resolution.
    
    INPUT VARIABLES:
    calib_info:  [dictionary (Key/Value pairs below)];
        signals: [xarray (channels, time samples) > floats (units: microvolts)]; Array of continuous voltage signals 
                 from only the relevant time samples for the calibration tasks. Time dimension is in units of seconds.
        states:  [xarray (time samples,) > ints (0 or 1)]; Array of states from only the relevant time samples for the 
                 calibration tasks. Time dimension is in units of seconds.
    
    GLOBAL PARAMETERS:
    sxx_shift:  [int (units: ms)]; Length of time by which sliding window (sxx_window) shifts along the time domain.
    sxx_window: [int (units: ms)]; Time length of the window that computes the frequency power.
    
    OUTPUT VARIABLES:
    calib_sxx:       [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_signals: [xarray (channels, frequency bins, time samples) > floats (units: V^2/Hz)]; Spectral power of the 
                     continuous voltage signals.
        sxx_states:  [xarray (time samples,) > ints (0 or 1)]; States array downsampled to match time resolution of the
                     signal spectral power.
    """
    
    # COMPUTATION:
    
    # Initializing the dictionary of spectral information.
    calib_sxx = collections.defaultdict(dict)

    # Extracting the signals and states of the current task from the data dictionary.
    signals = calib_info['signals']
    states  = calib_info['states']

    # Computing the constant which is used to correct for signal sampling rates that differ from 1000 samples/s.
    factor = int(sampling_rate/1000) # samples/s / (samples/s) = unitless

    # Resizing the spectral window and shift to account for the sampling rate.
    sxx_window_samples = sxx_window * factor # samples x (unitless) = samples
    sxx_shift_samples  = sxx_shift * factor  # samples x (unitless) = samples

    # Computing the spectrogram input parameters.
    f_s        = 1e3*factor                             # samples/s * unitless = samples/s
    my_overlap = sxx_window_samples - sxx_shift_samples # samples
    my_nperseg = sxx_window_samples                     # samples

    # Computing the spectrogram of the signals over all channels. 
    f_sxx, t_sxx, sxx = signal.spectrogram(signals, fs = f_s, noverlap = my_overlap, nperseg = my_nperseg, axis = 1)

    # Logging the spectrograms. 
    sxx = np.log10(sxx)
    
    # Replacing potential -infinity values or NANs (from log10) with 0.
    sxx[np.isneginf(sxx)] = 0
    sxx[np.isnan(sxx)]    = 0
    
    # Transforming the time array of the spectral signals such that the signals are causal. This means that rather than 
    # the first spectral sample being associated with 128 ms in the t_sxx array (if sxx_window = 256), it will be 
    # associated with 256 ms.
    t_sxx = t_sxx + (sxx_window/2)/1000

    # Converting the spectral array into an xarray.
    sxx_signals = xr.DataArray(sxx, 
                               coords={'channel': signals.channel, 'frequency': f_sxx, 'time': t_sxx}, 
                               dims=["channel", "frequency", "time"])
    
    # Downsampling the states such as to be at the same time resolution as the spectral signals. The resulting time 
    # coordinates will be causal.
    sxx_states = states[sxx_window_samples-1::sxx_shift_samples]
    
    # Updating the data dictionary.
    calib_sxx['sxx_signals'] = sxx_signals
    calib_sxx['sxx_states']  = sxx_states

    return calib_sxx



