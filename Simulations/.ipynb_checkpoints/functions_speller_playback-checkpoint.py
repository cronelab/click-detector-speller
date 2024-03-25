
import collections
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.signal as signal
import shutil
import tensorflow as tf
import time
import xarray as xr

from ecogconf import ECoGConfig
from h5eeg import H5EEGFile
from tqdm import trange

from scipy.io import loadmat, savemat

import time

def save_script_backup():
    
    """
    Automatically saving this entire script immediately when it's called.
    """
    
    # Creating the original and target file directories.
    original = r'/home/dan/Projects/PseudoOnlineTests_for_RTCoG/Scripts/Recent/SpellerAnalysis/functions_speller_playback.py'
    target   = r'/mnt/shared/danprocessing/BACKUP/Projects/PseudoOnlineTests_for_RTCoG/Scripts/SpellerAnalysis/functions_speller_playback.py'

    # Saving.
    shutil.copyfile(original, target)
    
# Immediately saving script.   
save_script_backup()






def adjusting_time_coords_to_block_start(data_exper_dict, t_start):
    """
    DESCRIPTION:
    Adjusting the time coordinates of the signals and states xarrays such that the starting time is the true
    starting time of the block.
    
    INPUT VARIABLES:
    data_exper_dict: [dict (Key: string (task ID); Values: dict (Key/Value pairs below))]; 
        signals:     [xarray (channels x time samples) > floats (units: microvolts)]; Array of raw time signals
                     of the experimental task across all channels. Time samples are in units of seconds.
        states:      [xarray (1 x time samples) > ints]; Array of states at each time sample of the experimental
                     task. Time samples are in units of seconds.
    t_start:         [float (units: s)]; True starting time of the block.
    
    OUTPUT VARIABLES:
    data_exper_dict: Same as above, only the time coordinates of the signals and states xarrays have been 
                     adjusted so that the starting time is t_start.
    """
    
    # COMPUTATION:
    
    # Extracting the seconds array from the signals xarray. This time array is the same for 
    # the states xarray.
    t_seconds = data_exper_dict['signals'].time.values

    # Zeroing the time array. 
    t_zeroed = t_seconds - t_seconds[0]

    # Adjusting the time array such that the first time point starts at the correct time.
    t_adjusted = t_zeroed + t_start

    # Overwriting the time coordinates of the signals and states xarrays.
    data_exper_dict['signals'] = data_exper_dict['signals'].assign_coords({"time": t_adjusted})
    data_exper_dict['states']  = data_exper_dict['states'].assign_coords({"time": t_adjusted})
    
    return data_exper_dict





def apply_preprocessing(calibration_sxx_mean, calibration_sxx_stdev, data_packets_dict):
    """
    DESCRIPTION:
    For each signal packet, computing the historical time features into a list.
    
    INPUT VARIABLES:
    calibration_sxx_mean   [xarray (channels, frequency bins) > floats (units: V^2/Hz)]; Calibration mean of each
                           channel and frequency bin across time.
    calibration_sxx_stdev: [xarray (channels, frequency bins) > floats (units: V^2/Hz)]; Calibration standard deviation
                           of each channel and frequency bin across time.
    data_packets_dict:     [dictionary (Keys/Value pairs below)];
        states:            [list > strings]; List of state strings corresponding to each data packet. 
        signals:           [list > array (channels, time samples) > floats (units: uV)]; List of data packets where each
                           packet contains continuous voltage data.
        t_seconds:         [list > floats (units: s)]; List of time points for each packet.
        
    GLOBAL PARAMETERS:
    buffer_size:   [float (units: ms)]; Window length of raw signal used for computing spectrogram, which continuously updates.
    sampling_rate: [int (samples/s)]; Sampling rate at which the data was recorded.
    t_history:     [float (unit: ms)]; Amount of feature time history.
    
    NECESSARY FUNCTIONS:
    buffer_raw_update
    buffer_sxx_update
    car_filter
    concatenating_power_bands
    mean_centering
    pc_transform
    power_generator
    spectrogram_generator
    standardize_to_calibration
    
    OUTPUT VARIABLES:
    feature_packets: [dictionary (key: packet number; Values: below)]; Dictionary containing packet data and corresponding
                     time for each packet.
        data:        [xarray (pc features, time samples) > floats (units: V^2/Hz)]; All historical time features for each 
                     packet.
        time:        [float (units: s)]; Corresponding time for feature packet.
    """
    
    # COMPUTATION:
    
    # Extracting the signal packets from the data packets dictionary.
    signal_packets = data_packets_dict['signals']
    
    # Computing the number of time samples in history.
    n_history = int(t_history/packet_size)
   
    # Extracting the number of included channels from the first signal packet. This number will be the same across
    # all signal packets.
    n_chs_include = int(signal_packets[0].channel.shape[0])
    
    # Computing the sampling rate factor.
    factor = sampling_rate/1000 # sa/s / (ms/s) = sa/s * s/ms = sa/ms
    
    # Computing the number of samples in the buffer.
    n_buffer_samples = int(buffer_size * factor) # ms * sa/ms = samples
    
    # Initializing a buffer for the raw data.
    buffer_raw = np.zeros((n_chs_include, n_buffer_samples))

    # Computing the total number of packets.
    n_packets = len(signal_packets)

    
    # Initializing the dictionary with per-packet data and time information.
    feature_packets = collections.defaultdict(dict)
    for p in range(n_packets):
        feature_packets[p]['data'] = None
        feature_packets[p]['time'] = None
        
    # Iterating across each packet.
    for p in trange(n_packets, position=0, desc='Iterating across all signal packets:'):

        # Extracting the current signal packet and transforming it from an xarray into an array.
        signal_packet = signal_packets[p]
        
        # Potentially CAR filtering the signal packet.
        signal_packet = car_filter(signal_packet)
        
        # Updating the buffer of raw signals (continuous voltage time signals).
        buffer_raw = buffer_raw_update(buffer_raw, signal_packet)

        # Computing the spectrogram of the current buffer window.
        sxx = spectrogram_generator(buffer_raw)
        
        # Initializing the spectrogram buffer once.
        if p == 0:
                        
            # Need to do this here because the number of frequency bands (n_fxx) has not yet been defined.
            n_fxx        = sxx.frequency.shape[0]            
            buffer_sxx_z = xr.DataArray(np.zeros((n_chs_include, n_fxx, n_history)),
                                        coords={'channel': sxx.channel, 'frequency': sxx.frequency}, 
                                        dims=["channel", "frequency", "time"])
        
        # Standardizing the signals for each frequency band to the calibration statistics.
        sxx_z = standardize_to_calibration(calibration_sxx_mean, calibration_sxx_stdev, sxx)
        
        # Updating the buffer of spectral features with the spectral features of the current packet.
        buffer_sxx_z = buffer_sxx_update(buffer_sxx_z, sxx_z)
        
        # Computing the powerband features from the spectrogram 
        features_all_bands = power_generator(buffer_sxx_z)
        
        # Concatenating features across all powerbands.
        features_bands_concat = concatenating_power_bands(features_all_bands)
        
        # Creating historical time features.
        features_all_history = concatenating_historical_features(features_bands_concat)
                
        # Mean-centering the current packet's features.
        features_mean_centered = mean_centering(features_all_history, training_data_mean)

        # Computing the reduced-dimension PC of the feature packet.
        this_feature_packet = pc_transform(features_mean_centered, eigenvectors)
        
        # Updating the list of feature packets and corresponding times.
        feature_packets[p]['data'] = this_feature_packet
        feature_packets[p]['time'] = data_packets_dict['t_seconds'][p]
                
    return feature_packets 



                                           
                                       
def buffer_raw_update(buffer_raw, packet):
    """
    DESCRIPTION:
    The buffer of raw signals (continuous voltage time signals) gets updated with each new packet of data. The 
    previous packet's worth of data is dropped from the buffer.
    
    INPUT VARIABLES:
    buffer_raw: [array (channels, time samples) > floats (units: microvolts)]; Buffer of raw time signals across
                all channels. Time samples are in units of seconds.
    packet:     [xarray (channels, time samples) > floats (units: microvolts)]; Array of packet's worth of time 
                signals across all channels. Time samples are in units of seconds.
    
    OUTPUT VARIABLES:
    buffer_raw: [array (channels, time samples) > floats (units: microvolts)]; Updated buffer of raw time signals
                across all channels. Time samples are in units of seconds.
    """                             
    
    # COMPUTATION:
    
    # Computing the number of samples in the buffer and in the packet.
    buffer_raw_samples = buffer_raw.shape[1]
    packet_samples     = packet.time.shape[0]
                              
    # Computing the indices in the buffer window for the old buffer data which will be removed.
    buffer_raw_old_ind = buffer_raw_samples - packet_samples
    buffer_raw_new_ind = buffer_raw_samples
    
    # Keeping all but the earliest packet's worth of data in the old buffer.
    buffer_raw_old = buffer_raw[:, packet_samples:]                            
                                       
    # Updating the buffer by concatenating the new packet to the old data. 
    buffer_raw[:, 0:buffer_raw_old_ind,]                 = buffer_raw_old
    buffer_raw[:, buffer_raw_old_ind:buffer_raw_new_ind] = np.asarray(packet)
                                               
    return buffer_raw





def buffer_sxx_update(buffer_sxx, sxx):
    """
    DESCRIPTION:
    Updating the buffer of spectral features. The number of time samples in this buffer are equal to the number of time
    points in the historical time features.
    
    INPUT VARIABLES:
    buffer_sxx: [xarray (channels, frequency, time samples) > floats (units: V^2/Hz)]; Updating buffer of spectral power, 
                where each new spectrogram entry is one time sample long. Time samples are in units of seconds.
    sxx:        [xarray (channels, frequency, time samples) > floats (units: V^2/Hz)]; Spectrogram array for only one time
                sample. 
    
    OUTPUT VARIABLES:
    buffer_sxx: [array (channels, frequency, time samples) > floats (units: V^2/Hz)]; Updated spectrogram buffer.
    """
    
    # COMPUTATION:
    
    # Updating the spectrogram buffer by re-using spectrogram information of all but the earliest time point. The spectrogram
    # of the new time point is concatenated at the leading edge of the buffer.
    buffer_sxx[:,:,:-1] = buffer_sxx[:,:,1:]
    buffer_sxx[:,:,-1]  = np.squeeze(np.asarray(sxx))
    
    return buffer_sxx





def calib_signals_relevant(calib_state_str, data_calib_dict):
    """
    DESCRIPTION:
    Only extracting the signals and states where the state equals the calibration state value.
    
    INPUT VARIABLES:
    calib_state_rel: [string]; The string state corresponding to the time samples from where to extract the
                     calibration data for computing baseline statistics.
    data_calib_dict: [dict (Key: string (task ID); Values: xarrays (below))]; 
        signals:     [xarray (channels, time samples) > floats (units: microvolts)]; Array of raw time signals
                     of the calibration task across all channels. Time samples are in units of seconds.
        states:      [xarray (1 x time samples) > ints]; Array of states at each time sample of the calibration
                     task. Time samples are in units of seconds.
    
    OUTPUT VARIABLES:
    signals_calib: [xarray (channels, time samples) > floats (units: microvolts)]; Array of raw time signals
                   across all channels  from only the time period with the appropriate calibration state. Time
                   samples are in units of seconds.
    """
    
    # COMPUTATION:
        
    # Extracting the calibration signals and states.
    signals = data_calib_dict['signals']
    states  = data_calib_dict['states']

    # Find indices where the states array only equals the relevant calibration states string.
    state_str_inds = states == calib_state_str

    # Extracting only signals of the state value indices.
    signals_calib = signals[:,state_str_inds]
    
    return signals_calib





def car_filter(signals):   
    """
    DESCRIPTION:
    The signals from the included channels will be extracted and referenced to their common average at each time
    point (CAR filtering: subtracting the mean of all signals from each signal at each time point). The experimenter
    may choose to CAR specific subset of channels or to CAR all the channels together. If the experimenter wishes to
    CAR specific subsets of channels, each subset of channels should be written as a sublist, within a larger nested
    list. For example: [[ch1, ch2, ch3],[ch8, ch10, ch13]].
    
    INPUT VARIABLES:
    signals: [xarray (channels, time samples) > floats (units: microvolts)]; Array of continuous voltage signals 
             from all channels. Time samples are in units of seconds.
        
    GLOBAL PARAMETERS:
    car:         [bool (True/False)] Whether or not CAR filtering will be performed.
    car_chs:     [list > list > strings]; Sublists of channels which will get CAR filtered independently of channels in 
                 other sublists.
    chs_include: [list > strings]; The list of channels to be included in further analysis.
        
    OUTPUT VARIABLES:
    signals_car: Same as above, only the signals data may or may not be CAR filtered.
    """
  
    # If CAR filtering will happen.
    if car:
        
        # Extracting all of the channel and time coordinates from the signals xarray.
        chs_include = list(signals.channel.values) 
        t_seconds   = signals.time

        # Extracting the signal values from the signals xarray for speed.
        signals = signals.values

        # Ensuring that the CAR channels are actually in the included channels list. If a particular channel is not in the 
        # included hannels list, it will be removed from the CAR channel list.
        n_car_sets  = len(car_chs)
        car_chs_inc = [None]*n_car_sets
        for (n, this_car_set) in enumerate(car_chs):
            this_car_set_verify = [x for x in this_car_set if x in chs_include]
            car_chs_inc[n]      = this_car_set_verify
        specific_car_chs = car_chs_inc

        # Extracting the total number of time samples in the signals array.   
        n_samples = signals.shape[1]

        # Initializing the CAR-ed signals array.
        signals_car = copy.deepcopy(signals)

        # If the user wishes to CAR only a specific subset of channels.
        if specific_car_chs:

            # Computing the number of specific CAR channel groups.
            n_sublists = len(specific_car_chs)

            # Iterating across each group.
            for s in range(n_sublists):

                # Extracting the specific CAR channels from the current group.
                these_car_chs = specific_car_chs[s]   

                # Computing the number of channels to CAR from the current group.
                n_these_car_chs = len(these_car_chs)

                # Only perform CAR on the current group of channels if that group contains more than one channel. If there is only one 
                # channel, the resulting CAR-ed activity will be equal to 0.
                if n_these_car_chs > 1:

                    # Extracting the specific channel indices which will be CAR-ed.
                    car_ch_inds = [chs_include.index(ch) for ch in these_car_chs]

                    # Iterating across all time samples.
                    for n in range(n_samples):

                        # CAR-ing only the specific channels in this particular subgroup.
                        signals_car[(car_ch_inds, n)] = signals[(car_ch_inds, n)] - np.mean(signals[(car_ch_inds, n)])

        # If the user wishes to CAR all channels.
        else:
            
            # Iterating through all time samples to CAR all channels.
            for n in range(n_samples):
                signals_car[:, n] = signals[:, n] - np.mean(signals[:, n])

        # Converting the CAR-ed signals back into an xarray.
        signals_car = xr.DataArray(signals_car, 
                                   coords={'channel': chs_include, 'time': t_seconds}, 
                                   dims=["channel", "time"])

    # If no CAR filtering of signals was applied. 
    else:
        
        # Simply assigning the signals_car variable to signals.
        signals_car = copy.deepcopy(signals)
    
    return signals_car





def command_from_voting_buffer(classification, n_votes_thr, voting_buffer):
    """
    DESCRIPTION:
    Using the voting buffer, the number of 'grasp' classifications are counted. If this number exceeds the voting 
    threshold, a 'click' command may be sent to the user-interface. Otherwise, nothing is sent (here, command is 
    simulated as 'nothing'). Note that for the 'click' command to be issued the lockout period starting from the
    most recent 'click' command must have elapsed (see function sending_command).
    
    INPUT VARIABLES:
    classification: [string ('rest'/'grasp')]; The classification of the feature packet corresponding to when this
                    function was called.
    n_votes_thr:    [int]; Number of grasp votes which must accumulate within the most recent n_votes classifications
                    to issue a click command.
    voting_buffer:  [list > strings ('rest'/'grasp')]; The updating buffer of the most recent classifications at any
                    time.
                                
    GLOBAL PARAMETERS:
    model_classes: [list > strings]; List of all the classes to be used in the classifier.
    packet_size:   [float (units: ms)]; Temporal size of data-streamed packet to decoder.
    
    OUTPUT VARIABLES:
    command:       [string ('nothing'/'click')]; Depending on whether or not the voting threshold was met, a 'click'
                   or 'nothing' command is simulated to be issued to the user-interface.
    voting_buffer: [list > strings ('rest'/'grasp')]; The updated voting buffer which removed the earliest
                   classification and now contains the most recent classification.
    """
    
    # COMPUTATION:

    # Updating the voting buffer.
    voting_buffer_old = voting_buffer[1:]
    voting_buffer     = voting_buffer_old + [classification]
        
    # Computing the number of grasp votes.
    n_votes_grasp = voting_buffer.count('grasp')
    
    # If the number of grasp votes exceeds the voting threshold, send a click command. Otherwise, send nothing.
    if n_votes_grasp >= n_votes_thr:
        command = 'click'
    else:
        command = 'nothing'
    
    return command, voting_buffer





def commands_per_packet_simulation(n_votes, n_votes_thr, model_outputs, t_lockout):
    """
    DESCRIPTION:
    Using the classification that occurs with each packet of data, the real-time output to the user-interface 
    is simulated below. After each packet of data, the decoder must decide whether or not to issue a click
    command to the user-interface. This is determined by the total number of grasp (or any other non-rest class)
    classifications (votes) accumulating within the voting buffer, which exceeds a pre-specified voting threshold.
    In other words, if a voting buffer takes into account the most recent N classfications (n_votes) at any given
    time, there must be a minimum threshold (number) of grasp votes (n_voting_thr) accumulating within this window
    to issue a click command to the user interface. Additionally, a lockout period (t_lockout) is enforced such 
    that multiple commands from the same attempted movement are not sent to the user-interface. This lockout 
    period starts at the most recent 'click' command sent to user-interface and during this time, no further 
    'click' commands are allowed to be sent, even if the voting threshold was surpassed.

    INPUT VARIABLES:
    model_outputs: [dictionary (key: packet number; Values: below)]; Dictionary containing the model 
                   probabilities, model classification, and corresponding time for each data packet.
        class:     [list > strings]; Classification for each packet.
        prob:      [array (1 x classes) > floats]; Per-packet model probabilities for each class.
        time:      [float (units: s)]; Corresponding time for each set of model probabilities.
    n_votes:       [int]; Number of most recent classifications to consider when voting on whether a click should
                   or should not be issued in real-time decoding simulation. For N number of votes, the voting 
                   window corresponds to N*packet size (unitless * ms) worth of data. For example, 7 votes with a
                   packet size of 100 ms corresponds to 700 ms of worth of data being considered in the voting
                   window.
    n_votes_thr:   [int]; Number of grasp votes which must accumulate within the most recent n_votes classifications
                   to issue a click command.
    t_lockout:     [float (units: ms)]; The minimum amount of time which must elapse after the most recent 'click'
                   command was issued, such that another 'click' command may be issued.

    GLOBAL PARAMETERS:
    packet_size: [float (units: ms)]; Temporal size of data-streamed packet to decoder.

    NECESSARY FUNCTIONS:
    command_from_voting_buffer
    sending_command

    OUTPUT VARIABLES:
    commands:    [dictionary (key: packet number; Values: below)]; Dictionary containing the per-packet commands
                 and corresponding times.
        command: [string]; Command for each packet.
        time:    [float (units: s)]; Correpsonding time for each command.
    """

    # COMPUTATION:

    # Initializing a voting buffer which at any time will contain the most recent n_votes classifications.
    voting_buffer = [None]*n_votes

    # Computing the total number of packet classifications.
    n_packets = len(model_outputs.keys())

    # Initializing a dictionary of commands and corresponding command times for each packet.
    commands = collections.defaultdict(dict)
    for p in range(n_packets):
        commands[p]['command'] = None
        commands[p]['time']    = None

    # Initializing state on whether a 'click' command may be sent to the user-interface. This state is False if the 
    # lockout period from the previous 'click' command has not yet elapsed. Otherwise, the state is True. This state
    # is initialized to True before the first feature packet, no 'click' commands could have occurred to activate the
    # lockout period.
    send_command = True
    
    # Initializing the time that the most recent 'click' command was sent to the user-interface. This time is initialized
    # to 0 ms becuase the first feature packet, there does not exist a prior 'click' command and thus no prior most
    # recent time.
    t_most_recent_click = 0

    # Iterating across all packet classifications.
    for p in range(n_packets):
        
        # Extracting the model output for the current packet.
        this_model_output = model_outputs[p]

        # Extracting the current classification and corresponding time..
        this_classification = this_model_output['class']
        this_time           = this_model_output['time']
        
        # Determining the command based on the current voting window.
        command, voting_buffer = command_from_voting_buffer(this_classification, n_votes_thr, voting_buffer)
        
        # Determing whether a 'click' or 'nothing' command should be sent to the user-interface depending on whether
        # the lockout period has elapsed.
        command, send_command, t_most_recent_click = sending_command(command, p, send_command, t_lockout,\
                                                                     t_most_recent_click)

        # Updating the dictionary with the current command and corresponding time.
        commands[p]['command'] = command
        commands[p]['time']    = this_time
    
    return commands





def computing_calibration_stats(sxx_signals_calib):
    """
    DESCRIPTION:
    Computing the mean and standard deviation of the calibration signals for each channel and frequency bin across
    all time points.

    INPUT VARIABLES:
    sxx_signals_calib: [xarray (channels, frequency bins, time samples) > floats (units: V^2/Hz)]; Spectral power
                       of the continuous voltage signal for each channel. Time samples are in units of seconds.

    OUTPUT VARIABLES:
    calibration_sxx_mean   [xarray (channels, frequency bins) > floats (units: V^2/Hz)]; Calibration mean of each 
                           channel and frequency bin across time.
    calibration_sxx_stdev: [xarray (channels, frequency bins) > floats (units: V^2/Hz)]; Calibration standard deviation
                           of each channel and frequency bin across time.
    """
    # COMPUTATION:
        
    # Computing the mean and standard deviation of the calibration signals across time. Results in statistics for each
    # channel and frequency pair.
    calibration_sxx_mean  = np.mean(sxx_signals_calib, axis = 2)
    calibration_sxx_stdev = np.std(sxx_signals_calib, axis = 2)
        
    return calibration_sxx_mean, calibration_sxx_stdev





def concatenating_power_bands(sxx_power):
    """
    DESCRIPTION:
    Features across all powerbands are concatenated into one feature dimension. For example, if a feature array
    has dimensions (chs: 128, powerbands: 3, time samples: 4500), the powerband concatenation will result in an
    array of size: (chs x pwrbands: 384, time samples: 4500).
    
    INPUT VARIABLES:
    sxx_power: [xarray (channel, powerband, time samples] > floats (units: V^2/Hz)]; For each standardized 
               (to calibration) frequency band, the band power exists for each channel across every time point. 
               Time dimensions is in units of seconds.
                        
    OUTPUT VARIABLES:
    sxx_power_all_bands: [xarray (features (chs x bands),  time samples) > floats (units: V^2/Hz)]; Concatenated
                         band power features over all power bands. 
    """
    
    # COMPUTATION:

    # Extracting the dimension sizes of the power features array.
    n_channels = sxx_power.shape[0]
    n_bands    = sxx_power.shape[1]
    n_samples  = sxx_power.shape[2]

    # Concatenating all powerbands and flipping the dimensions
    sxx_power_all_bands = np.asarray(sxx_power).reshape(n_channels*n_bands, n_samples)

    # Converting the concatenated band features back into an xarray.
    sxx_power_all_bands = xr.DataArray(sxx_power_all_bands, 
                                       coords={'feature': np.arange(n_channels*n_bands), 'time': np.arange(n_samples)}, 
                                       dims=["feature", "time"])
    
    return sxx_power_all_bands





def concatenating_historical_features(sxx_power_all_bands):
    """
    DESCRIPTION:
    Based on the experimenter-determined time history (t_history) and the time resolution (global variable sxx_shift), the number of 
    historical time points are calculated (t_history/sxx_shift). An xarray with dimensions (history, features, time) is created, 
    where each coordinate in the history dimension represents how much the features were shifted in time. For example, consider one 
    coordinate in the feature array, and suppose a time length of 10 samples and a total time history of 3 samples. For this feature, 
    the resulting xarray would look like:
    
    historical time shifts
         n=2 shifts      [[0.000, 0.000, 0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101],
         n=1 shift        [0.000, 0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521],
         n=0 shifts       [0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521, 0.882]]
                            t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7    t=8    t=9     (time samples)   
    
    and the resulting dimensions of this array are (history=3, features = 1, time = 10).
    
    INPUT VARIABLES:
    sxx_power_all_bands: [xarray (channels, time samples) > floats (units: V^2/Hz))]; Concatenated band power features
                         across all power bands. Time dimension is in units of seconds.
    
                           
    GLOBAL PARAMETERS: 
    sxx_shift: [int (units: ms)]; Length of time by which sliding window (sxx_window) shifts along the time domain.
    t_history: [float (unit: ms)]; Amount of time history used as features.
    
    OUTPUT VARIABLES:
    sxx_power_all_history: [xarray (time history, features, sample) > floats (units: V^2/Hz))]; Array of historical time features.
                           The sample dimension is only coordinate long.
    """
    
    # COMPUTATION:
    
    # Computing the total number of historical time features.
    n_history = int(t_history/sxx_shift)
    
    # Extracting the number of features over all powerbands. 
    n_features = sxx_power_all_bands.feature.shape[0]
    
    # Extracting the time array and corresponding number of time samples.
    t_seconds = sxx_power_all_bands.time
    n_samples = t_seconds.shape[0]
    
    # Initializing a feature array which will contain all historical time features.
    sxx_power_all_history = np.zeros((n_history, n_features, n_samples))
    
    # Iterating across all historical time shifts. The index, n, is the number of samples back in time that
    # will be shifted.
    for n in range(n_history):

        # If currently extracting historical time features (time shift > 0)
        if n >= 1:

            # Extracting the historical time features for the current time shift.
            these_features_history = sxx_power_all_bands[:,:-n]

            # Creating a zero-padded array to make up for the time curtailed from the beginning of the features
            # array.
            zero_padding = np.zeros((n_features, n))

            # Concatenating the features at the current historical time point with a zero-padded array.
            these_features = np.concatenate((zero_padding, these_features_history), axis=1)

        # If extracting the first time set of features (time shift = 0). 
        else:
            these_features = sxx_power_all_bands
            
        # Assigning the current historical time features to the xarray with all historical time features.
        sxx_power_all_history[n,:,:] = these_features
    

    # Converting the historical power features for this task to xarray. Only extracting the very last time sample
    # of the historical spectral power. All other time samples have 0's due to the zero padding. In training, this
    # is accounted for by curtailing the beginning of this historical power array by the first n samples
    # corresponding to time history.
    sxx_power_all_history = xr.DataArray(np.expand_dims(sxx_power_all_history[:,:,-1], axis=2), 
                                         coords={'history': np.arange(n_history), 'feature': np.arange(n_features)}, 
                                         dims=["history", "feature", "sample"])
    
    return sxx_power_all_history





def creating_data_packets(data_exper_dict):
    """
    DESCRIPTION:
    Creating a list of continuous voltage data packets which will be used to simulate packets of data throughput
    during real-time decoding. 
    
    INPUT VARIABLES:
    data_exper_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        signals:     [xarray (channels x time samples) > floats (units: microvolts)]; Array of continuous voltage time
                     signals of the experimental task from all channels. Time samples are in units of seconds.
        stimuli:     [xarray (1 x time samples) > strings ('click'/'no_click')]; Array of states at each time sample
                     of the experimental task. Time samples are in units of seconds.
    
    GLOBAL PARAMETERS:
    packet_size:   [float (units: ms)]; Temporal size of data-streamed packet to decoder.
    sampling_rate: [int (samples/s)]; Sampling rate at which the data was recorded.
    
    NECESSARY FUNCTIONS:
    index_advancer
    
    OUTPUT VARIABLES:
    data_packets_dict: [dictionary (Keys/Value pairs below)];
        states:        [list > strings]; List of state strings corresponding to each data packet. 
        signals:       [list > array (channels, time samples) > floats (units: uV)]; List of data packets where each
                       packet contains continuous voltage data.
        t_seconds:     [list > floats (units: s)]; List of time points for each packet.
    """
    
    # COMPUTATION:
    
    # Initializing the dictionary of data packets.
    data_packets_dict = {}

    # Computing the sampling rate factor.
    factor = sampling_rate/1000 # sa/s / (ms/s) = sa/s * s/ms = sa/ms

    # Computing the total number of samples in a packet.
    packet_samples = int(packet_size * factor) # ms * sa/ms = sa
    
    # Extracting the states, signals, and time arrays.
    states  = data_exper_dict['states']
    signals = data_exper_dict['signals']
    time    = data_exper_dict['signals'].time

    # Computing the total number of samples in the raw data array.
    n_samples = time.shape[0]
    
    # Computing the total number of packets in the raw data. Want to use np.ceil() such as to account for the last
    # bit of samples which may not be a full packet's size.
    n_packets = math.ceil(n_samples/packet_samples) # sa / (sa/packet) = sa * packet/sa = n_packets
    
    # Initializing the lists of state, signals, and time packets.
    state_packets    = [None] * n_packets
    signal_packets   = [None] * n_packets
    t_second_packets = [None] * n_packets

    # Initializing the packet indices.
    packet_inds = np.zeros((2,))

    # Iterating across all packets.
    for this_packet in range(n_packets):

        # Updating the packet indices.
        packet_inds = index_advancer(packet_inds, packet_samples)
        
        # Extracting the packet starting and ending indices.
        packet_start_ind = packet_inds[0]
        packet_end_ind   = packet_inds[1] 
                
        # If the packet ending index of the final packet is greater than the total number of samples.
        if packet_end_ind >= n_samples:
            packet_end_ind = n_samples-1
                        
        # Extracting the state, signals, and time from the current packet.
        this_state_packet  = states[packet_end_ind] 
        this_signal_packet = signals[:,packet_start_ind:packet_end_ind]
        this_packet_time   = time[packet_end_ind] 
                
        # Updating the lists of packets.
        state_packets[this_packet]    = this_state_packet
        signal_packets[this_packet]   = this_signal_packet
        t_second_packets[this_packet] = np.round(this_packet_time.values, 3)

    # Uploading the packetd time domain data into the dictionary.
    data_packets_dict['states']    = state_packets
    data_packets_dict['signals']   = signal_packets
    data_packets_dict['t_seconds'] = t_second_packets
    
    return data_packets_dict





def data_upload(date, eeglabels, exper_name, file_extension, patient_id, state_marker):
    """
    DESCRIPTION:
    For the experimenter-input block of speller data, the signal, clicks, and sampling information
    are stored in a dictionary.
    
    INPUT VARIABLES:
    date:           [string]; Date on which the task was run. Format: YYYY_MM_DD.
    eeglabels:      [list > strings (eeg channel names)]: EEG channels extracted from the .hdf5 or
                    .mat file.
    exper_name:     [string]; Name of the experimental task that was run.
    file_extension: [string (hdf5/mat)]; The data file extension to be used.
    patient_id:     [string]; Patient PYyyNnn ID or CCXX ID.
    state_marker:   [string]; Name of the state to be extracted at each time sample.
    
    GLOBAL PARAMETERS:
    sampling_rate: [int (samples/s)]; Sampling rate at which the data was recorded.
  
    OUTPUT VARIABLES:
    data_dict:   [dict (Key: string (task ID); Values: dict (Key/Value pairs below))]; 
        signals: [xarray (channels x time samples) > floats (units: microvolts)]; Array of raw time
                 signals across all channels. Time samples are in units of seconds.
        states:  [xarray (1 x time samples) > ints]; Array of states at each time sample. Time samples
                 are in units of seconds.
    """
    
    # COMPUTATION:

    # Initializing the dictionary to contain all the data from the speller block.
    data_dict = {}

    # Creating the path for the current date/task pair.
    path = '/mnt/shared/ecog/' + patient_id + '/' + file_extension + '/' + date + '/' + exper_name + '.' + file_extension

    # Extracting the eeg signals, depending on whether they come from an .hdf5 or .mat file.
    if file_extension == 'hdf5':

        # Extracting the .hdf5 data.
        h5file      = H5EEGFile(path)
        eeg         = h5file.group.eeg(); 
        eeg_signals = eeg.dataset[:]

        # Extracting the .hdf5 states.
        aux    = h5file.group.aux();
        states = aux[:, state_marker]

    if file_extension == 'mat':

        # Extracting the .mat data.
        matlab_data = loadmat(path, simplify_cells=True)
        eeg_signals = matlab_data['signal']
        
        # Extracting the .mat states.
        states = matlab_data['states'][state_marker]
        
        
    # Creating the time array. Note that the first time sample is not 0, as the first recorded signal
    # sample does not correspond to a 0th time.
    time_seconds = (np.arange(eeg_signals.shape[0])+1)/sampling_rate

    
    # Converting the clicks into an xarray.
    states = xr.DataArray(states,
                          coords={'time': time_seconds},
                          dims=["time"])

    # Multiplying our eeg signals by a gain that's done downstream of the .dat collection in BCI2000.
    eeg_signals = eeg_signals * 0.25
    
    # Computing the total number of channels and time samples.
    n_samples  = eeg_signals.shape[0]
    n_channels = len(chs_include)

    # Initializing an xarray for the signals.
    signals = xr.DataArray(np.zeros((n_channels, n_samples)), 
                           coords={'channel': chs_include, 'time': time_seconds}, 
                           dims=["channel", "time"])
    
    # Populating the signals xarray with the eeg signals.
    for ch_name in chs_include:

        # Extracting the appropriate channel index from the original eeglabels list and populating the
        # signals xarray with the channel's activity.
        eeg_ind              = eeglabels.index(ch_name)            
        signals.loc[ch_name] = eeg_signals[:,eeg_ind]

    # Updating the data dictionary
    data_dict['signals'] = signals
    data_dict['states']  = states

    return data_dict





def downsample_to_video_fps(click_highlights_bci2k, fps, t_start):
    """
    DESCRIPTION:
    Downsampling the click highlights list to the video FPS resolution.
    
    INPUT VARIABLES:
    click_highlights_bci2k: [xarray > strings (click/no-click)]; For each time sample (at the resolution of the 
                            BCI2000 sampling rate), there exists a click or no-click entry. The time dimension
                            of the xarray is in units of seconds.
    fps:                    [int]; Frames per second of click highlights array simulation from video feed.
    t_start:                [float (units: s)]; True starting time of the block.
    
    GLOBAL PARAMETERS:
    sampling_rate: [int (samples/s)]; Sampling rate at which the data was recorded.
    
    OUTPUT VARIABLES:
    click_highlights_video: [xarray > strings]; At the video resolution, there exists a click or no-click entry.
                            Time dimension in units of seconds at video resolution.
    """
    
    # COMPUTATION:
    
    # Extracting the BCI2000 times from the click highlights xarray.
    bci2k_times = click_highlights_bci2k.time_seconds
    
    # Computing the seconds per frame.
    spf = 1/fps # sec/frame

    # Computing the total BCI2000 block time.
    last_bci2k_time  = bci2k_times[-1]
    t_diff           = last_bci2k_time - t_start
    
    # Computing the total number of frames.
    n_frames = np.ceil(t_diff * fps) # (s * (frames/s)) = number of frames.
    
    # Adding an additional frame to account for the frame lost in subtraction of the starting and ending
    # times. For example, the difference between 0.100 and 0.000 is 0.1 s. At 30 FPS, frames would occur
    # every 0.033 ms. However, owever this does not account for the sample at 0.000 s.
    n_frames += 1
    
    # Extracting the indices of the BCI2000 time array corresponding to the frame times.
    these_inds = np.round(np.arange(0, n_frames) * spf * sampling_rate).astype(int) # frames x sec/frame x samples/sec = samples
    
    # THIS ISN'T A GOOD SOLUTION, BUT I DON'T HAVE TIME TO MAKE IT BETTER.
    if these_inds[-1] > click_highlights_bci2k.shape[0]:
        these_inds = these_inds[:-1]
        n_frames   -= 1
        
    # Creating the downsampled click highlights xarray.
    click_highlights_video = xr.DataArray(click_highlights_bci2k[these_inds].values,
                                          coords={'time_seconds': t_start + np.round(np.arange(n_frames)*spf, 3)},
                                          dims=["time_seconds"])    
    
    return click_highlights_video






def import_electrode_information(date, exper_name, file_extension, patient_id):
    """
    DESCRIPTION:
    The eeglabels and auxlabels lists will be populated with eeg channel names and auxilliary channel
    names respectively. These lists are created differently based on whether the data is extracted from
    a .hdf5 file or a .mat file.

    INPUT VARIABLES:
    date:           [string]; Date on which the speller was run. Format: YYYY_MM_DD.
    exper_name:     [string]; Name of the experimental task that was run.
    file_extension: [string (hdf5/mat)]; The data file extension to be used.
    patient_id:     [string]; Patient PYyyNnn ID or CCXX ID.
    
    OUTPUT VARIABLES:
    auxlabels: [array > strings (aux channel names)]: Auxilliary channels extracted from the .hdf5 or 
               .mat file.
    eeglabels: [list > strings (eeg channel names)]: EEG channels extracted from the .hdf5 or .mat file.
    """
    
    # COMPUTATION:
    
    # Creating the path for the current date/task pair.
    path = '/mnt/shared/ecog/' + patient_id + '/' + file_extension + '/' + date + '/' + exper_name + '.' + file_extension
    
    # Defining the ecog object.
    ecog = ECoGConfig()

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





def index_advancer(indices, shift):
    """
    DESCRIPTION:
    This function is used to update the indices of a vector X which is being repeatedly filled with smaller
    vectors A, B,  C, D, of a particular size, shift. For example, if vector X is initialized as A = zeros(1,20)
    and vectors A-D are defined as follows:
    A = 1:5, B = 1:5, C = 12:16, D = 7:11;
    
    Then we may wish to fill the first 5 elements of X with A and so the inputs are: indices = [0,0],
    shift = length(A), and the output is indices = [1,5], We may wish to fill the next 5 elements with B, and so
    the inputs are ndices = [1,5], shift = length(B), and the output is indieces = [6,10].

    INPUT VARIABLES:
    indices: [array (1 x 2) > int]; The beginning and ending indices within the array that is being updated. The
             first time this function is called, indices = [0,0]. 
    shift:   [int]; Difference between the starting and ending indices. The length of the vector that is being
             placed inside the array for which this function is used.

    OUTPUT VARIABLES:
    indices: [array (1 x 2) > int]; The beginning and ending indices within the array that is being updated.
    """

    # COMPUTATION:
    indices[0] = indices[1];
    indices[1] = indices[0] + shift; 
    
    # Converting values inside indices array to integers.
    indices = indices.astype(int)
    
    return indices





def load_start_stop_times(block_id, date, dir_intermediates, patient_id):
    """
    DESCRIPTION:
    Loading the true starting and stopping times for the current block. 
    
    INPUT VARIABLES:
    block_id:          [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:              [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_intermediates: [string]; [string]; Intermediates directory where relevant information is stored.
    patient_id:        [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    
    OUTPUT VARIABLES:
    t_start: [float (units: s)]; True starting time of the block.
    t_end:   [float (units: s)]; True ending time of the block.
    """
    
    # COMPUTATION:
    
    # Creating the directory and filename for the .txt file containing the starting and ending times of the block.
    dir_start_stop      = dir_intermediates + patient_id + '/Speller/BlocksStartAndStops/' + date + '/'
    filename_start_stop = date + '_' + block_id +'_StartStop.txt'

    # Creating the pathway for the .txt file.
    path_start_stop = dir_start_stop + filename_start_stop

    # Opening up the block starting and ending times from the pathway.
    txt_file_start_stop = open(path_start_stop)

    # Reading the content of the text file with the start and end times.
    text_file_lines = txt_file_start_stop.readlines()

    # Reading in the lines with the starting and ending times.
    line_time_start = text_file_lines[5]
    line_time_end   = text_file_lines[6]

    # Extracting the start and stop times from the corresponding strings.
    t_start = float(line_time_start[7:])
    t_end   = float(line_time_end[7:])
    
    # PRINTING:
    print('Start time (s): ', t_start)
    print('End time (s): ', t_end)

    return t_start, t_end





def loading_buffer_params(foldername_buffer_params, path_params):
    """
    DESCRIPTION:
    Loading parameters related to data throughput for simulating real-time processing.
    
    PARAMETERS:
    buffer_size: [float (units: ms)]; Window length of raw signal used for computing spectrogram, which continuously
                 updates.
    packet_size: [float (units: ms)]; Temporal size of data-streamed packet to decoder.
    """
    
    # Making each of the following parameters global variables.
    global buffer_size
    global packet_size    
    
    # LOADING PARAMETERS:.
    with open(path_params + '/' + foldername_buffer_params + '/buffer_size', "rb") as fp:
        buffer_size = pickle.load(fp)
    with open(path_params + '/' + foldername_buffer_params + '/packet_size', "rb") as fp:
        packet_size = pickle.load(fp)

    # PRINTING:
    print('PACKET SIZE: ', packet_size, ' ms')
    print('BUFFER SIZE: ', buffer_size, ' ms')
    
    return None




def loading_car_params(foldername_car_params, path_params):
    """
    DESCRIPTION:
    Loading parameters related to common-average referencing (CAR) filtering.
    
    PARAMETERS:
    car:     [bool (True/False)] Whether or not CAR filtering will be performed.
    car_chs: [list > list > strings]; Sublists of channels which will get CAR filtered independently of channels 
             in other sublists.
    """
    
    # Making each ofthe following parameters global variables.
    global car
    global car_chs
    
    # LOADING PARAMETERS:
    with open(path_params + '/' + foldername_car_params + '/car', "rb") as fp:
        car = pickle.load(fp)
    with open(path_params + '/' + foldername_car_params + '/car_chs', "rb") as fp:
        car_chs = pickle.load(fp)
        
    # PRINTING:
    print('CAR: ', car)
    
    return None





def loading_channel_params(foldername_channel_params, path_params):
    """
    DESCRIPTION:
    Loading channel information.
    
    PARAMETERS:
    chs_all:     [list > strings]; All possible channels.
    chs_exclude: [list > strings]; The list of channels to be excluded in further analysis.
    chs_include: [list > strings]; The list of channels to be included in further analysis.
    """
    
    # Making each of the following parameters global variables.
    global chs_all
    global chs_exclude    
    global chs_include
    
    # LOADING PARAMETERS
    with open(path_params + '/' + foldername_channel_params + '/channels_all', "rb") as fp:
        chs_all = pickle.load(fp)
    with open(path_params + '/' + foldername_channel_params + '/channels_exclude', "rb") as fp:
        chs_exclude = pickle.load(fp)
    with open(path_params + '/' + foldername_channel_params + '/channels_include', "rb") as fp:
        chs_include = pickle.load(fp)
        
    # PRINTING:
    print('CHANNELS INCLUDED: ', chs_include)
        
    return None





def loading_history_params(foldername_history_params, path_params):
    """
    DESCRIPTION:
    Loading the feature time history.
    
    PARAMETERS:
    t_history: [float (unit: ms)]; Amount of feature time history.
    """
    
    # Making each of the following parameters global variables.
    global t_history
    
    # LOADING PARAMETERS.
    with open(path_params + '/' + foldername_history_params + '/t_history', "rb") as fp:
        t_history = pickle.load(fp)
        
    # PRINTING:
    print('TIME HISTORY: ', t_history, ' ms')

    return None





def loading_model(foldername_model_params, path_params):
    """
    DESCRIPTION:
    Loading the classification model.
    
    OUTPUT VARIABLES:
    model: [classification model]; The classification model.
    """
    # Making each of the following parameters global variables.
    global model
    
    # LOADING MODEL:
    with open(path_params + '/' + foldername_model_params + '/model_type', "rb") as fp:
        model_type = pickle.load(fp)

    # Creating the model directory from where to import the model.
    model_dir = path_params + '/' + foldername_model_params
    
    # If the loaded model type is LSTM.
    if model_type == 'LSTM':
        model_path = model_dir + '/Model'
        print('MODEL PATH: ', model_path)
        model = tf.keras.models.load_model(model_path)
        
    else:
        model = 'MODEL NOT FOUND'
        
        
    # PRINTING: 
    print('MODEL: ', model)
    
    return None





def loading_model_params(foldername_model_params, path_params):
    """
    DESCRIPTION:
    Loading the supplemenal information for the model.
    
    PARAMETERS:
    model_classes: [list > strings]; List of all the classes to be used in the classifier.
    model_type:    [string ('SVM','LSTM')]; The model type that will be used for classification.
    """
    
    # Making each of the following parameters global variables.
    global model_classes
    global model_type
    
    # LOADING PARAMETERS
    with open(path_params + '/' + foldername_model_params + '/model_classes', "rb") as fp:
        model_classes = pickle.load(fp)
    with open(path_params + '/' + foldername_model_params + '/model_type', "rb") as fp:
        model_type = pickle.load(fp)
        
    # PRINTING:
    print('MODEL CLASSES: ', model_classes)
    print('MODEL TYPE: ', model_type)

    return None





def loading_pc_params(foldername_pc_params, path_params):
    """
    DESCRIPTION:
    Loading parameters related to principal component dimensionality reduction.
    
    PARAMETERS:
    eigenvectors:       [array (features x pc features) > floats]; Array in which columns consist eigenvectors 
                        which explain the variance in features in descending fashion. 
    training_data_mean: [xarray (history x features) > floats (units: V^2/Hz)]; Mean power of each feature of 
                        only the 0th time shift. This array is repeated for each historical time point.
    """
    
    # Making each of the following parameters global variables.
    global eigenvectors
    global training_data_mean    
    
    # LOADING PARAMETERS:
    with open(path_params + '/' + foldername_pc_params + '/eigenvectors', "rb") as fp:
        eigenvectors = pickle.load(fp)
    with open(path_params + '/' + foldername_pc_params + '/training_data_mean', "rb") as fp:
        training_data_mean = pickle.load(fp)
        
    return None





def loading_sxx_params(foldername_sxx_params, path_params):
    """
    DESCRIPTION:
    Loading parameters related to spectral processing.
    
    PARAMETERS:
    f_max_bounds:  [list > int (units: Hz)]; For each frequency band, maximum power band frequency.
    f_min_bounds:  [list > int (units: Hz)]; For each frequency band, minimum power band frequency.
    sampling_rate: [int (samples/s)]; Sampling rate at which the data was recorded.
    sxx_shift:     [int (units: ms)]; Length of time by which sliding window (sxx_window) shifts along the
                   time domain.
    sxx_window:    [int (units: ms)]; Time length of the window that computes the frequency power.
    """
    
    # Making each of the following parameters global variables.
    global f_max_bounds
    global f_min_bounds   
    global sampling_rate
    global sxx_shift
    global sxx_window   
    
    # LOADING PARAMETERS:
    with open(path_params + '/' + foldername_sxx_params + '/f_max_bounds', "rb") as fp:
        f_max_bounds = pickle.load(fp)
    with open(path_params + '/' + foldername_sxx_params + '/f_min_bounds', "rb") as fp:
        f_min_bounds = pickle.load(fp)
    with open(path_params + '/' + foldername_sxx_params + '/sampling_rate', "rb") as fp:
        sampling_rate = pickle.load(fp)
    with open(path_params + '/' + foldername_sxx_params + '/sxx_shift', "rb") as fp:
        sxx_shift = pickle.load(fp)
    with open(path_params + '/' + foldername_sxx_params + '/sxx_window', "rb") as fp:
        sxx_window = pickle.load(fp)
        
    # PRINTING
    print('F-BAND MAX VALS: ', f_max_bounds, 'Hz')
    print('F-BAND MIN VALS: ', f_min_bounds, 'Hz')
    print('SAMPLING RATE:   ', sampling_rate, 'Sa/s')
    print('SPECTRAL SHIFT:  ', sxx_shift, 'ms')
    print('SPECTRAL WINDOW: ', sxx_window, 'ms')

    return None





def mean_centering(data, data_mean):
    """
    DESCRIPTION:
    Mean-centering all features at all historical time points by subtracting the data mean, averaged across time.
        
    INPUT VARIABLES:
    data:      [xarray (history x features x time samples) > floats (units: V^2/Hz)]; Historical power features across time
               samples. Time samples are in units of seconds.
    data_mean: [xarray (history x features) > floats (units: V^2/Hz)]; Mean power of each feature of only the 0th time shift.
               This array is repeated for each historical time point.
    
    OUTPUT VARIABLES:
    data_centered: [xarray (history x features x time samples) > floats (units: V^2/Hz)]; Mean-centered historical power features
                   across time samples. Time samples are in units of seconds.
    """
    
    # COMPUTATION:
    
    # Extracting the number of historical time points, features, and time samples from the data xarray.
    n_history  = data.history.shape[0]
    n_features = data.feature.shape[0]
    n_samples  = data.sample.shape[0]
    
    # Converting the data mean xarray to a regular array and repeating the means for each sample.
    data_mean = np.array(data_mean)
    # print('DATA MEAN SHAPE: ', data_mean.shape)
    
    data_mean = np.expand_dims(data_mean, axis=2)
    data_mean = np.tile(data_mean, (1, 1, n_samples))
    
    # Converting the data mean array back into an xarray
    data_mean = xr.DataArray(data_mean, 
                             coords={'history': np.arange(n_history), 'feature': np.arange(n_features), 'sample': np.arange(n_samples)}, 
                             dims=["history", "feature", "sample"])
    
    # Computing the mean-centered data.
    data_centered = data - data_mean
    
    return data_centered





def model_output_per_packet(feature_packets):
    """
    DESCRIPTION:
    Computing the model probabilities and classifications for each feature packet.
    
    INPUT VARIABLES:
    feature_packets: [dictionary (key: packet number; Values: below)]; Dictionary containing packet data and corresponding
                     time for each packet.
        data:        [xarray (pc features, time samples) > floats (units: V^2/Hz)]; All historical time features for each 
                     packet.
        time:        [float (units: s)]; Corresponding time for feature packet.
        
    GLOBAL PARAMETERS:
    model_classes: [list > strings]; List of all the classes to be used in the classifier.
                     
    NECESSARY FUNCTIONS:
    rearranging_features
    
    OUTPUT VARIABLES:
    model_outputs: [dictionary (key: packet number; Values: below)]; Dictionary containing the model probabilities, model
                   classification, and corresponding time for each data packet.
        class:     [list > strings]; Classification for each packet.
        prob:      [array (1 x classes) > floats]; Per-packet model probabilities for each class.
        time:      [float (units: s)]; Corresponding time for each set of model probabilities.
    """
    
    # COMPUTATION:
    
    # Computing the number of total feature packets.
    n_packets = len(feature_packets)

    # Initializing the dictionary of model outputs.
    model_outputs = collections.defaultdict(dict)
    for p in range(n_packets):
        model_outputs[p]['class'] = None
        model_outputs[p]['prob']  = None
        model_outputs[p]['time']  = None
    
    # Iterating across each packet.
    for p in trange(n_packets, position=0, desc='Iterating across all signal packets:'):
        
        # Extracting the features from the current packet of data.
        this_packet_features = feature_packets[p]['data']
    
        # Rearranging the packet features according to the model used.
        this_packet_features = rearranging_features(this_packet_features)

        # Converting the packet features to a regular array.
        this_packet_features = np.asarray(this_packet_features)
        
        # Computing the predicted probabilities for the current feature packet.
        this_packet_probs = np.squeeze(model.predict(this_packet_features))
        
        # Classifying the current feature packet according to the predicted probabilities.
        classification_ind         = int(np.argmax(this_packet_probs))
        this_packet_classification = model_classes[classification_ind]
    
        # Updating the model outputs for the current feature packet.
        model_outputs[p]['class'] = this_packet_classification
        model_outputs[p]['prob']  = this_packet_probs
        model_outputs[p]['time']  = feature_packets[p]['time']
    
    return model_outputs





def pc_transform(data, eig_vectors):
    """
    DESCRIPTION:
    Transforming the data into PC space by multiplying the features at each historical time point by the same eigenvectors.
    
    INPUT VARIABLES:
    data:        [xarray (features x time samples) > floats (units: V^2/Hz)];
    eig_vectors: [array (features x pc features) > floats]; Array in which columns consist eigenvectors which explain the variance in 
                 features in descending fashion. Time samples are in units of seconds.
    
    OUTPUT VARIABLES:
    data_pc: [xarray (pc features x time samples) > floats (units: V^2/Hz)]; Reduced-dimensionality data. Time samples are in units of
             seconds.
    """
    
    # COMPUTATION:
    
    # Extracting the number of historical time shifts and number of samples from the data.
    n_history = data.history.shape[0]
    n_samples = data.sample.shape[0]
    
    # Computing the number of PC features.
    n_features_pc = eig_vectors.shape[1]
    
    # Initializing an xarray of PC data.
    data_pc = xr.DataArray((np.zeros((n_history, n_features_pc, n_samples))),
                            coords={'history': np.arange(n_history), 'feature': np.arange(n_features_pc), 'sample': np.arange(n_samples)},
                            dims=["history", "feature", "sample"])
    
    # Iterating across all historical time shifts and multiplying the data at each historical time shift by the same eigenvectors.
    for n in range(n_history):
        
        # Extracting the data at the nth time shift.
        this_history_data = np.asarray(data.loc[n,:,:])
        
        # Transforming hte data of the nth time shift to PC space.
        data_pc.loc[n,:,:] = np.matmul(this_history_data.transpose(), eig_vectors).transpose()   
    
    return data_pc





def power_generator(sxx_signals):
    """
    DESCRIPTION:
    Given the experimenter-specified minimum and maximum frequencies, the spectral band power is computed.
    
    INPUT VARIABLES:
    sxx_signals: [xarray (channels, frequency bins, time samples) > floats (units: V^2/Hz)]; Spectrogram power
                 of the continuous voltage time signals. Time dimension is in units of seconds.
                       
    GLOBAL PARAMETERS:
    chs_include:  [list > strings]; The list of channels to be included in further analysis.
    f_max_bounds: [list > int (units: Hz)]; For each frequency band, maximum power band frequency.
    f_min_bounds: [list > int (units: Hz)]; For each frequency band, minimum power band frequency.
        
    OUTPUT VARIABLES:
    sxx_power: [xarray (channel, powerband, time samples] > floats (units: V^2/Hz)]; For each standardized 
               (to calibration) frequency band, the band power exists for each channel across every time point. 
               Time dimensions is in units of seconds.
    """
    
    # COMPUTATION:

    # Creating the names of the powerbands.
    powerbands = ['powerband'+str(n) for n in range(len(f_max_bounds))]

    # Computing the number of channels, powerbands and time samples.
    n_bands = len(powerbands)
    n_chs   = sxx_signals.shape[0]
    n_times = sxx_signals.shape[2]

    # Initializing the power arrays.
    sxx_power = xr.DataArray(np.zeros((n_chs, n_bands, n_times)), 
                             coords={'channel': sxx_signals.channel, 'powerband': powerbands}, 
                             dims=["channel", "powerband", "time"])

    # Extracting the frequency array from the spectral signals dictionary.
    f_sxx = sxx_signals.frequency.values

    # Iterating over every pair of frequency band bounds to compute the power within them.
    for f_band_ind, (this_freqband_min, this_freqband_max) in enumerate(zip(f_min_bounds, f_max_bounds)):

        # Extracting the frequency bins for the current set of frequency bounds.
        f_range = np.logical_and(f_sxx > this_freqband_min, f_sxx < this_freqband_max) 

        # Computing the power from the spectrogram.
        this_freqband_power = np.sum(sxx_signals[:,f_range,:], axis=1)

        # Naming the power band key for the current feature band.
        powerband_id = 'powerband' + str(f_band_ind)

        # Assigning the power to the appropriate dictionaries.
        sxx_power.loc[:,powerband_id,:] = this_freqband_power.values
        
    return sxx_power





def rearranging_features(data):
    """
    DESCRIPTION:
    Rearranging the data dimensions as necessary to fit the experimenter-determined model.
    
    INPUT VARIABLES:
    data: [xarray (time history x features x time samples) > floats (units: V^2/Hz)]; Array of historical time features.
    
    GLOBAL PARAMETERS:
    model_type: [string ('SVM','LSTM')]; The model type that will be used to fit the data.
    
    OUTPUT VARIABLES:
    data_rearranged: [xarray (dimensions vary based on model type) > floats (units: V^2/Hz)]; Rearranged data. 
    """
    # COMPUTATION:
    
    # Extracting the dimension sizes of the current features array.
    n_history  = data.history.shape[0]
    n_features = data.feature.shape[0]
    n_samples  = data.sample.shape[0]
    
    # If the model type is a SVM.
    if model_type == 'SVM':

        # Concatenating all the historical time features into one dimension.
        data_rearranged = np.asarray(data).reshape(n_history*n_features, n_samples)
        data_rearranged = data_rearranged.transpose()

        # Converting the rearranged features back into an xarray.
        data_rearranged = xr.DataArray(data_rearranged, 
                                       coords={'sample': np.arange(n_samples), 'feature': np.arange(n_history*n_features)}, 
                                       dims=["sample", "feature"])

    # If the model type is an LSTM.
    if model_type == 'LSTM':

        # Don't rearrange the features. They already have the correct dimensionality for LSTM.
        data_rearranged = data.transpose("sample","history","feature")
    
    return data_rearranged
 

    
    
    
def saving_click_highlights(block_id, click_highlights, click_highlights_name, date, n_votes, n_votes_thr):
    """
    DESCRIPTION:
    Saving the click highlights xarray.
    
    INPUT VARIABLES:
    block_id:              [string]; Block number of the task.
    click_highlights:      [xarray > strings ('click'/'nothing')]; The array of 'click' commands simulating for how long
                           a button on the user-interface is highlighted.
    click_highlights_name: [string]; Name of the xarray file which will hold the click highlights array.
    date:                  [string]; Date on which the task was run. Format: YYYY_MM_DD.    
    n_votes:               [int]; Number of most recent classifications to consider when voting on whether a 
                           click should or should not be issued in real-time decoding simulation. For N number of
                           votes, the voting window corresponds to N*packet size (unitless * ms) worth of data.
                           For example, 7 votes with a packet size of 100 ms corresponds to 700 ms of worth of 
                           data being considered in the voting window.
    n_votes_thr:           [int]; Number of grasp votes which must accumulate within the most recent n_votes
                           classifications to issue a click command.
    """
    
    # COMPUTATION:
    
    # Creating the pathway for saving the click highlights array.
    dir_saving  = '/mnt/shared/danprocessing/Projects/PseudoOnlineTests_for_RTCoG/Intermediates/CC01/Speller/ClickDetections/Simulated/' + date + '/'\
                  + block_id + '/' + str(n_votes) +'_vote_window_' + str(n_votes_thr) + '_vote_thr' + '/'
    filename    = date + '_' + block_id + '_' + click_highlights_name + '.nc'
    path_saving = dir_saving + filename
    
    print(path_saving)
    
    # Checking if the directory to save the click highlights exists.
    dir_saving_exists = os.path.exists(dir_saving)
    if not dir_saving_exists: 
        os.makedirs(dir_saving)
        
        # Saving
        click_highlights.to_netcdf(path_saving)

    # If the directory does exist, check to see if the file exists.
    else:
        path_saving_exists = os.path.exists(path_saving)
        
        # If the pathway does exist, delete it before re-saving the file.
        if path_saving_exists:
            os.remove(path_saving)
        
        # Saving
        click_highlights.to_netcdf(path_saving)
        
        
    # Checking if the pathway 
    
        
    # Saving
    #click_highlights.to_netcdf(path_saving)

    return None
    
    
    
    
    
def sending_command(command, packet_counter, send_command, t_lockout, t_most_recent_click):
    """
    DESCRIPTION:
    Determining whether a 'click' or 'nothing' command should be sent to the user-interface. Specifically,
    if the lockout period starting from the time when the most recent 'click' command was sent to the user-
    interface has not yet elapsed, then regardless of whether there were enough votes in the voting buffer
    to issue a 'click' command, a 'click' command will not be sent to the user-interface, and the command
    will be changed to 'nothing'.
    
    INPUT VARIABLES:
    command:             [string ('nothing'/'click')]; Depending on whether or not the voting threshold was
                         met, a 'click' or 'nothing' command is simulated to be sent to the user-interface.
    packet_counter:      [int]; Number of feature packets that have elapsed.
    send_command:        [boolean (True/False)]; Whether or not a 'click' command may be sent to the user-
                         interface. This state is False if the lockout period from the most recent 'click'
                         command has not yet elapsed. Otherwise, the state is True.
    t_lockout:           [float (units: ms)]; The minimum amount of time which must elapse after the most 
                         recent 'click' command was sent to the user-interface, such that another 'click'
                         command may be issued.
    t_most_recent_click: [float (units: ms)]; Time at which the most recent 'click' command was issued. This
                         is used as a reference to determine whether or not the lockout time has elapsed.
      
    GLOBAL PARAMETERS:
    packet_size: [float (units: ms)]; Temporal size of data-streamed packet to decoder.
        
    OUTPUT VARIABLES:
    command:             [string ('nothing'/'click')]; Updated command depending on whether or not the lockout
                         period starting from the most recent 'click' command has elapsed. If the command is 
                         'click' and this lockout period has not elapsed, change the command to 'nothing'. 
                         Otherwise, leave unchanged.
    send_command:        [boolean (True/False)]; Whether or not a 'click' command may be sent to the user-
                         interface. This state is False if the lockout period from the most recent 'click'
                         command has not yet elapsed. Otherwise, the state is True.
    t_most_recent_click: [float (units: ms)]; If a 'click' command is sent to the user-interface, this is the
                         starting time of that lockout period.
    """
    
    # COMPUTATION:
    
    # The lockout period from the most recent 'click' command has not yet elapsed, and it is impossible to
    # send a 'click' command to the user-interface regardless of whether the command variable is 'click'. As
    # such, the command will be overwritten as 'nothing' (this is simply for visualization purposes later on
    # in).
    if not send_command:
        
        # Setting the command to 'nothing'. 
        command = 'nothing'
        
        # Computing the time since the most recent 'click' command was sent to the user-interface. 
        t_current_packet   = packet_counter*packet_size             # N x ms = ms
        t_since_prev_click = t_current_packet - t_most_recent_click # ms - ms = ms
        
        # If the time since the most recent click has elapsed the lockout time, switch the send_command gate to
        # True, so that a future 'click' command may be sent to the user-interface.
        if t_since_prev_click > t_lockout: # ms - ms
            send_command = True
    
    # The lockout time starting from the most recent 'click' command has elapsed, and it is now possible to
    # issue another 'click' command to the user-interface.
    if send_command:
        
        # If the command is 'click', do not change the command name to 'nothing'. Reset the starting time of
        # the lockout period and switch the send_command gate to False. The time of the most recent 'click' 
        # command sent to the user-interface is updated.
        if command == 'click':
            t_most_recent_click = packet_counter*packet_size # N x ms = ms
            send_command        = False

    return command, send_command, t_most_recent_click





def shifting_commands_by_simulated_network_delay(commands_bci2k, networking_delay):
    """
    DESCRIPTION:
    In offline analysis of real-time use data, a networking delay between BCI2000 and the user-interface was observed,
    which caused the on-screen click to occur roughly 200 ms after it was detected by the decoder. This function 
    simulates that networking delay by shifting all commands at BCI2000 sampling rate resolution further in time by 
    200 ms.
    
    INPUT VARIABLES:
    commands_bci2k:   [list > strings ('nothing'/'click')]; Upsampled commands list at the BCI2000 sampling rate.
    networking_delay: [int (units: ms)]; Networking delay between the click being detected and being received by the
                      user-interface.
    
    GLOBAL PARAMETERS:
    sampling_rate: [int (samples/s)]; Sampling rate at which the data was recorded.
    
    OUTPUT VARIABLES:
    commands_bci2k: [list > strings ('nothing'/'click')]; Commands shifted by the number of samples corresponding to
                    the networking delay.    
    """
    
    # COMPUTATION:

    # Computing the processing delay in units of samples.
    n_samples_delay = int(networking_delay/1000*sampling_rate) # ms/(ms/s) * sample/s = ms * s/ms * samples/s = samples

    # Shifting the commands (at BCI2000 sampling rate resolution) by the number of samples corresponding to the 
    # simulated networking delay.
    if n_samples_delay == 0:
        pass
    if n_samples_delay > 0:
        commands_bci2k = commands_bci2k[:-n_samples_delay]
        commands_bci2k = ['nothing'] * n_samples_delay + commands_bci2k
    
    return commands_bci2k





def simulating_click_duration(click_duration, commands_bci2k, t_start):
    """
    DESCRIPTION:
    A list of 'click' or 'nothing' commands will be created at each time sample, where after each 'click' is 
    detected, it lasts for the number of samples corresponding to the click duration. This simulates how long the
    click appears on the user-interface.
  
    INPUT VARIABLES:
    click_duration: [int (units: ms)]; Duration of how long each click lasts.
    commands_bci2k: [list > strings ('nothing'/'click')]; Commands shifted by the number of samples corresponding to
                    the networking delay.    
    t_start:        [float (units: s)]; True starting time of the block.
    
    GLOBAL PARAMETERS:
    sampling_rate: [int (samples/s)]; Sampling rate at which the data was recorded.
    
    OUTPUT VARIABLES:
    click_highlights_bci2k: [xarray > strings (click/no-click)]; For each time sample (at the resolution of the 
                            BCI2000 sampling rate), there exists a click or no-click entry. The time dimension
                            of the xarray is in units of seconds.
    """
    
    # COMPUTATION:

    # Computing the number of BCI2000 samples and initializing a list of click highlights at the BCI2000 sampling
    # rate resolution.
    n_bci2k_samples        = len(commands_bci2k)
    click_highlights_bci2k = ['nothing']*n_bci2k_samples
    
    # Computing a time array corresponding to the 'click'/'nothing' values in the click_highlights list.
    time_seconds = np.round(np.arange(0,n_bci2k_samples)/sampling_rate, 3) # samples * sec/samples
    
    # Adjusting the time array such that the first sample is relative to the block's start time.
    time_seconds += t_start

    # Computing the number of samples that will be highlighted given the click duration.
    n_samples_highlight = int(sampling_rate * click_duration/1000) # samples/s * ms * s/ms = samples/s * s = samples

    # Iterating across all commands at the BCI2000 sampling rate resolution.
    for n, this_command in enumerate(commands_bci2k):

        # If the current command is a click.
        if this_command == 'click':

            # Computing the starting and ending for the highlight samples.
            start_ind = n
            end_ind   = start_ind + n_samples_highlight

            # Updating the click highlights corresponding to the current command.
            click_highlights_bci2k[start_ind:end_ind] = ['click']*n_samples_highlight
            
            # Curtailing the click highlights list in the case that the start index happened too late such that 
            # there was not a click highlight's worth of samples in the remaining length of indices. In that case,
            # the list added more samples than originally designtated.
            click_highlights_bci2k = click_highlights_bci2k[:n_bci2k_samples]
                        
    # Converting the click highlights list to an xarray.
    click_highlights_bci2k = xr.DataArray(click_highlights_bci2k,
                                          coords={'time_seconds': time_seconds},
                                          dims=["time_seconds"])

    return click_highlights_bci2k




def spectrogram_generator(signals):
    """
    DESCRIPTION:
    Computing the spectrogram for each channel across all time. 
    
    INPUT VARIABLES:
    signals: [array (channels, time samples > floats (units: microvolts)]; Array of continuous voltage time
             signals across all channels. Time samples are in units of seconds.
             
    GLOBAL PARAMETERS:
    chs_include:   [list > strings]; The list of channels to be included in further analysis.
    sampling_rate: [int (samples/s)]; Sampling rate at which the data was recorded.
    sxx_shift:     [int (units: ms)]; Length of time by which sliding window (sxx_window) shifts along the
                   time domain.
    sxx_window:    [int (units: ms)]; Time length of the window that computes the frequency power.
    
    OUTPUT VARIABLES:
    sxx_signals: [xarray (channels, frequency bins, time samples) > floats (units: V^2/Hz)]; Spectral power
                 of the continuous voltage signals for each channel. Time samples are in units of seconds.
    """
    
    # COMPUTATION:

    # Computing the constant which is used to correct for signal sampling rates that differ from 1000 samples/s.
    factor = int(sampling_rate/1000) # sa/s / (ms/s) = sa/s * s/ms = sa/ms

    # Resizing the spectral window and shift to account for the sampling rate.
    sxx_window_samples = sxx_window * factor # ms x sa/ms = samples
    sxx_shift_samples  = sxx_shift * factor  # ms x sa/ms = samples

    # Computing the spectrogram input parameters.
    f_s        = 1000*factor                            # ms/s x sa/ms = samples/s
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
    # the first spectral sample being associated with 128 ms in the t_sxx array (if sxx_window = 256), it will rather be
    # 256 ms.
    t_sxx = t_sxx + (sxx_window/2)/1000
    
    # Converting the spectral array into an xarray.
    sxx_signals = xr.DataArray(sxx, 
                               coords={'channel': np.asarray(chs_include), 'frequency': f_sxx}, 
                               dims=["channel", "frequency", "time"])
        
    return sxx_signals





def standardize_to_calibration(calibration_sxx_mean, calibration_sxx_stdev, sxx_signals):
    """
    DESCRIPTION:
    For each channel at each frequency, standardizing the spectral power to the statistics of the 
    calibration period. 
    
    INPUT VARIABLES:
    calibration_sxx_mean   [xarray (channels, frequency bins) > floats (units: V^2/Hz)]; Calibration mean of each
                           channel and frequency bin across time.
    calibration_sxx_stdev: [xarray (channels, frequency bins) > floats (units: V^2/Hz)]; Calibration standard deviation
                           of each channel and frequency bin across time.
    sxx_signals:           [xarray (channels, frequency bins, time samples) > floats (units: V^2/Hz)]; Spectral power
                           of the raw signals. Time dimension is in units of seconds.
                     
    OUTPUT VARIABLES:
    sxx_signals_z: [xarray (channels, frequency bins, time samples) > floats (units: V^2/Hz)]; Spectral power of the 
                   continuous voltage time signals that is standardized to the statistics from the calibration period. 
                   Time dimension is in units of seconds.
    """

    # COMPUTATION:

    # Extracting the channels, frequencies, and time coordinates.
    channels = sxx_signals.channel
    freqs    = sxx_signals.frequency
    times    = sxx_signals.time

    # Counting the number of coordinates in each dimension.
    n_chs   = channels.shape[0]
    n_freqs = freqs.shape[0]
    n_times = times.shape[0]
    
    # Expanding the calibration mean and standard deviation dimensions.
    calibration_sxx_mean  = np.expand_dims(calibration_sxx_mean, 2)
    calibration_sxx_stdev = np.expand_dims(calibration_sxx_stdev, 2)
    
    # Tiling the same number of samples for the calibration mean and standard deviation arrays.
    calibration_sxx_mean  = np.tile(calibration_sxx_mean, (1, n_times))
    calibration_sxx_stdev = np.tile(calibration_sxx_stdev, (1, n_times))

    # Computing the standardized spectrograms across all channels.
    sxx_signals_z = np.divide(np.subtract(sxx_signals, calibration_sxx_mean), calibration_sxx_stdev)

    return sxx_signals_z





def string_state_maker(data_dict, state_map):
    """
    DESCRIPTION:
    Converting the state array elements from integer to strings values using mapping dictionaries. 
    
    INPUT data_dict:
    data_dict:   [dict (Key: string (task ID); Values: dict (Key/Value pairs below))]; 
        signals: [xarray (channels x time samples) > floats (units: microvolts)]; Array of raw time signals
                 across all channels. Time samples are in units of seconds.
        states:  [xarray (1 x time samples) > ints]; Array of states at each time sample. Time samples are in 
                 units of seconds.
    state_map:   [dict (key: int; Value: strings)]; Mapping from the numerical values in the states array to
                 corresponding string names.
    
    OUTPUT VARIABLES:
    data_dict:   [dict (Key: string (task ID); Values: dict (Key/Value pairs below))]; 
        signals: Same as input.
        states:  [xarray (1 x time samples) > strings ('state_on'/'state_off')]; Array of states at each time
                 sample. Time samples are in units of seconds.
    """
    
    # COMPUTATION:
    
    # Extracting the states array.
    state_ints = data_dict['states']

    # Extracting the states array times and number of samples.
    t_seconds = state_ints.time
    n_samples = t_seconds.shape[0]

    # Initializing the array of states strings.
    state_strings = xr.DataArray(np.asarray([None]*n_samples),  
                                 coords={'time': t_seconds}, 
                                 dims=["time"])

    # Iterating across all pairs of state values and corresponding string names.
    for state_val, state_string in state_map.items():

        # Extracting the state string of the current state value.
        state_string = state_map[state_val]

        # Extracting the states indices of the current state value.
        these_inds_state_val = np.where(state_ints == state_val)

        # Updating the array of state strings.
        state_strings[these_inds_state_val] = state_string

    # Updating the states array with the string version.
    data_dict['states'] = state_strings
    
    return data_dict





def uploading_parameters(dir_base, model_config):
    """
    DESCRIPTION:
    Uploading all calibration and model configuration parameters. 
    
    INPUT VARIABLES:
    dir_model:    [string]; Directory where the model configuration and calibration information are stored.
    model_config: [string]; Name of the model configuration.
    """
    
    # COMPUTATION:
    
    # Creating the directories for the model configuration.
    dir_model_config = dir_base + '/' + model_config    
    
    # Extracting all the folder names in that directory. These folders all contain parameters that will be 
    # used in conjunction with the classification model.
    foldername_buffer_params,\
    foldername_car_params,\
    foldername_channel_params,\
    foldername_history_params,\
    foldername_model_params,\
    foldername_pc_params,\
    foldername_sxx_params = [name for name in sorted(os.listdir(dir_model_config))]
                
    # Loading each set of parameters.
    loading_buffer_params(foldername_buffer_params, dir_model_config)
    loading_car_params(foldername_car_params, dir_model_config)
    loading_channel_params(foldername_channel_params, dir_model_config)    
    loading_history_params(foldername_history_params, dir_model_config)
    loading_model(foldername_model_params, dir_model_config)
    loading_model_params(foldername_model_params, dir_model_config)
    loading_pc_params(foldername_pc_params, dir_model_config)
    loading_sxx_params(foldername_sxx_params, dir_model_config)
    





def upsampling_command_info_to_bci2000_resolution(commands, t_start):
    """
    DESCRIPTION:
    Upsampling the commands to match the sampling rate of BCI2000. This is because during a real-time session, 
    command states are saved at the BCI2000 sampling rate resolution, and the analysis pipeline that will be used
    (speller_analysis_simulated_click_detections.ipynb) to compute the performance metrics of this simulation 
    assumes the corresponding time resolution. 
  
    INPUT VARIABLES:
    commands:    [dictionary (key: packet number; Values: below)]; Dictionary containing the per-packet commands
                 and corresponding times.
        command: [string]; Command for each packet.
        time:    [float (units: s)]; Correpsonding time for each command.              
    
    GLOBAL PARAMETERS:
    sampling_rate: [int (samples/s)]; Sampling rate at which the data was recorded.
    
    OUTPUT VARIABLES:
    bci2k_times:    [array > floats (units: ms)]; Time array at the sampling rate of BCI2000 which is bounded by
                    the final packet time.
    commands_bci2k: [list > strings ('nothing'/'click')]; Upsampled commands list at the BCI2000 sampling rate.
    """

    # COMPUTATION:
    
    # Computing the total number of frames in the entire window. Rounding, because due to numerical imprecision, 
    # the last packet time might be a number whose decimal is 0.99999999, and should be rounded to 1.
    last_packet_time  = commands[list(commands.keys())[-1]]['time']
    t_diff            = last_packet_time - t_start
    n_bci2k_samples   = round(t_diff * sampling_rate) # s * samples/s = samples
    
    # Adding an additional sample to account for the sample lost in subtraction. For example, the difference 
    # between 0.100 and 0.000 is 0.1 (100 samples). However this does not account for the sample at 0.000 s.
    n_bci2k_samples += 1
    
    # Computing the array of upsampled packets according to the BCI2000 sampling rate.
    bci2k_times = np.arange(n_bci2k_samples)/sampling_rate # samples / (samples/sec) = samples * sec/samples = secs 
    
    # Zeroing the bci2k times to the beginning of the first packet.
    bci2k_times += t_start
    
    # Rounding to eliminate numerical inaccuracy.
    bci2k_times = np.round(bci2k_times, 3)
    
    # Initializing a list of commands which corresponds to the BCI2000 sampling rate resolution. By default all 
    # entries are populated with the 'nothing' command, which simulates nothing being sent to the user-interface. 
    commands_bci2k = ['nothing'] * n_bci2k_samples
        
    # Iterating across all values in commands dictionary.
    for _, value in commands.items():

        # Extracting the commands and corresponding times.
        this_command     = value['command']
        this_packet_time = value['time'] 
                 
        # If the current command is 'click', find the index of the BCI2000 time corresponding to the packet time
        # of this command. This index is where the 'click' command will be placed in the upsampled commands list.
        if this_command == 'click':
            upsampled_ind                 = np.argwhere(bci2k_times == this_packet_time)[0][0]
            commands_bci2k[upsampled_ind] = this_command
                
    return bci2k_times, commands_bci2k





def visualizing_classification_outputs(model_outputs, t_visualization_start, t_visualization_end):
    """
    DESCRIPTION:
    Extracting the per-packet values for each key in the model_outputs dictionary and then plotting
    the model probabilities and packet classifications at between the experimenter-input starting and
    ending packet times.

    INPUT VARIABLES:
    model_outputs:         [dictionary (key: packet number; Values: below)]; Dictionary containing the
                           model probabilities, model classification, and corresponding time for each 
                           data packet.
        class:             [list > strings]; Classification for each packet.
        prob:              [array (1 x classes) > floats]; Per-packet model probabilities for each class.
        time:              [float (units: s)]; Corresponding time for each set of model probabilities.
    t_visualization_start: [float (units: ms)]; Starting visualization time point.
    t_visualization_end:   [float (units: ms)]; Ending visualization time point.
    """
    
    # COMPUTATION:
    
    # Computing the number of packets and classes.
    n_packets = len(model_outputs.keys())
    n_classes = model_outputs[0]['prob'].shape[0]

    # Initializing the packet classifications, model probabilities, and packet times.
    packet_classifications = [None] * n_packets
    model_probabilities    = np.zeros((n_packets,n_classes))
    packet_times           = np.zeros((n_packets,))

    # Iterating across each key of the model output.
    for p in model_outputs.keys():

        # Populating each of packet classifications list, array of model probabilities, and array of 
        # packet times.
        packet_classifications[p] = model_outputs[p]['class']
        model_probabilities[p,:]  = model_outputs[p]['prob']
        packet_times[p]           = model_outputs[p]['time']

    # Converting the packet classfications list to an array.
    packet_classifications = np.asarray(packet_classifications)

    # Computing the indicies where the packet times will be plotted.
    inds_plot_bool = np.logical_and(packet_times >= t_visualization_start, packet_times <= t_visualization_end)

    # Extracting the packet classifications, model probabilities and times for plotting.
    packet_classifications_plot = packet_classifications[inds_plot_bool]
    model_probabilities_plot    = model_probabilities[inds_plot_bool,:]
    packet_times_plot           = packet_times[inds_plot_bool]

    # PLOTTING

    # Initializing the figure.
    fig, ax = plt.subplots(2,1, figsize = (20,7.5))
    ax[0].plot(packet_times_plot, packet_classifications_plot)
    ax[0].set_ylabel('Classification')
    ax[1].plot(packet_times_plot, model_probabilities_plot)
    ax[1].grid()
    ax[1].set_ylabel('Model Probabilities')
    ax[1].set_xlabel('Time (s)')
    
    return None





def visualizing_click_highlights(click_highlights_video, t_visualization_start, t_visualization_end):
    """
    DESCRIPTION:
    Visualizing the clicks highlights between the experimneter-defined window.

    INPUT VARIABLES:
    click_highlights_video: [xarray > strings]; At the video resolution, there exists a click or no-click entry.
                            Time dimension in units of seconds at video resolution.
    t_visualization_start:  [float (units: ms)]; Starting visualization time point.
    t_visualization_end:    [float (units: ms)]; Ending visualization time point.
    """
    
    # COMPUTATION:
    
    # Extracting the indices for plotting.
    plotting_inds = np.logical_and(click_highlights_video.time_seconds >= t_visualization_start,\
                                   click_highlights_video.time_seconds <= t_visualization_end)

    # Extracting only the times and click highlights that will be plotted.
    click_highlights_times_plot = click_highlights_video.time_seconds[plotting_inds]
    click_highlights_plot       = click_highlights_video[plotting_inds]

    # PLOTTING
    fig = plt.figure(figsize=(20,5))
    plt.plot(click_highlights_times_plot, click_highlights_plot)
    plt.xlabel('Click Highlights')
    plt.ylabel('Time (s)')                          
    
    return None





def visualizing_commands_outputs(bci2k_times, commands_bci2k, t_visualization_start, t_visualization_end):
    """
    DESCRIPTION:
    Visualizing the commands for each packet between the experimneter-defined window.

    INPUT VARIABLES:
    bci2k_times:           [array > floats (units: ms)]; Time array at the sampling rate of BCI2000 which is 
                           bounded by the final packet time.
    commands_bci2k:        [list > strings ('nothing'/'click')]; Commands shifted by the number of samples 
                           corresponding to the networking delay.  
    t_visualization_start: [float (units: ms)]; Starting visualization time point.
    t_visualization_end:   [float (units: ms)]; Ending visualization time point.
    """
    
    # COMPUTATION:
    
    # Extracting the indices for plotting.
    plotting_inds = np.logical_and(bci2k_times >= t_visualization_start, bci2k_times <= t_visualization_end)

    # Extracting only the times and commands that will be plotted.
    bci2k_times_plot    = bci2k_times[plotting_inds]
    commands_bci2k_plot = np.asarray(commands_bci2k)[plotting_inds]

    # PLOTTING
    fig = plt.figure(figsize=(20,5))
    plt.plot(bci2k_times_plot, commands_bci2k_plot)
    plt.xlabel('Commands')
    plt.ylabel('Time (s)')
    
    return None