
# IMPORTING LIBRARIES
import numpy as np
import re
from scipy.io import loadmat, savemat





def creating_time_array(bci2000_data):
    """
    DESCRIPTION:
    Creating the time array according to the total number of data samples and the sampling rate.
    
    INPUT VARIABLES:
    bci2000_data: [dictionary]; Contains all experimentally-related signals, states, and parameters.
    
    OUTPUT VARIABLES:
    t_seconds: [array (time samples,) > floats (units: s)]; Time array for the recorded data block.
    """
    
    # COMPUTATION:
    
    # Computing the number of recorded samples.
    n_samples = bci2000_data['signal'].shape[0]

    # Extracting the sampling rate.
    sampling_rate_string = bci2000_data['parameters']['SamplingRate']['Value']
    sampling_rate        = int(re.sub("[^0-9]", "", sampling_rate_string))
    
    # Creating the time array. Note that the first time sample is not 0 s, because the first recorded signal sample does
    # not correspond to a time at 0 s.
    t_seconds = (np.arange(n_samples) + 1)/sampling_rate
    
    return t_seconds





def curtail_bci2000_data_by_block_start_stop_times(signals, states_dict, t_seconds, t_start, t_stop):
    """
    DESCRIPTION:
    Curtailing the signals, each array of the states dictionary, and time array such that only the samples 
    corresponding to time samples between the block start and stop times remain. 
    
    INPUT VARIABLES:
    signals:     [array (time samples, channels) > floats (units: uV)]; Array of signals for each neural and analog 
                 channel at each time point. Time samples are curtailed according to the time lag. 
    states_dict: [dictionary (Key: string (state); Value: array (time samples,) > ints)]; For each key, there exists an 
                 array of integer values describing the state at each time sample. For each array, time samples are 
                 curtailed according to the time lag. 
    t_seconds:   [array (time samples,) > floats (units: s)]; Time array for the recorded data block. Shifted by the
                 time lag and curtailed such that the resulting negative times are deleted.
    t_start:     [float (units: s)]; True starting time of the block.
    t_stop:      [float (units: s)]; True ending time of the block.
                
    OUTPUT VARIABLES:
    signals:     [array (time samples, channels) > floats (units: uV)]; Array of signals for each neural and analog 
                 channel at each time point. Time samples are curtailed according to the time lag. Time samples are
                 curtailed  according to the block start and stop times.
    states_dict: [dictionary (Key: string (state); Value: array (time samples,) > ints)]; For each key, there exists an 
                 array of integer values describing the state at each time sample. For each array, time samples are 
                 curtailed according to the block start and stop times. 
    t_seconds:   [array (time samples,) > floats (units: s)]; Time array for the recorded data block. Time samples are
                 curtailed according to the block start and stop times.
    """
    
    # COMPUTATION:
    
    # Computing the boolean array for the time samples only between the block start and 
    # stop times.
    curt_bool = np.logical_and(t_seconds >= t_start, t_seconds <= t_stop)

    # Extracting only the time samples that fall between the block start and stop times.
    t_seconds = t_seconds[curt_bool]

    # Extracting only the signals that fall between the block start and stop times.
    signals = signals[curt_bool,:]

    # Iterating across each state.
    for this_state in states_dict.keys():

        # Extracting the current state array from the state dictionary.
        this_state_array = states_dict[this_state]

        # Curtailing the current state array.
        this_state_array_curt = this_state_array[curt_bool]

        # Updating the state dict with the current curtailed state array.
        states_dict[this_state] = this_state_array_curt 

    return signals, states_dict, t_seconds





def curtail_bci2000_data_by_time_lag(bci2000_data, t_lag_seconds, t_seconds):
    """
    DESCRIPTION:
    Shifting the time array by the time lag and deleting the part of the resulting array that contains negative time
    values. Deleting the corresponding parts of the signals array and each array in the states dictionary.
    
    INPUT VARIABLES:
    bci2000_data:   [dictionary]; Contains all experimentally-related signals, states, and parameters.
    t_lag_seconds:  [float (units: s)]; The time lag between the audio from BCI2000 and the video audio. In other words, 
                    t_lag is the amount of time that the BCI2000 audio signal is leading the video audio signal. If 
                    t_lag = 150 ms, this means that BCI2000 audio is ahead of the video audio by 150 ms. For example, an
                    audio event registered by the video to be at 3.0 s would actually be registered at 3.15 s by 
                    BCI2000. 
    t_seconds:      [array (time samples,) > floats (units: s)]; Time array for the recorded data block.

    OUTPUT VARIABLES:
    signals:     [array (time samples, channels) > floats (units: uV)]; Array of signals for each neural and analog 
                 channel at each time point. Time samples are curtailed according to the time lag. 
    states_dict: [dictionary (Key: string (state); Value: array (time samples,) > ints)]; For each key, there exists an 
                 array of integer values describing the state at each time sample. For each array, time samples are 
                 curtailed according to the time lag. 
    t_seconds:   [array (time samples,) > floats (units: s)]; Time array for the recorded data block. Shifted by the
                 time lag and curtailed such that the resulting negative times are deleted.
    """
    
    # COMPUTATION:
    
    # Extracting the BCI2000 time array only after lag.
    bci2k_lag        = t_seconds - t_lag_seconds # units of seconds. 
    bci2k_bool       = bci2k_lag > 0
    bci2k_times_bool = bci2k_lag[bci2k_bool]

    # Zeroing the BCI2000 time array after lag. 
    t_seconds = bci2k_times_bool - bci2k_times_bool[0] + t_seconds[0]

    # Extracting the signals from the BCI2000 data dictionary.
    signals = bci2000_data['signal']

    # Extracting the state dictionary from the BCI2000 data dictionary.
    states_dict = bci2000_data['states']

    # Curtailing the signals to keep only the signals after the time lag.
    signals = signals[bci2k_bool,:]

    # Iterating across each state.
    for this_state in states_dict.keys():

        # Extracting the current state array from the state dictionary.
        this_state_array = states_dict[this_state]

        # Curtailing the current state array.
        this_state_array_curt = this_state_array[bci2k_bool]

        # Updating the state dict with the current curtailed state array.
        states_dict[this_state] = this_state_array_curt
    
    return signals, states_dict, t_seconds





def load_bci2000_data(block_id, date, dir_bci2000_data, patient_id, task):
    """
    DESCRIPTION:
    Loading the matlab file whose data (signals and states) we will align and crop.
    
    INPUT VARIABLES:
    block_id:          [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:              [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_bci2000_data:  [string]; Directory where the BCI2000 data is stored.
    patient_id:        [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:              [string]; Type of task that was run.

    OUTPUT VARIABLES:
    bci2000_data: [dictionary]; Important keys are 'signals' and 'states'.
    """
    
    # COMPUTATION:
    
    # Creating the directory and the filename for the .mat file.
    dir_mat_file      = dir_bci2000_data + patient_id + '/mat/' + date + '/'
    filename_mat_file = task + '_' + block_id + '.mat'

    # Creating the pathway for the .mat file.
    pathway_mat_file = dir_mat_file + filename_mat_file

    # Loading the BCI2000 data in a matlab file.
    bci2000_data = loadmat(pathway_mat_file, simplify_cells=True)
    
    return bci2000_data





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
    t_stop:  [float (units: s)]; True ending time of the block.
    """
    
    # COMPUTATION:
    
    # Creating the directory and filename for the .txt file containing the starting and ending times of the block.
    dir_start_stop      = dir_intermediates + patient_id + '/Speller/BlocksStartAndStops/' + date + '/'
    filename_start_stop = date + '_' + block_id +'_StartStop.txt'

    # Creating the pathway for the .txt file.
    path_start_stop = dir_start_stop + filename_start_stop

    # Opening up the block starting and ending times from the pathway.
    txt_file_start_stop = open(path_start_stop)

    # Reading the content of the text file with the start and stop times.
    text_file_lines = txt_file_start_stop.readlines()

    # Reading in the lines with the starting and stopping times.
    line_time_start = text_file_lines[5]
    line_time_stop  = text_file_lines[6]

    # Extracting the start and stop times from the corresponding strings.
    t_start = float(line_time_start[7:])
    t_stop  = float(line_time_stop[7:])

    return t_start, t_stop





def load_time_lag(block_id, date, dir_intermediates, patient_id, task):
    """
    DESCRIPTION:
    Loading the time lag between BCI2000 and the video recording.
    
    INPUT VARIABLES:
    block_id:          [String (BlockX, where X is an int)]; Block ID of the task that was run.
    date:              [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_intermediates: [string]; [string]; Intermediates directory where relevant information is stored.
    patient_id:        [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:              [string]; Type of task that was run. 
    
    OUTPUT VARIABLES:
    t_lag_seconds: [float (units: s)]; The time lag between the audio from BCI2000 and the video audio. In other words, 
                   t_lag is the amount of time that the BCI2000 audio signal is leading the video audio signal. If 
                   t_lag =  150 ms, this means that BCI2000 audio is ahead of the video audio by 150 ms. For example, 
                   an audio event registered by the video to be at 3.0 s would actually be registered at 3.15 s by
                   BCI2000. 
    """
    
    # COMPUTATION:
    
    # Creating the base path and filename for the time lags.
    dir_lag      = dir_intermediates + patient_id + '/' + task + '/' + 'LagsBetweenVideoAndBCI2000/' + date + '/' +\
                   block_id + '/'
    filename_lag = date + '_' + block_id + '.txt'
    
    # Creating the pathway for the lag for the current date+block pair.
    path_lag = dir_lag + filename_lag

    # Opening up the text file with the time lag from the pathway.
    txt_file_t_lag = open(path_lag)

    # Reading the content of the text file with the time lag.
    text_file_lines = txt_file_t_lag.readlines()

    # Extracting the time lag from the text file line.
    t_lag = int(text_file_lines[0])

    # Converting the lag into units of seconds.
    t_lag_seconds = t_lag/1000 # ms * s/ms = s
    
    # PRINTING:
    print('TIME LAG (s):', t_lag_seconds) 

    return t_lag_seconds





def save_curtailed_BCI2000_data(bci2000_data, block_id, date, dir_bci2000_data, patient_id, signals, states_dict,\
                                t_seconds, task):
    """
    DESCRIPTION:
    Overwriting the signals and states values in the BCI2000 data dictionary. Also adding a 'time' key to which holds
    the time array. Saving this BCI2000 data as a new .mat file. 
    
    INPUT VARIABLES:
    bci2000_data:     [dictionary]; Contains all experimentally-related signals, states, and parameters.
    block_id:         [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:             [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_bci2000_data: [string]; Directory where the BCI2000 data is stored.
    patient_id:       [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    signals:          [array (time samples, channels) > floats (units: uV)]; Array of signals for each neural and analog 
                      channel at each time point. Time samples are curtailed according to the time lag. Time samples are
                      curtailed  according to the block start and stop times.
    states_dict:      [dictionary (Key: string (state); Value: array (time samples,) > ints)]; For each key, there 
                      exists an array of integer values describing the state at each time sample. For each array, time 
                      samples are curtailed according to the block start and stop times. 
    t_seconds:        [array (time samples,) > floats (units: s)]; Time array for the recorded data block. Time samples
                      are curtailed according to the block start and stop times.
    task:             [string]; Type of task that was run.
    """
    
    # COMPUTATION:
    
    # Overwriting the BCI2000 dictionary with the updated signals array and states dictionary. Adding a 'time' key
    # also.
    bci2000_data['signal'] = signals
    bci2000_data['states'] = states_dict
    bci2000_data['time']   = t_seconds
    
    # Creating the directory and the filename for the "Adjusted" .mat file.
    dir_adj_mat_file = dir_bci2000_data + patient_id + '/mat/' + date + '/'
    filename_mat_file = task + '_Adjusted_' + block_id + '.mat'

    # Creating the pathway for the "Adjusted" .mat file.
    pathway_adj_mat_file = dir_adj_mat_file + filename_mat_file

    # Saving the BCI2000 data to an "Adjusted" .mat file.
    savemat(pathway_adj_mat_file, bci2000_data, long_field_names = True)
    
    return None