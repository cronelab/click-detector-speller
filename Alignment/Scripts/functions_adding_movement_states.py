
# IMPORTING LIBRARIES
import numpy as np
import os
import pickle
import re

from pprint import pprint
from scipy.io import loadmat, savemat





def adding_movement_states(bci2000_data, movement_onsetsoffsets, t_interval):
    """
    DESCRIPTION:
    Adding movement onset and offset states (MovementOnset and MovementOffset) to the BCI2000 data. For each movement 
    type a corresponding numerical value is assigned in both the MovementOnset and MovementOffset state arrays. For 
    example, grasp movements will correspond to values of 1 in both state arrays while thumb_flexion movements may 
    correspond to values of 2. Additionally, each state change will be an interval length in both state arrays as 
    opposed to simply a discrete one-sample change. For example, a MovementOnset and MovementOffset state array may look
    like the following:
    
    MovementOnset:
    [0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0]
    
    MovementOffset:
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0]
    
    Notice that in the two state arrays above the state changes corresponding to grasp and thumb_flexion movement onsets
    and offsets (values of 1 and 2, respectively) are of interval length and each state change is not one sample long. 
    
    The reason for this is because in further scripts in which spectrograms of the data are computed, the stimulus array
    must be downsampled to match the time resolution of the spectral data. If the state array contains state changes 
    that are just one sample long, these state changes will likely be aliased out when downsampling. That is, they wil
    not be part of the downsampled state array. As such, the interval length of each state change must be at least as
    long as the spectral shift. The longest that this spectral shift is planned to be is 0.1 s. By making each state at 
    least the size of the spectral shift, it guarantees that the state change is not aliased out during downsampling. Be
    sure that the block true ending time can accomodate the length of the last MovementOffst state change.
    
    INPUT VARIABLES:
    bci2000_data:           [dictionary]; Contains all experimentally-related signals, states, and parameters.
    movement_onsetsoffsets: [dictionary (key: string (movement); Value: list > list [t_onset, t_offset] > floats 
                            (units: s))]; The dictionary containing all movement onset and offset times for each 
                            movement type.
    t_interval:             [float (units: s)]; Time length of each state change in the movement state array.
    
    OUTPUT VARIABLES:
    bci2000_data: [dictionary]; Contains all experimentally-related signals, states, and parameters. States dictionary
                  is updated to contain the MovemnetOnset and MovementOffset states.
    """
    
    # COMPUTATION:
    
    # Extracting the sampling rate.
    sampling_rate_string = bci2000_data['parameters']['SamplingRate']['Value']
    sampling_rate        = int(re.sub("[^0-9]", "", sampling_rate_string))

    # Converting the time interval into samples.
    interval_samples = int(0.1 * sampling_rate) # s x samples/s = samples

    # Extracting the time array from the BCI2000 dictionary.
    t_seconds = bci2000_data['time']

    # Extracting the states from BCI2000 data.
    states_dict = bci2000_data['states']

    # Computing the total number of time samples.
    n_samples = t_seconds.shape[0]
    
    # Initializing the state arrays for movement onset and offset times.
    movement_onset_state_array  = np.zeros((n_samples,))
    movement_offset_state_array = np.zeros((n_samples,))

    # Iterating across all movement types:
    for n, this_movement_type in enumerate(movement_onsetsoffsets.keys()):

        # Extracting the onset and offset times for the current movement.
        this_movement_onset_offset_times = movement_onsetsoffsets[this_movement_type]

        # Initializing lists of movement onset and offset inds.
        this_movement_onset_inds  = []
        this_movement_offset_inds = []

        # Iterating across all movement times for the current movement.
        for t_onset, t_offset in this_movement_onset_offset_times:

            # Determining movement onset and offset starting and ending indices. The starting and ending indices 
            # describe the artificial time interval for movement onset and movement offset. 
            movement_onset_start_ind  = np.argmin(np.abs(t_seconds - t_onset))
            movement_offset_start_ind = np.argmin(np.abs(t_seconds - t_offset))
            movement_onset_end_ind    = movement_onset_start_ind + interval_samples
            movement_offset_end_ind   = movement_offset_start_ind + interval_samples
            
            # Computing arrays that determine the current movement onset and offset intervals.
            this_onset_interval  = np.arange(movement_onset_start_ind, movement_onset_end_ind).tolist()
            this_offset_interval = np.arange(movement_offset_start_ind, movement_offset_end_ind).tolist()

            # Appending the current interval arrays to the movement onset and offset indices list.        
            this_movement_onset_inds.extend(this_onset_interval)
            this_movement_offset_inds.extend(this_offset_interval)    
        
        # Updaging the state array with the movement onset and offset intervals for the current movement.
        movement_onset_state_array[this_movement_onset_inds]   = n+1
        movement_offset_state_array[this_movement_offset_inds] = n+1

    # Adding the movement onset and offset state arrays to the states dictionary.
    states_dict['MovementOnset']  = movement_onset_state_array
    states_dict['MovementOffset'] = movement_offset_state_array

    # Updating the states subdictionary in the BCI2000 data dictionary.
    bci2000_data['states'] = states_dict

    return bci2000_data





def creating_time_array(bci2000_data):
    """
    DESCRIPTION:
    Creating the time array according to the total number of data samples and the sampling rate.
    
    INPUT VARIABLES:
    bci2000_data: [dictionary]; Contains all experimentally-related signals, states, and parameters.
    
    OUTPUT VARIABLES:
    t_seconds: [array (1 x time samples) > floats (units: s)]; Time array for the recorded data block.
    """
    
    # COMPUTATION:
    
    # Computing the number of recorded samples.
    n_samples = bci2000_data['signal'].shape[0]

    # Extracting the sampling rate.
    sampling_rate_string = bci2000_data['parameters']['SamplingRate']['Value']
    sampling_rate        = int(re.sub("[^0-9]", "", sampling_rate_string))
    
    # Creating the time array. Note that the first time sample is not 0 s, because the first recorded
    # signal sample does not correspond to a time at 0 s.
    t_seconds = (np.arange(n_samples) + 1)/sampling_rate
    
    return t_seconds





def load_bci2000_data(block_id, date, dir_bci2000_data, patient_id, task):
    """
    DESCRIPTION:
    Loading the matlab file whose data (signals and states) we will align and crop.
    
    INPUT VARIABLES:
    block_id:         [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:             [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_bci2000_data: [string]; Directory where the BCI2000 data is stored.
    patient_id:       [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:             [string]; Type of task that was run.

    OUTPUT VARIABLES:
    bci2000_data: [dictionary]; Important keys are 'signals' and 'states'.
    """
    
    # COMPUTATION:
    
    # Creating the directory and the filename for the .mat file.
    dir_mat_file      = dir_bci2000_data + patient_id + '/mat/' + date + '/'
    filename_mat_file = task + '_Adjusted_' + block_id + '.mat'

    # Creating the pathway for the .mat file.
    pathway_mat_file = dir_mat_file + filename_mat_file

    # Loading the BCI2000 data in a matlab file.
    bci2000_data = loadmat(pathway_mat_file, simplify_cells=True)
    
    return bci2000_data





def save_BCI2000_data_with_new_states(bci2000_data, block_id, date, dir_bci2000_data, patient_id, task):
    """
    DESCRIPTION:
    Saving this BCi2000 data with the new state arrays as a new .mat file. 
    
    INPUT VARIABLES:
    bci2000_data:     [dictionary]; Contains all experimentally-related signals, states, and parameters.
    block_id:         [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:             [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_bci2000_data: [string]; Directory where the BCI2000 data is stored.
    patient_id:       [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:             [string]; Type of task that was run.
    """
    
    # COMPUTATION:
    
    # Creating the directory and the filename for the "Adjusted" .mat file.
    dir_adj_mat_file = dir_bci2000_data + patient_id + '/mat/' + date + '/'
    filename_mat_file = task + '_Adjusted_' + block_id + '.mat'

    # Creating the pathway for the "Adjusted" .mat file.
    pathway_adj_mat_file = dir_adj_mat_file + filename_mat_file

    # Saving the BCI2000 data to an "Adjusted" .mat file.
    savemat(pathway_adj_mat_file, bci2000_data, long_field_names = True)
    
    return None





def upload_movement_onsetsoffsets(block_id, date, dir_intermediates, patient_id, task):
    """
    DESCRIPTION:
    The dictionary containing the movement onset and offset times for each movement type will be uploaded if it exists. 
    This dictionary would contain the previously saved movement onset/offset times for each movement. If there were no
    previously saved onset/offset times for a particular movement (or all movements) a dictionary will be initiated and
    saved for current and future inputting of onset/offset times.
    
    INPUT VARIABLES:
    block_id:          [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:              [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_intermediates: [string]; [string]; Intermediates directory where relevant information is stored.
    patient_id:        [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:              [string]; Type of task that was run.

    OUTPUT VARIABLES:
    movement_onsetsoffsets: [dictionary (key: string (movement); Value: list > list [t_onset, t_offset] > floats)]; The 
                            dictionary containing all movement onset and offset times for each movement type.
    """
    
    # COMPUTATION:
    
    # Creating the directory and filename to the movement onsets/offsets dictionary.
    dir_onsetsoffsets      = dir_intermediates + patient_id + '/' + task + '/MovementOnsetsAndOffsets/' + date + '/'
    filename_onsetsoffsets = 'dict_OnsetOffset_' + block_id
    
    # Creating the pathway to the movement onset/offset dictionary.
    path_onsetsoffsets_dict = dir_onsetsoffsets + filename_onsetsoffsets
    
    # Checking to make sure the pathway exists.
    pathway_exists = os.path.exists(path_onsetsoffsets_dict)
    
    # The onsets/offsets dictionary exists in the specified pathway.
    if pathway_exists:
        
        # Read in the dictionary from the pathway.
        with open(path_onsetsoffsets_dict, "rb") as fp:
            movement_onsetsoffsets = pickle.load(fp)
            
        # Print the dictionary.
        pprint(movement_onsetsoffsets)
        
    # The onsets/offsets dictionary does not exist in the specified pathway.
    else:
        print('Cannot find dictionary of movement onset and offset times.')
    
    return movement_onsetsoffsets