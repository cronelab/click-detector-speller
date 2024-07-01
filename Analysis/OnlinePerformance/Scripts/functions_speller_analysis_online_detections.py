
import collections
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil
import xarray as xr

from pprint import pprint

from scipy.io import loadmat





def combining_clicks(click_info):
    """
    DESCRIPTION:
    Combining all click types from the click_info_ui into one array.

    INPUT VARIABLES:
    click_info_ui: [dict (key: string ('backspace','keyboard','stimcolumn'); Values: below)];
        data:      [xarray (time samples,) > strings];  For each time sample of the array of each key there is a 
                   'no_click' string or a click-string specific to that xarray. For example, the 'backspace' key of the
                   dictionary has an array where each element is a string named either 'no_click' or 'backspace_click'.
                   The 'backspace_click' elements do not occur  consecutively and describe the instance a click on the
                   backspace key occured. For the 'keyboard' and 'stimcolumn' keys, similar rules apply. Time dimension
                   is in units of s.
        plotcolor: [string]; Color corresponding to the type of click for plotting.

    OUTPUT VARIABLES:
    clicks_combined: [xarray (N frames, )> strings ('nothing'/'click')]; Xarray containing click information from all 
                     click types.
    """
    
    # COMPUTATION:
    
    # Initializing a flag that is only True for the first click type.
    flag_click_type = True

    # Iterating across all types of clicks.
    for this_click_type in click_info.keys():

        # Extracting the information from the current type of click.
        this_click_info = click_info[this_click_type]

        # Extracting the data from the current click type.
        this_click_data = this_click_info['data'].values

        # If this is the first click type.
        if flag_click_type:

            # Extracting the time array that is shared across all clicks.
            t_seconds = this_click_info['data'].time_seconds.values

            # Computing the total number of video frames.
            n_frames = t_seconds.shape[0]

            # Create an list of all combined clicks.
            clicks_combined = np.asarray(['nothing'] * n_frames)

            # Extracting the indices where the current click occurred.
            click_type_inds = np.squeeze(np.argwhere(this_click_data == this_click_type + '_click'))

            # Replacing the correct indices in the combined clicks array with just 'click'.
            clicks_combined[click_type_inds] = 'click'

            # Setting flag to false to not enter this IF statement again.
            flag_click_type = False

        else:

             # Extracting the indices where the current click occurred.
            click_type_inds = np.squeeze(np.argwhere(this_click_data == this_click_type + '_click'))

            # Replacing the correct indices in the combined clicks array with just 'click'.
            clicks_combined[click_type_inds] = 'click'

        # Convertng the combined clicks array into an xarray.
        clicks_combined =  xr.DataArray(clicks_combined,
                                        coords={'time_seconds': t_seconds},
                                        dims=["time_seconds"])

    return clicks_combined





def computing_fp(movement, movement_onsetsoffsets, t_after_movement_limit, t_click_onsets):
    """
    DESCRIPTION:
    Counting the number of clicks that occur without any corresponding movement (outside the click limit window). These
    are not necessarily the number of false positives. This is because some of these "clicks" may be due to the speller
    highlighter cycling back to the first element in a particular row. Therefore, at the time of each of these 
    independently occurring "clicks", it is necessary to go back to the video and check whether or not they were actual
    clicks or simply caused by the automated highlighting of a row's first element.

    INPUT VARIABLES:
    movement:               [string]; The movement from which the onsets and offsets will be extracted.
    movement_onsetsoffsets: [dictionary (key: string (movement); Value: list > list [t_onset, t_offset] > floats)]; The 
                            dictionary containing all movement onset and offset times for each movement type.
    t_after_movement_limit: [float (unit: s)]; Amount of time from movement onset that can pass for a click to occur and
                            be associated with the movement onset.
    t_click_onsets:         [array > floats (units: s)]; The times the click changes from 0 to 1. 

    OUTPUT VARIABLES:
    n_fp: [int]; Total number of potential false positives for the current block of the session.
    """
    
    # COMPUTATION:
    
    # Initializing the FP count.
    n_fp = 0
    
    # Extracting the list of movement onsets and offsets.
    these_onset_offset_pairs = movement_onsetsoffsets[movement]
    
    # Extracting the movement onsets.
    movement_onsets = np.array([this_onset for this_onset, _ in these_onset_offset_pairs])
        
    # Finding potential false positives.
    print('False Positives?')
    
    # Iterating across all command onset times.
    for this_click_onset in t_click_onsets:
        
        # Computing the time limit in which a click onset could be produced such that it counts as due to the current
        # movement onset.
        t_movement_limit = this_click_onset - t_after_movement_limit
                
        # Extracting the time(s) which falls within the period between the movement onset period and time command limit.
        bool_movements   = np.logical_and(movement_onsets >= t_movement_limit, movement_onsets <= this_click_onset)
        t_bool_movements = movement_onsets[bool_movements]
        
        # If there were no accompanying movements associated with the command change, then that is a potential false
        # positive.
        if not t_bool_movements.any():
            print(round(this_click_onset, 2))
            n_fp += 1
              
    return n_fp





def computing_latencies(movement, movement_onsetsoffsets, t_after_movement_limit, t_click_onsets):
    """
    DESCRIPTION:
    Computing the latency from movement onset to click detection. 
    
    INPUT VARIABLES:
    movement:               [string]; The movement from which the onsets and offsets will be extracted.
    movement_onsetsoffsets: [dictionary (key: string (movement); Value: list > list [t_onset, t_offset] > floats)]; The 
                            dictionary containing all movement onset and offset times for each movement type.
    t_after_movement_limit: [float (unit: s)]; Amount of time from movement onset that can pass for a click to occur 
                            and be associated with the movement onset.
    t_click_onsets:         [array > floats (units: s)]; The times the click changes from 0 to 1.
    
    OUTPUT VARIABLES:
    t_click_latencies: [array > floats (units: s)]; Latencies of all detected clicks relative to corresponding
                       movement onset.
    """

    # COMPUTATION:

    # Extracting the list of movement onsets and offsets.
    these_onset_offset_pairs = movement_onsetsoffsets[movement]
    
    # Extracting the movement onsets.
    movement_onsets = np.array([this_onset for this_onset, _ in these_onset_offset_pairs])
    
    # Defining a list which will hold all of the latencies between movement onset and click onset, given that
    # a click occurs within the experimenter-defiend movement limit.
    t_click_latencies = []    
    
    # Iterating across all movement onsets.
    for this_movement_onset in movement_onsets:
        
        # Computing the time limit after movement onset in which a click could be produced such that it is associated
        # with the current attempted movement.
        t_click_limit = this_movement_onset + t_after_movement_limit
        
        # Extracting the time(s) which falls within the period between the movement onset and click onset.
        bool_limit   = np.logical_and(t_click_onsets >= this_movement_onset, t_click_onsets <= t_click_limit)
        t_click_bool = t_click_onsets[bool_limit]

        # If there is any click in the click time limit.
        if t_click_bool.any(): 

            # Extracting the time of the click. 
            this_click_onset = t_click_bool[0]

            # Computing the latency from movement onset to click onset and appending it to the list of click latencies.
            this_click_latency = this_click_onset - this_movement_onset
            t_click_latencies.append(this_click_latency)

    # Converting the click latencies into an array and saving them to a dictionary.
    t_click_latencies = np.asarray(t_click_latencies)
    
    # Computing the mean and standard deviation of the latencies.  
    mean_latency  = round(np.mean(t_click_latencies), 3)
    stdev_latency = round(np.std(t_click_latencies), 3)

    # PRINTING
    print('Mean Latency to Command: ', mean_latency)
    print('Stdev Latency to Command: ', stdev_latency)
    
    # PLOTTING
    
    # Plotting the histogram
    fig = plt.figure();
    plt.hist(t_click_latencies, bins = 20);
    plt.title('Latency to Click Detection');
    plt.xlabel('Time (s)');
    plt.ylabel('Trials');
    plt.grid();
    
    return t_click_latencies





def computing_sensitivities(movement, movement_onsetsoffsets, t_after_movement_limit, t_click_onsets):
    """
    DESCRIPTION:
    Computing the sensitivity. The sensitivity is defined as the number of true positives over the number of all click 
    detections, where a true positive must occur within the experimenter-defined time post-movement onset.
    
    INPUT VARIABLES:
    movement:               [string]; The movement from which the onsets and offsets will be extracted.
    movement_onsetsoffsets: [dictionary (key: string (movement); Value: list > list [t_onset, t_offset] > floats)]; The 
                            dictionary containing all movement onset and offset times for each movement type.
    t_after_movement_limit: [float (unit: s)]; Amount of time from movement onset that can pass for a click to occur 
                            and be associated with the movement onset.
    t_click_onsets:         [array > floats (units: s)]; The times the click changes from 0 to 1.
    
    OUTPUT VARIABLES:
    sensitivity: [float (units: %)]; Percentage of correctly detected clicks.
    """

    # COMPUTATION:

    # Extracting the list of movement onsets and offsets.
    these_onset_offset_pairs = movement_onsetsoffsets[movement]
    
    # Extracting the movement onsets and setting the number of movement onsets as the number of desired clicks.
    movement_onsets  = np.array([this_onset for this_onset, _ in these_onset_offset_pairs]) 
    n_click_attempts = movement_onsets.shape[0]
    
    # Initializing the number of correctly detected clicks.
    n_correct_clicks = 0
    
    # Iterating across all movement onsets.
    for this_movement_onset in movement_onsets:
        
        # Computing the time limit after movement onset in which a click could be produced such that it is associated
        # with the current attempted movement.
        t_click_limit = this_movement_onset + t_after_movement_limit
        
        # Extracting the time(s) which falls within the period between the movement onset and click onset.
        bool_limit   = np.logical_and(t_click_onsets >= this_movement_onset, t_click_onsets <= t_click_limit)
        t_click_bool = t_click_onsets[bool_limit]

        # If there is any click in the click time limit, update the number of clicks.
        if t_click_bool.any(): 
            n_correct_clicks += 1

        # If no click occurred within the time limit, print the time of the movement onset.
        else:
            print('Miss: ', round(this_movement_onset, 3))
 
    # Computing the sensitivity.
    sensitivity = round((n_correct_clicks/n_click_attempts)*100, 2)
    
    # PRINTING
    print('Sensitivity: ', sensitivity)

    return sensitivity





def extracting_click_onset_offset_times_bci2k(click_info_bci2k):
    """
    DESCRIPTION:
    Producing the arrays of click onsets and offsets from BCI2000.
    
    INPUT VARIABLES:
    click_info_bci2k: [xarray (time samples, ) > ints (0 or 1)]; Xarray of click states at each time sample. Time
                      dimension is in units of seconds.
    
    NECESSARY FUNCTIONS:
    unique_value_index_finder
    
    OUTPUT VARIABLES:
    t_onsets:  [array > floats (units: s)]; The times the click changes from 0 to 1. 
    t_offsets: [array > floats (units: s)]; The times the click changes from 1 to 0.
    """
    
    # COMPUTATION:
    
    # Extracting the time array from the click trace array and onverting the click trace xarray to a regular array.
    t_seconds   = click_info_bci2k.time_seconds.values
    click_trace = click_info_bci2k.values
        
    # Extracting the starting and ending indices of each click and no-click period.
    unique_vals, _, _, start_end_inds = unique_value_index_finder(click_trace)
    
    # Using the 'click' index in the unqiue_vals list to extract the starting and ending times of the click periods.
    click_index       = unique_vals.index(1)    
    onset_offset_inds = start_end_inds[click_index]
        
    # Computing the number of command changes (whether its BCI2000 state changes or UI clicks).
    n_click_changes = len(onset_offset_inds)
    
    # Initializing an array of click onset/offset times for each click period.
    t_onsets  = np.zeros((n_click_changes,)) 
    t_offsets = np.zeros((n_click_changes,)) 
    
    # Iterating across all pairs of state indices.
    for n, (onset_ind, offset_ind) in enumerate(onset_offset_inds):
        
        # Extracting the onset and offset times using the respective indices.
        t_onsets[n]  = t_seconds[onset_ind]
        t_offsets[n] = t_seconds[offset_ind]
    
    return t_onsets, t_offsets





def extracting_click_onset_offset_times_ui(click_trace):
    """
    DESCRIPTION:
    Producing the arrays of command onsets and offsets.
    
    INPUT VARIABLES:
    click_trace: [xarray > strings)]; for each time sample at the video resolution, there exists a click or no-click 
                 entry. Time dimension is in units of seconds at video resolution.
    
    NECESSARY FUNCTIONS:
    unique_value_index_finder
    
    OUTPUT VARIABLES:
    t_onsets:  [array > floats (units: s)]; The times the click changes from 0 to 1. 
    t_offsets: [array > floats (units: s)]; The times the click changes from 1 to 0.
    """
    
    # COMPUTATION:
    
    # Extracting the time array from the click trace array and onverting the click trace xarray to a regular array.
    t_seconds   = click_trace.time_seconds
    click_trace = click_trace.values
        
    # Extracting the starting and ending indices of each click and no-click period.
    unique_vals, _, _, start_end_inds = unique_value_index_finder(click_trace)
    
    # Using the 'click' index in the unqiue_vals list to extract the starting and ending times of the click periods.
    click_index       = unique_vals.index('click')    
    onset_offset_inds = start_end_inds[click_index]
        
    # Computing the number of command changes (whether its BCI2000 state changes or UI clicks).
    n_click_changes = len(onset_offset_inds)
    
    # Initializing an array of click onset/offset times for each click period.
    t_onsets  = np.zeros((n_click_changes,)) 
    t_offsets = np.zeros((n_click_changes,)) 
    
    # Iterating across all pairs of state indices.
    for n, (onset_ind, offset_ind) in enumerate(onset_offset_inds):
        
        # Extracting the onset and offset times using the respective indices.
        t_onsets[n]  = t_seconds[onset_ind]
        t_offsets[n] = t_seconds[offset_ind]
    
    return t_onsets, t_offsets





def extracting_relevant_trajectories(hand_trajectories_ref, relevant_hand_landmarks):
    """
    DESCRIPTION:
    For each movement type, the experimenter enters the most relevant hand landmarks for visualization. The experimenter
    creates a relevant_hand_landmarks dictionary where the keys of the dictionary are the possible movement classes and
    the value for each key is a list of the most relevant hand landmarks for that class. The plotting cells above should
    be used to determine these landmarks. Then for each movement type a dictionary, hand_trajectories_relevant is 
    created where for each movement, only the relevant hand trajectories are stored.

    INPUT VARIABLES:
    hand_trajectories_ref:   [xarray (landmarks, time samples) > floats]; The trajectories of the x- and y-coordinates
                             for each hand landmark. These are referenced in the x- and y-dimensions according to the
                             reference landmarks. The time domain is in units of seconds. 
    relevant_hand_landmarks: [dictionary (key: string (movement type); Value: list > strings (hand landmarks))]; Each
                             movement holds a list of the most useful landmarks used to detect the corresponding
                             movemet type.
    
    OUTPUT VARIABLES:
    hand_trajectories_relevant: [dict (Key: string (movement type); Value: xarray (relevant landmarks, time samples) > 
                                floats]; For each movement type, only the relevant hand trajectories are stored. The
                                time dimension of each xarray is in units of s.   
    """
    
    # COMPUTATION:
    
    # Initializing the dictionary of relevant hand trajectories per movement.
    hand_trajectories_relevant = {}

    # Iterating across all movement types:
    for this_movement in relevant_hand_landmarks.keys():

        # Extracting the relevant landmarks for the current movement.
        this_movement_relevant_landmarks = relevant_hand_landmarks[this_movement]

        # Extracting only the trajectories of the relevant landmarks for the current movement.
        this_movement_hand_trajectories = hand_trajectories_ref.loc[this_movement_relevant_landmarks,:]

        # Assigning the hand trajectorires specific to this movement to the dictionary.
        hand_trajectories_relevant[this_movement] = this_movement_hand_trajectories
    
    return hand_trajectories_relevant





def load_bci2000_clicks(block_id, date, dir_bci2000_data, patient_id, task):
    """
    DESCRIPTION:
    Loading the matlab file from which the ControlClick state will be extracted. 
    
    INPUT VARIABLES:
    block_id:         [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:             [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_bci2000_data: [string]; Directory where the BCI2000 data is stored.
    patient_id:       [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:             [string]; Type of task that was run.

    OUTPUT VARIABLES:
    click_info_bci2k: [xarray (time samples, ) > ints (0 or 1)]; Xarray of ControlClick state at each time sample. Time
                      dimension is in units of seconds.
    """
    
    # COMPUTATION:
    
    # Creating the directory and the filename for the .mat file.
    dir_mat_file      = dir_bci2000_data + patient_id + '/mat/' + date + '/'
    filename_mat_file = task + '_Adjusted_' + block_id + '.mat'

    # Creating the pathway for the .mat file.
    pathway_mat_file = dir_mat_file + filename_mat_file

    # Loading the BCI2000 data in a matlab file.
    bci2000_data = loadmat(pathway_mat_file, simplify_cells=True)
    
    # Extracting the click information from the BCI2000 data.
    clicks_bci2k     = bci2000_data['states']['ControlClick']
    t_seconds_clicks = bci2000_data['time']
    
    # Transforming the click array into an xarray.
    clicks_bci2k = xr.DataArray(clicks_bci2k,
                                coords={'time_seconds': t_seconds_clicks},
                                dims=["time_seconds"])
    
    return clicks_bci2k





def load_click_information(block_id, date, dir_clickdetections):
    """
    DESCRIPTION:
    Loading the dictionary of click information from the user-interface..
    
    INPUT VARIABLES:
    block_id:            [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:                [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_clickdetections: [string]; Directory where the click information is stored.
    
    OUTPUT VARIABLES:
    click_info: [dict (key: string ('backspace','keyboard','stimcolumn'); Values: below)];
        data:      [xarray (time samples,) > strings];  For each time sample of the array of each key there is a 
                   'no_click' string or a click-string specific to that xarray. For example, the 'backspace' key of the
                   dictionary has an array where each element is a string named either 'no_click' or 'backspace_click'. 
                   The 'backspace_click' elements do not occur consecutively and describe the instance a click on the
                   backspace key occured. For the 'keyboard' and 'stimcolumn' keys, similar rules apply. Time dimension
                   is in units of s.
        plotcolor: [string]; Color corresponding to the type of click for plotting.
    """
    
    # COMPUTATION:
    
    # Creating the filename for the click information.
    filename_clickinfo = date + '_' + block_id + '_click_highlights'
    
    # Pathway for the click detections.
    path_clickinfo = dir_clickdetections + date + '/Curtailed/'  + filename_clickinfo
    
    # Loading the click information.
    click_info = np.load(path_clickinfo, allow_pickle=True)
    
    return click_info





def load_hand_trajectories(block_id, date, dir_handtrajectories):
    """
    DESCRIPTION:
    Importing the xarray of hand trajectories.
    
    INPUT VARIABLES:
    block_id:             [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:                 [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_handtrajectories: [string]; Directory where the hand trajectories are stored.
    
    OUTPUT VARIABLES:
    hand_trajectories: [xarray (landmarks, time samples) > floats]; The time traces of the x- and y-coordinates for each 
                       hand landmark. The time domain is in units of seconds. 
    """
    
    # COMPUTATION:
    
    # Creating the for the hand trajectories.
    filename_handtrajectories = date + '_' + block_id + '_hand_trajectories.nc'
    
    # Pathway for uploading the hand trajectories.
    path_handtrajectories = dir_handtrajectories + date + '/Curtailed/' + filename_handtrajectories
    
    # Loading the xarray with the hand trajectories.
    hand_trajectories = xr.open_dataarray(path_handtrajectories)
    hand_trajectories.load()

    # PRINTING
    print('HAND TRAJECTORIES ARRAY')
    print(hand_trajectories)
    
    # Printing all the hand landmarks.
    print('\nHAND LANDMARKS LIST:')
    pprint(list(hand_trajectories.landmarks.values))

    return hand_trajectories





def plotting_landmarks_and_clicks(click_info, hand_trajectories, landmark_trajectories_plotting):
    """
    DESCRIPTION:
    Plotting the experimenter-specified hand landmarks and the click information across the entirety of the spelling 
    block.
    
    INPUT VARIABLES:
    click_info:                     [dict (key: string ('backspace','keyboard','stimcolumn'); Values: below)];
        data:                       [xarray (time samples,) > strings];  For each time sample of the array of each key
                                    there is a 'no_click' string or a click-string specific to that xarray. For example,
                                    the backspace' key of the dictionary has an array where each element is a string 
                                    named either 'no_click' or 'backspace_click'. The 'backspace_click' elements do not
                                    occur consecutively and describe the instance a click on the backspace key occured. 
                                    For the 'keyboard' and 'stimcolumn' keys, similar rules apply. Time dimension is in
                                    units of s.
        plotcolor:                  [string]; Color corresponding to the type of click for plotting.
    hand_trajectories_ref:          [xarray (landmarks x time samples) > floats]; The trajectories of the x- and y-
                                    coordinates for each landmark. These are referenced in the x- and y-dimensions 
                                    according to the reference landmarks. The time dimension is in units of seconds. 
    landmark_trajectories_plotting: [list > strings]; Possible landmarks to display.
    """
    
    # PLOTTING

    # Initializing the figure.
    fig, ax = plt.subplots(2,1, figsize = (20,7.5))

    
    # Plotting Landmark Trajectories.

    # Iterating across each landmark trajectory that will be plotted.
    for this_landmark in landmark_trajectories_plotting:

        # Extracting the trajectory for the current landmark.
        this_landmark_trajectory = hand_trajectories.loc[this_landmark]

        # Extracting the time array from the landmark trajectory data.
        t_seconds = this_landmark_trajectory['time_seconds'].values

        # Plotting the current landmark trajectory.
        ax[0].plot(t_seconds, this_landmark_trajectory)


    # Plotting Click Information.

    # Iterating across each type of click array recorded onthe participant's monitor.
    for this_click_type in click_info.keys():

        # Extracting the subdictionary for the current type of click information.
        this_click_type_dict = click_info[this_click_type]

        # Extracting the data and corresponding color (for plotting) of the current click information.
        this_click_data  = this_click_type_dict['data']
        this_click_color = this_click_type_dict['plotcolor']

        # Extracting time array from the click information.
        t_seconds = this_click_data['time_seconds'].values

        # Plotting the current click data.
        ax[1].plot(t_seconds, this_click_data, color=this_click_color)

        
    # Extra plot info.
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Landmark Amplitude')
    ax[0].grid()
    ax[1].set_xlabel('Time (s)')
    ax[1].grid()

    return None        




def referencing_hand_trajectories(hand_trajectories, ref1_x, ref2_x, refa_y, refb_y):
    """
    DESCRIPTION:
    Each hand landmark is referenced according to experimenter-specified landmarks. Make sure that the landmarks that 
    are selected will not be used for further analysis as they will get normalized out to 0.
    
    INPUT VARIABLES:
    hand_trajectories: [xarray (landmarks, time samples) > floats]; The time traces of the x- and y-coordinates for each 
                       hand landmark. The time domain is in units of seconds. 
    ref1_x:            [string]; First horizontal reference landmark
    ref2_x:            [string]; Second horizontal reference landmark
    refa_y:            [string]; First vertical reference landmark
    refb_y:            [string]; Second vertical reference landmark
    
    OUTPUT VARIABLES:
    hand_trajectories_ref: [xarray (landmarks, time samples) > floats]; The trajectories of the x- and y-coordinates for 
                           each hand landmark. These are referenced in the x- and y-dimensions according to the 
                           reference landmarks. The time domain is in units of seconds. 
    """
    
    # COMPUTATION:
    
    # Initializing the xarray that holds the normalized hand trajectories. Just deep-copying the un-normalized version.
    hand_trajectories_norm = copy.deepcopy(hand_trajectories)
    
    # Extracting the hand landmark trajectories of the reference landmarks.
    ref1_x_trajectory = hand_trajectories.loc[ref1_x].values
    ref2_x_trajectory = hand_trajectories.loc[ref2_x].values
    refa_y_trajectory = hand_trajectories.loc[refa_y].values
    refb_y_trajectory = hand_trajectories.loc[refb_y].values
    
    # Computing the element-wise distances for the x- and y-references.
    diff_ref_x_trajectory = ref2_x_trajectory - ref1_x_trajectory
    diff_ref_y_trajectory = refa_y_trajectory - refb_y_trajectory
    
    # Extracting the list of hand landmarks.
    hand_landmarks = list(hand_trajectories.landmarks.values)
    
    # Iterating across all hand landmarks.
    for this_landmark in hand_landmarks:

        # Extracting the data for the current hand landmark.
        this_landmark_trajectory = hand_trajectories.loc[this_landmark].values

        # If the landmark has an 'x' coordinate, reference to ref1 and ref2. 
        if 'x' in this_landmark:

            # Computing the normalized difference of the current landmark trajectory to ref1.
            diff_landmark_to_ref1 = this_landmark_trajectory - ref1_x_trajectory

            # Computing the percentage of the x-distance the current landmark is between ref2 and ref1.
            this_landmark_ref = np.divide(diff_landmark_to_ref1, diff_ref_x_trajectory)

        # If the landmark has an 'y' coordinate, reference to refa and refb.  
        if 'y' in this_landmark:

            # Computing the normalized difference of the current landmark trajectory to the refb.
            diff_landmark_to_thumb_tip = this_landmark_trajectory - refb_y_trajectory

            # Computing the percentage of the y-distance the current landmark is between refb and refa.
            this_landmark_ref = np.divide(diff_landmark_to_thumb_tip, diff_ref_y_trajectory)

        # Assigning the normalized landmark trajectory to the xarray.
        hand_trajectories_norm.loc[this_landmark] = this_landmark_ref
    
    return hand_trajectories_norm





def saving_click_latencies(block_id, date, dir_save_latencies, folder_bci2k_or_ui, movement, t_click_latencies):
    """
    DESCRIPTION:
    Saving click latencies for this block to array of latencies from all other blocks recorded on the current 
    date.

    INPUT VARIABLES:
    block_id:           [String]; Block ID of the task that was run. Should be format 'Block#'.
    date:               [string (YYYY_MM_DD)]; Date on which the current block was run.
    dir_save_latencies: [string]; Directory where the click latencies will be saved.
    movement:           [string]; The movement from which the onsets and offsets will be extracted.
    t_click_latencies:  [array > floats (units: s)]; Latencies of all detected clicks relative to corresponding
                        movement onset.
    """

    # COMPUTATION:

    # Creating the pathway where the click latencies are stored.    
    # this_directory       = dir_save_latencies
    this_directory       = dir_save_latencies + date + '/' + folder_bci2k_or_ui + '/'
    this_filename        = date + '_' + 'click_latencies'
    path_click_latencies = this_directory + this_filename
    
    # Pathway exists (yes or no?): Checking to ensure that the click latencies dictionary exists.
    pathway_exists = os.path.exists(path_click_latencies)
    
    # The dictionary exists in the specified pathway.
    if pathway_exists:

        # Read in the dictionary from the pathway.
        with open(path_click_latencies, "rb") as fp:   
            dict_click_latencies = pickle.load(fp)

    # The dictionary does not exist in the specified pathway.
    if not pathway_exists:

        # If the directory does not exist, make it.
        directory_exists = os.path.exists(this_directory)
        if not directory_exists:
            os.mkdir(this_directory)

        # Initialize the dictionary with the click latencies from the current block.
        dict_click_latencies = collections.defaultdict(dict)

    # Updating the dictionary with the click latencies from the current block.
    dict_click_latencies[movement][block_id.lower()] = t_click_latencies
    

    # Saving the updated dictionary.
    with open(path_click_latencies, "wb") as fp: pickle.dump(dict_click_latencies, fp)

    return None





def unique_value_index_finder(my_array):
    """
    DESCRIPTION: 
    This function finds useful indexing information about multi-step step functions.

    INPUT VARIABLES:
    my_array: [list]; List of a step signal with multiple steps of various heights
    
    OUTPUT VARIABLES:
    unique_vals:     [list] Individual values (heights) of all the steps in the vector. For example, the UniqueValues of 
                     V = [1,1,1,1,2,2,2,2,1,1,1,1,3,3,3,3,1,1,1,1] would be [1,2,3]
    n_steps_per_val: [list > ints] The number of occurances of a step of a certein height. For example, the NumberStepsPerValue of V would be
                     [3,1,1] because tep size 1 occurs 3 times, whereas step sizes 2 and 3 occurs only once.
    unique_val_inds: [list > ints] All the indices within the vector of where a step of a certain height occur. For example, UniqueValueIndices of
                     V would be [[0,1,2,3,8,9,10,11,16,17,18,19],[4,5,6,7],[12,13,14,15]]
    start_end_inds:  [list > ints] The indices within the vector where the steps of different heights start and stop. For example, start_end_inds of 
                     V would be [[[0,3],[8,11],[16,19]],[[4,7]],[[12,15]]]
    """
        
    # COMPUTATION:
    # Find the unique sorted elements of my_array.
    unique_vals = np.unique(my_array)
        
    # Initiate the list of indices.
    unique_val_inds = []
    start_end_inds  = []
    n_steps_per_val = []
    
    # Iterate across each value in unique_vals.
    for this_val in unique_vals:
        
        this_val_inds = [i for i, x in enumerate(my_array) if x == this_val]
        inds          = []
        these_inds    = []
        
        for n in range(len(this_val_inds)):
            if n == 0:
                these_inds.extend([this_val_inds[n]])
            if n > 0:
                if this_val_inds[n] - this_val_inds[(n - 1)] > 1:
                    these_inds.extend([this_val_inds[(n - 1)]])
                    inds.append(these_inds)
                    these_inds = [this_val_inds[n]]
            if n == len(this_val_inds) - 1:
                these_inds.extend([this_val_inds[n]])
                inds.append(these_inds)

        unique_val_inds.append(this_val_inds)
        start_end_inds.append(inds)
        n_steps_per_val.extend([len(inds)])

    unique_vals = list(unique_vals)
    
    return unique_vals, n_steps_per_val, unique_val_inds, start_end_inds





def upload_movement_onsetsoffsets(block_id, date, dir_onsetsoffsets):
    """
    DESCRIPTION:
    The dictionary containing the movement onset and offset times for each movement type will be uploaded if it exists. 
    This dictionary would contain the previously saved movement onset/offset times for each movement.
    
    INPUT VARIABLES:
    block_id:          [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:              [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_onsetsoffsets: [string]; Directory where the dictionaries of movement onsets and offset times are stored.
    patient_id:        [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:              [string]; Type of task that was run.

    OUTPUT VARIABLES:
    movement_onsetsoffsets: [dictionary (key: string (movement); Value: list > list [t_onset, t_offset] > floats)]; The 
                            dictionary containing all movement onset and offset times for each movement type.
    """
    
    # COMPUTATION:
    
    # Creating the filename to the movement onsets/offsets dictionary.
    filename_onsetsoffsets = 'dict_OnsetOffset_' + block_id
    
    # Creating the pathway to the movement onset/offset dictionary.
    path_onsetsoffsets_dict = dir_onsetsoffsets + date + '/' + filename_onsetsoffsets
    
    # Checking to make sure the pathway exists.
    pathway_exists = os.path.exists(path_onsetsoffsets_dict)
        
    # The onsets/offsets dictionary exists in the specified pathway.
    if pathway_exists:
        
        # Read in the dictionary from the pathway.
        with open(path_onsetsoffsets_dict, "rb") as fp:
            movement_onsetsoffsets = pickle.load(fp)
            
        # Print the dictionary.
        pprint(movement_onsetsoffsets)
    
    else:
        print('Dictionary of movement onsets/offsets does not exist.')
    
    return movement_onsetsoffsets



    
    
def zooming_in(click_info, hand_trajectories_relevant, movement_colors, movement_onsetsoffsets, t_end_zoom, \
               t_start_zoom):
    """
    DESCRIPTION:
    The experimenter inputs a start and an end time between which to zoom in to view the relevant hand trajectories for
    each movement and click information. The hand landmark trajectories are shown for each movement in a separate plot
    and should be used to inform determining the movement onset and offset times. If there already exists in the 
    movement onset/offset times dictionary onset and offset times within the zoomed-in region for a particular movement,
    these will also be displayed as well as their numerical cardinality (as a list).
    
    INPUT VARIABLES:
    click_info:             [dict (key: string ('backspace','keyboard','stimcolumn'); Values: below)];
        data:               [xarray (time samples,) > strings];  For each time sample of the array of each key there is
                            a 'no_click' string or a click-string specific to that xarray. For example, the 'backspace'
                            key of the dictionary has an array where each element is a string named either 'no_click' or 
                            'backspace_click'. The 'backspace_click' elements do not occur  consecutively and describe
                            the instance a click on the backspace key occured. For the 'keyboard' and 'stimcolumn' keys,
                            similar rules apply. Time dimension is in units of s.
        plotcolor:          [string]; Color corresponding to the type of click for plotting.
    hand_trajectories_ref:  [xarray (landmarks x time samples) > floats]; The trajectories of the x- and y-coordinates
                            for each landmark. These are referenced in the x- and y-dimensions according to the 
                            reference landmarks. The time dimension is in units of seconds. 
    movement_colors:        [dictionary (key: string (movement); Value: string (color))]; There is a color associated 
                            with each movement for plotting.
    movement_onsetsoffsets: [dictionary (key: string (movement); Value: list > list [t_onset, t_offset] > floats)]; The 
                            dictionary containing all movement onset and offset times for each movement type.
    t_end_zoom:             [float (units: s)]; The ending time point for the zoomed in window. To set as the last time
                            point, leave as empty list [].
    t_start_zoom:           [float (units: s)]; The starting time point for the zoomed in window. To set as the first 
                            time point, leave as empty list [].    
    """
    
    # COMPUTATION:
    
    # Extracting the time array from the first key of the relevant hand trajectories dictikonary. This is the same time
    # array that is used for the click information.
    t_seconds = hand_trajectories_relevant[list(hand_trajectories_relevant.keys())[0]].time_seconds.values

    # Defaulting to the first and last time points if the experimenter-specified starting and ending time points are 
    # left empty ( [] ).
    if not t_start_zoom:
        t_start_zoom = t_seconds[0]
    if not t_end_zoom:
        t_end_zoom = t_seconds[-1]

    # Computing the zoomed-in boolean indices for the hand trajectories and click information.
    zoom_bool = np.logical_and(t_seconds > t_start_zoom, t_seconds < t_end_zoom)

    # Extracting only the zoomed-in time interval.
    t_seconds_zoom = t_seconds[zoom_bool]

    # Iterating across all movements in the relevant hand trajectories dictionary.
    for this_movement in hand_trajectories_relevant.keys():

        # Extracting the relevent hand trajectories for the current movement and zoomed-in interval. Transposing is 
        # necessary for plotting.
        these_hand_trajectories_zoomed = hand_trajectories_relevant[this_movement][:,zoom_bool].transpose()

        # Extracting the onset and offsets times for the current movement.
        this_movement_onsets_offset_times = movement_onsetsoffsets[this_movement]

        # Initializing lists of movement onset and offset times as well as correspnding indices that fall within the
        # zoomed-in period.
        move_onset_times  = []
        move_offset_times = []
        move_onset_inds   = []
        move_offset_inds  = []

        # Iterating across the list of movement onset/offset time pairs.
        for this_onset_time, this_offset_time in this_movement_onsets_offset_times:

            # Determining whether the current onset time and offset time fall within the zoom period.
            onset_in_zoom  = (this_onset_time > t_start_zoom) and (this_onset_time < t_end_zoom)
            offset_in_zoom = (this_offset_time > t_start_zoom) and (this_offset_time < t_end_zoom)

            # The current movement onset time falls within the zoomed-in period.
            if onset_in_zoom:

                # Add the current movement onset time to the list of movement onsets within the zoomed-in period.
                move_onset_times.append(this_onset_time)

                # Find the index location for the current movement onset time.
                this_onset_location = ["{} {}".format(ind1,ind2) for (ind1,val1) in \
                                       enumerate(this_movement_onsets_offset_times) for (ind2,val2) in enumerate(val1)\
                                       if val2 == this_onset_time]
                this_onset_location = this_onset_location[0].split()

                # Add the index of the current movement onset time to the list of movement onsets indices within the 
                # zoomed-in period.
                move_onset_inds.append(int(this_onset_location[0]))

            # The current movement offset time falls within the zoomed-in period.
            if offset_in_zoom:

                # Add the current movement offset time to the list of movement offsets within the zoomed-in period.
                move_offset_times.append(this_offset_time)

                # Find the index location for the current movement offset time.
                this_offset_location = ["{} {}".format(ind1,ind2) for (ind1,val1) in \
                                        enumerate(this_movement_onsets_offset_times) for (ind2,val2) in enumerate(val1)\
                                        if val2 == this_offset_time]
                this_offset_location = this_offset_location[0].split()

                # Add the index of the current movement offset time to the list of movement offsets indices within the
                # zoomed-in period.
                move_offset_inds.append(int(this_offset_location[0]))

        # Printing movement times that fall in the zoomed-in region as well as the cardinality of the onset/offset pairs
        # within the zoomed-in region.
        print('Movement: ', this_movement)
        print('\nMovement Onset Times: ', move_onset_times)
        print('Movement Onset Inds: ', move_onset_inds)
        print('\nMovement Offset Times: ', move_offset_times)
        print('Movement Offset Inds: ', move_offset_inds)

        # For the current movement, initializing a plot the movement onsets and offsets as well as the click information.
        fig, ax1 = plt.subplots(figsize=(20,2.5))
        ax2 = ax1.twinx()

        # Defining the color for the current movement.
        this_movement_color = movement_colors[this_movement]

        # Plotting the zoomed-in hand trajectories
        ax1.plot(t_seconds_zoom, these_hand_trajectories_zoomed, color = this_movement_color, linewidth = 2)

        # Plotting the zoomed-in click information by iterating across each click type.
        for this_click_type in click_info.keys():

            # Extracting the current click type and zoomed-in interval.
            this_click_info_zoomed = click_info[this_click_type]['data'][zoom_bool]

            # Extracting the color of the current click information.
            this_click_color = click_info[this_click_type]['plotcolor']

            # Plotting the current click information in the zoomed-in period.
            ax2.plot(t_seconds_zoom, this_click_info_zoomed, color = this_click_color)

        # Plotting the movement onset and offset times as stem plots.
        if move_onset_times:
            stems_onsets = np.ones((len(move_onset_times),))
            ax2.stem(move_onset_times, stems_onsets, basefmt = 'black', linefmt = 'gray')
        if move_offset_times:
            stems_offsets = np.ones((len(move_offset_times),))
            ax2.stem(move_offset_times, stems_offsets, basefmt = 'black', linefmt = 'gray');


        ax1.set_xlabel("Time (s)", fontsize = 14)
        ax1.set_ylabel("Amplitude", fontsize = 14, color = this_movement_color)
        ax1.tick_params(axis = "y", colors = this_movement_color)
        ax1.set_title('Grasp Traces')
        ax1.grid()
        ax2.set_ylabel("Click Type",fontsize = 14, color = "black")
        ax2.tick_params(axis="y", colors = "black")
    
    return None