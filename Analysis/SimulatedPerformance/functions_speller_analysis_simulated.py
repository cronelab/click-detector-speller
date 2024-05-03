
import collections
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil
import xarray as xr

from pprint import pprint





def computing_fp(movement, movement_onsetsoffsets, t_after_movement_limit, t_click_onsets):
    """
    DESCRIPTION:
    Counting the number of clicks that occur without any corresponding movement (outside the click limit window).

    INPUT VARIABLES:
    movement:               [string]; The movement from which the onsets and offsets will be extracted.
    movement_onsetsoffsets: [dictionary (key: string (movement); Value: list > list [t_onset, t_offset] > floats)]; The 
                            dictionary containing all movement onset and offset times for each movement type.
    t_after_movement_limit: [float (unit: s)]; Amount of time from movement onset that can pass for a click to occur 
                            and be associated with the movement onset.
    t_click_onsets:         [array > floats (units: s)]; The times the click changes from 0 to 1. 

    OUTPUT VARIABLES:
    n_fp: [int]; Total number of false positives for the current block of the session.
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
        
        # Computing the time limit in which a click onset could be produced such that it counts as due to the current movement onset.
        t_movement_limit = this_click_onset - t_after_movement_limit
        
        # print(t_movement_limit, this_click_onset)
        
        # Extracting the time(s) which falls within the period between the movement onset period and time command limit.
        bool_movements   = np.logical_and(movement_onsets >= t_movement_limit, movement_onsets <= this_click_onset)
        t_bool_movements = movement_onsets[bool_movements]
        
        # If there were no accompanying movements associated with the command change, then that is a false positive.
        if not t_bool_movements.any():
            print(round(this_click_onset, 2))
            n_fp += 1
              
    return n_fp





def computing_latency_and_tp(movement, movement_onsetsoffsets, patient_id, t_after_movement_limit, t_click_onsets):
    """
    DESCRIPTION:
    Computing the latency from movement onset to click and the resulting sensitivity. The sensitivity is defined as the
    number of true positives over the number of all click detectiosn, where a true positive must occur within the 
    experimenter-defined time post-movement onset.
    
    INPUT VARIABLES:
    movement:               [string]; The movement from which the onsets and offsets will be extracted.
    movement_onsetsoffsets: [dictionary (key: string (movement); Value: list > list [t_onset, t_offset] > floats)]; The 
                            dictionary containing all movement onset and offset times for each movement type.
    patient_id:             [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    t_after_movement_limit: [float (unit: s)]; Amount of time from movement onset that can pass for a click to occur and
                            be associated with the movement onset.
    t_click_onsets:         [array > floats (units: s)]; The times the click changes from 0 to 1.
    
    OUTPUT VARIABLES:
    n_clicks:          [int]; Number of detected clicks.
    t_click_latencies: [array > floats (units: s)]; Latencies of all detected clicks relative to corresponding
                       movement onset.
    """

    # COMPUTATION:

    # Extracting the list of movement onsets and offsets.
    these_onset_offset_pairs = movement_onsetsoffsets[movement]
    
    # Extracting the movement onsets.
    movement_onsets = np.array([this_onset for this_onset, _ in these_onset_offset_pairs])
    
    # Defining a list which will hold all of the latencies between movement onset and click onset, given that a click 
    # occurs within the experimenter-defiend movement limit.
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

        # If there is no click in the click time limit.
        else:
            print('Miss: ', round(this_movement_onset, 3))
            
    # Converting the click latencies into an array and saving them to a dictionary.
    t_click_latencies = np.asarray(t_click_latencies)
    
    # Computing the mean and standard deviation of the latencies.  
    mean_latency  = round(np.mean(t_click_latencies), 3)
    stdev_latency = round(np.std(t_click_latencies), 3)
 
    # Computing the sensitivity.
    n_clicks    = t_click_latencies.shape[0]
    n_attempts  = movement_onsets.shape[0]
    sensitivity = round((n_clicks/n_attempts)*100, 2)
    
    # PRINTING
    print('N grasps: ', n_attempts)
    print('N clicks: ', n_clicks)
    print('Sensitivity: ', sensitivity)
    print('Mean Latency to Command: ', mean_latency)
    print('Stdev Latency to Command: ', stdev_latency)
    
    # PLOTTING
    
    # Plotting the histogram
    fig = plt.figure()
    plt.hist(t_click_latencies, bins = 20)
    plt.title('Latency to Click Detection')
    plt.xlabel('Time (s)')
    plt.ylabel('Trials')
    plt.grid()
    
    return n_clicks, t_click_latencies
    
    
    


def extracting_click_onset_offset_times(click_trace):
    """
    DESCRIPTION:
    Producing the arrays of command onsets and offsets.
    
    INPUT VARIABLES:
    click_trace: [xarray (time samples,)> strings]; At the video resolution, there exists a click or no-click entry. 
                 Time dimension is in units of seconds at video resolution.
    
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
    
    print(unique_vals)
    
    # Using the 'click' index in the unqiue_vals list to extract the starting and ending times of the click periods.
    click_index       = unique_vals.index('click')    
    onset_offset_inds = start_end_inds[click_index]
        
    # Computing the number of command changes (whether its BCI2000 state changes or NAVI clicks).
    n_click_changes = len(onset_offset_inds)
    
    # Initializing an array of click onset/offset times for each click period.
    t_onsets  = np.zeros((n_click_changes,)) 
    t_offsets = np.zeros((n_click_changes,)) 
    
    # Iterating across all pairs of state indices.
    for n, (onset_ind, offset_ind) in enumerate(onset_offset_inds):
        
        # Extracting the output onset and offset times using the respective indices.
        t_onsets[n]  = t_seconds[onset_ind]
        t_offsets[n] = t_seconds[offset_ind]
    
    return t_onsets, t_offsets





def extracting_relevant_trajectories(hand_trajectories_ref, relevant_hand_landmarks):
    """
    DESCRIPTION:
    For each movement type, the experimenter enters the most relevant hand landmarks for visualization. The experimenter
    creates a relevant_hand_landmarks dictionary where the keys of the dictionary are the possible movement classes and
    the value for each key is a list of the most relevant hand landmarks to that class. 

    INPUT VARIABLES:
    hand_trajectories_ref: [xarray (landmarks, time samples) > floats]; The trajectories of the x- and y- coordinates 
                           for each landmark. These are referenced in the x- and y-dimensions according to the reference 
                           landmarks. The time domain is in units of seconds. 
    relevant_landmarks:    [dictionary (key: string (movement type); Value: list > strings (hand landmarks))]; Each 
                           movement holds a list of the most useful landmarks used to detect the corresponding movement 
                           type.
    
    OUTPUT VARIABLES:
    hand_trajectories_rel: [dictionary (Key: string (movement type); Value: xarray (relevant landmarks x time samples)
                           > floats]; For each movement type, only the relevant hand trajectories are stored. The time
                           dimension of each xarray is in units of s.
    """
    
    # COMPUTATION:
    
    # Initializing the dictionary of relevant hand trajectories per movement.
    hand_trajectories_rel = {}

    # Iterating across all movement types:
    for this_movement in relevant_hand_landmarks.keys():

        # Extracting the relevant landmarks for the current movement.
        this_movement_relevant_landmarks = relevant_hand_landmarks[this_movement]

        # Extracting only the trajectories of the relevant landmarks for the current movement.
        this_movement_hand_trajectories = hand_trajectories_ref.loc[this_movement_relevant_landmarks,:]

        # Assigning the hand trajectorires specific to this movement to the dictionary.
        hand_trajectories_rel[this_movement] = this_movement_hand_trajectories
    
    return hand_trajectories_rel





def load_click_information(block_id, date, dir_intermediates, n_votes, n_votes_thr, patient_id, task):
    """
    DESCRIPTION:
    Loading the simulated click trace. Note that the simulation does not make a distinction between whether a row click, 
    column click, or backspace click occurred, because there does not exist a corresponding video from which this 
    distinction could be make.

    Feel free to modify the pathway in which this simulated click information is stored and modify the necessary 
    experimenter inputs appropriately.

    INPUT VARIABLES:
    block_id:          [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:              [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_intermediates: [string]; Intermediates directory where relevant information is stored.
    n_votes:           [int]; Number of most recent classifications to consider when voting on whether a click should or 
                       should not be issued in real-time decoding simulation. For N number of votes, the voting window 
                       corresponds to N*packet size (unitless * ms) worth of data. For example, 7 votes with a packet
                       size of 100 ms corresponds to 700 ms of worth of data being considered in the voting window.
    n_votes_thr:       [int]; Number of grasp votes which must accumulate within the most recent n_votes classifications
                       to issue a click command.
    patient_id:        [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:              [string]; Type of task that was run.
    
    OUTPUT VARIABLES:
    click_trace: [xarray (time samples,)> strings]; At the video resolution, there exists a click or no-click entry. 
                 Time dimension is in units of seconds at video resolution.
    """
    
    # COMPUTATION:
    
    # Creating the pathway for extracting the hand traces.
    this_directory   = dir_intermediates + patient_id + '/' + task + '/ClickDetections/Simulated/' + date + '/' +\
                       block_id + '/' + str(n_votes) +'_vote_window_' + str(n_votes_thr) + '_vote_thr' + '/'
    this_filename    = date + '_' + block_id + '_click_highlights_video.nc'
    path_click_trace = this_directory + this_filename
    
    # Loading the click highlights xarray.
    click_trace = xr.open_dataarray(path_click_trace)
    click_trace.load()
    
    # PRINTING
    print('\nCLICK TRACE:')
    print(click_trace)

    return click_trace





def load_hand_trajectories(block_id, date, dir_intermediates, patient_id, task):
    """
    DESCRIPTION:
    Importing the xarray of hand trajectories. Note that these hand trajectories are curtailed between the 
    block start and stop times.
    
    INPUT VARIABLES:
    block_id:          [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:              [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_intermediates: [string]; [string]; Intermediates directory where relevant information is stored.
    patient_id:        [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:              [string]; Type of task that was run.
    
    OUTPUT VARIABLES:
    hand_trajectories: [xarray (landmarks, time samples) > floats]; The time traces of the x- and y-coordinates for each 
                       landmark. The time domain is in units of seconds. 
    """
    
    # COMPUTATION:
    
    # Creating the directory and filename for the hand trajectories.
    dir_handtrajectories      = dir_intermediates + patient_id + '/' + task + '/HandTrajectories/'  + date +\
                                '/Curtailed/'
    filename_handtrajectories = date + '_' + block_id + '_hand_trajectories.nc'
    
    # Pathway for uploading the hand trajectories.
    path_handtrajectories = dir_handtrajectories + filename_handtrajectories
    
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





def referencing_hand_trajectories(hand_trajectories, ref1_x, ref2_x, refa_y, refb_y):
    """
    DESCRIPTION:
    Each hand landmark is referenced according to experimenter-specified landmarks. Make sure that the landmarks that
    are selected will not be used for further analysis as they will get normalized out to 0.
    
    INPUT VARIABLES:
    hand_trajectories: [xarray (landmarks, time samples) > floats]; The time traces of the x- and y-coordinates for each 
                       landmark. The time domain is in units of seconds. 
    ref1_x:            [string]; First horizontal reference landmark
    ref2_x:            [string]; Second horizontal reference landmark
    refa_y:            [string]; First vertical reference landmark
    refb_y:            [string]; Second vertical reference landmark
    
    OUTPUT VARIABLES:
    hand_trajectories_ref: [xarray (landmarks, time samples) > floats]; The trajectories of the x- and y- coordinates
                           for each landmark. These are referenced in the x- and y-dimensions according to the reference 
                           landmarks. The time domain is in units of seconds. 
    """
    
    # COMPUTATION:
    
    # Initializing the xarray that holds the normalized hand trajectories. Just deep-copying the un-normalized version.
    hand_trajectories_norm = copy.deepcopy(hand_trajectories)
    
    # Extracting the hand landmark trajectories of the reference landmarks.
    ref1_x_trajectory = hand_trajectories.loc[ref1_x].values
    ref2_x_trajectory = hand_trajectories.loc[ref2_x].values
    refa_y_trajectory = hand_trajectories.loc[refa_y].values
    refb_y_trajectory = hand_trajectories.loc[refb_y].values
    
    # Computing the element-wise distances for the x- and y- references.
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





def saving_click_latencies(block_id, date, dir_intermediates, movement, patient_id, t_click_latencies, task):
    """
    DESCRIPTION:
    Saving click latencies for this block to array of latencies from all other blocks recorded on the current date. This
    is for computing across-block error for average mean latencies. 

    INPUT VARIABLES:
    block_id:          [String]; Block ID of the task that was run. Should be format 'Block#'.
    date:              [string (YYYY_MM_DD)]; Date on which the current block was run.
    dir_intermediates: [string]; Intermediates directory where relevant information is stored.
    movement:          [string]; The movement from which the onsets and offsets will be extracted.
    patient_id:        [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    t_click_latencies: [array > floats (units: s)]; Latencies of all detected clicks relative to corresponding movement 
                       onset.
    task:              [string]; Type of task that was run.
    """

    # COMPUTATION:

    # Creating the pathway where the click latencies are stored.
    this_directory       = dir_intermediates + patient_id + '/' + task + '/ClickLatencies/Simulated/' + date + '/' 
    this_filename        = date + '_' + block_id + '_click_latencies.txt'
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
    dir_intermediates: [string]; Intermediates directory where relevant information is stored.
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
        print('Dictionary of movement onsets and offsets does not exist.')
    
    return movement_onsetsoffsets





def unique_value_index_finder(my_vector):
    """
    DESCRIPTION: 
    This function finds useful indexing information about multi-step step functions.

    INPUT VARIABLES:
    my_vector: [list]; List of a step signal with multiple steps of various heights
    
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
    # Find the unique sorted elements of my_vector.
    unique_vals = np.unique(my_vector)
        
    # Initiate the list of indices.
    unique_val_inds = []
    start_end_inds  = []
    n_steps_per_val = []
    
    # Iterate across each value in unique_vals.
    for this_val in unique_vals:
        
        this_val_inds = [i for i, x in enumerate(my_vector) if x == this_val]
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





def zooming_in(click_trace, hand_trajectories_rel, movement, movement_onsetsoffsets, t_end_view, t_start_view):
    """
    DESCRIPTION:
    Zooming in on the experimenter-specified time period of the hand traces and click trace.
    
    INPUT VARIABLES:
    click_trace:            [xarray (time samples,)> strings]; At the video resolution, there exists a click or no-click 
                            entry. Time dimension is in units of seconds at video resolution.
    hand_trajectories_rel:  [dictionary (Key: string (movement type); Value: xarray (relevant landmarks x time samples)
                            > floats]; For each movement type, only the relevant hand trajectories are stored. The time
                            dimension of each xarray is in units of s.
    movement_onsetsoffsets: [dictionary (key: string (movement); Value: list > list [t_onset, t_offset] > floats)]; The 
                            dictionary containing all movement onset and offset times for each movement type.
    movement:               [string]; The movement from which the onsets and offsets will be extracted.
    t_end_view:             [float (units: s)]; The ending point of the visualization window.
    t_start_view:           [float (units: s)]; The starting point of the visualization window.
    """
    
    # COMPUTATION:
    
    # Extracting the time array from the click trace.
    t_seconds = click_trace.time_seconds
    
    # Extracting the time indices from the experimenter-specified zoom-in period.
    zoom_bool = np.logical_and(t_seconds > t_start_view, t_seconds < t_end_view)
    
    # Extracting the movement onset/offset times for the specified movement.
    this_movement_onsetoffsets = movement_onsetsoffsets[movement]

    # Extracting only the hand trajectories for grasp.
    these_hand_trajectories = hand_trajectories_rel[movement]
    
    # Initializing the list of movement onset/offset times.
    move_onsets  = []
    move_offsets = []
    
    # Iterating across each movement onset/offset times within the current set of movement onset/offset pairs.
    for this_onset, this_offset in this_movement_onsetoffsets:

        # Logical statement as to whether the the current movement onset and offset times fall within the zoom period.
        onset_within_zoom  = (this_onset > t_start_view) and (this_onset < t_end_view)
        offset_within_zoom = (this_offset > t_start_view) and (this_offset < t_end_view)
            
        # The current movement onset falls within the zoomed period.
        if onset_within_zoom:
            
            # Add the current movement onset to the list of movement onsets within this period.
            move_onsets.append(this_onset)
            
        # The current movement offset falls within the zoomed period.
        if offset_within_zoom:
            
            # Add the current movement offset to the list of movement offsets within this period.
            move_offsets.append(this_offset)
    
    # Creating the movement onset/offset stem plot amplitudes.
    stems_onsets  = np.ones((len(move_onsets),))
    stems_offsets = np.ones((len(move_offsets),))

    # PLOTTING
    
    # Plotting click traces
    fig, axs = plt.subplots(2,1, figsize=(20,7.5));
    axs[0].plot(t_seconds[zoom_bool], click_trace[zoom_bool], color = 'black')
    axs[0].set_title('Click');
    axs[0].margins(x=0);
    
    # Plotting hand traces.
    axs[1].plot(t_seconds[zoom_bool], these_hand_trajectories.loc[:,zoom_bool].transpose())
    axs[1].margins(x=0);
    axs[1].set_title('Hand Traces');
    if move_onsets:
        axs[1].stem(move_onsets, stems_onsets, basefmt = 'black', linefmt = 'gray', use_line_collection = True) 
    if move_offsets:
        axs[1].stem(move_offsets, stems_offsets, basefmt = 'black', linefmt = 'gray', use_line_collection = True) 
