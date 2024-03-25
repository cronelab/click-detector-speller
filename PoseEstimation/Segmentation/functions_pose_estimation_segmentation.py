
# Importing packages.
import copy
import numpy as np
import os
import pickle
import shutil
import xarray as xr

import matplotlib.pyplot as plt

from pprint import pprint



################################### SAVE SELF ###################################
def save_script_backup():
    
    """
    Automatically saving this entire script immediately when it's called.
    """
    
    # Creating the original and target file directories.
    original = r'/home/dan/Projects/PseudoOnlineTests_for_RTCoG/Scripts/Recent/PoseEstimation/functions_pose_estimation_segmentation.py'
    target   = r'/mnt/shared/danprocessing/BACKUP/Projects/PseudoOnlineTests_for_RTCoG/Scripts/PoseEstimation/functions_pose_estimation_segmentation.py'

    # Saving.
    shutil.copyfile(original, target);
    
# Immediately saving script.   
save_script_backup()





def add_movement_onsetoffset_pair(block_id, date, dir_intermediates, hand_trajectories_relevant, movement_onsetsoffsets,\
                                  patient_id, task):
    """
    DESCRIPTION:
    The experimenter may add a movement onset/offset time pair. This can be done by looking at the zoomed-in relevant 
    hand landmark trajectories above to determine which movement occurred and the onset and offset times of when it 
    occurred. As such, the experimenter is prompted to enter which movement to add, followed by being prommpted to add
    the movement onset and offset times.
    
    INPUT VARIABLES:
    block_id:                   [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:                       [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_intermediates:          [string]; [string]; Intermediates directory where relevant information is stored.
    hand_trajectories_relevant: [dictionary (Key: string (movement type); Value: xarray (relevant landmarks x time samples)
                                > floats]; For each movement type, only the relevant hand trajectories are stored. The time
                                dimension of each xarray is in units of s.
    movement_onsetsoffsets:     [dictionary (key: string (movement); Value: list > list [t_onset, t_offset] > floats)]; The 
                                dictionary containing all movement onset and offset times for each movement type.
    patient_id:                 [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:                       [string]; Type of task that was run.
    """
    
    # COMPUTATION:
    
    # Extracting the time array from the relevant hand trajectories list. Using the time array from the 
    # first key, but it doesn't matter since all keys share the same time array.
    t_seconds = hand_trajectories_relevant[list(hand_trajectories_relevant.keys())[0]].time_seconds.values

    # Extracting the list movement types from the dictionary.
    movement_types = list(movement_onsetsoffsets.keys())

    # Display all movement types.
    print('Movement types: ', movement_types)

    # Prompt the experimenter to enter for which movement an onset/offset time will be added.
    movement_add = input('For which movement would you like to add onset/offset times?')

    # Initializing the logical flag for the while loop to ensure that the experimenter does not enter
    # an invalid movement type.
    move_flag = True

    # While loop to ensure experimenter does not enter an invalid movement type.
    while move_flag:

        # Only exit the loop if the experimenter entered a valid movement.
        if movement_add in movement_types:
            move_flag = False

        # If the movement type was not entered correctly, prompt the experimenter to enter it again.
        if movement_add not in movement_types:
            movement_add = input('Invalid entry. For which movement would you like to add an onset/offset time pair?')

    # Prompting the experimenter to enter the movement onset/offset times for the selected movement.
    t_onset  = input('Please enter the desired onset time (s):')
    t_offset = input('Please enter the desired offset time (s):')

    # Converting the string inputs to floats.
    t_onset  = float(t_onset)
    t_offset = float(t_offset)

    # Determining the onset and offset times from the time array of the hand movement xarray.
    onset_idx_min  = np.argmin(np.abs(t_seconds - t_onset))
    offset_idx_min = np.argmin(np.abs(t_seconds - t_offset))
    t_onset        = t_seconds[onset_idx_min]
    t_offset       = t_seconds[offset_idx_min]

    # Initializing the logical flag for the while loop to ensure that the experimenter does not enter
    # an onset time greater than the offset time.
    onoff_flag = True

    # While loop to ensure that the experimenter does not enter a movement onset time greater than the
    # movement offset time.
    while onoff_flag:

        # Only exit the loop if the offset time is greater than the onset time.
        if t_offset > t_onset:
            onoff_flag = False

        # If the onset time is greater than the offset time, prompt the experimenter to enter these times again.
        else:
            print('You entered a movement onset time that was greater than the offset time. Please try again.')
            t_onset  = input('Please enter the desired onset time (s):')
            t_offset = input('Please enter the desired offset time (s):')
            t_onset  = float(t_onset)
            t_offset = float(t_offset)

            # Determining the onset and offset times by finding the closest times in the video time array.
            onset_idx_min  = np.argmin(np.abs(t_seconds - t_onset))
            offset_idx_min = np.argmin(np.abs(t_seconds - t_offset))
            t_onset        = t_seconds[onset_idx_min]
            t_offset       = t_seconds[offset_idx_min]

    # Pairing the movement onset and offset times.
    onset_offset_pair = [t_onset, t_offset]

    # Extracting the list of onset and offset times for the selected movement from the dictionary, and
    # appending this list with the new onset/offset pair.
    this_movement_onsetsoffsets = movement_onsetsoffsets[movement_add]
    this_movement_onsetsoffsets.append(onset_offset_pair)

    # Computing the total number of onset/offset pairs.
    n_movement_onsetoffsets = len(this_movement_onsetsoffsets)

    # Extracting only the movement onset times, and sorting them in ascending order.
    onset_times = [onset_time for onset_time, _ in this_movement_onsetsoffsets]
    onset_times.sort()

    # Sorting the movement onset/offset time pairs in increasing order based on the onset times.
    this_movement_onsetoffsets_sorted = [tuple for x in onset_times for tuple in this_movement_onsetsoffsets if tuple[0] == x]

    # Updating the dictionary with the movement onset/offset times for the selected movement with the
    # onset/offset times that were just added.
    movement_onsetsoffsets[movement_add] = this_movement_onsetoffsets_sorted


    # SAVING:

    # Creating the directory and filename to the movement onsets/offsets dictionary.
    dir_onsetsoffsets      = dir_intermediates + patient_id + '/' + task + '/MovementOnsetsAndOffsets/' + date + '/'
    filename_onsetsoffsets = 'dict_OnsetOffset_' + block_id

    # Creating the pathway to the movement onset/offset dictionary.
    path_onsetsoffsets_dict = dir_onsetsoffsets + filename_onsetsoffsets

    # Saving the updated dictionary.
    with open(path_onsetsoffsets_dict, "wb") as fp: pickle.dump(movement_onsetsoffsets, fp)
    
    return None





def delete_movement_onsetoffset_pair(block_id, date, dir_intermediates, movement_onsetsoffsets, patient_id, task):
    """
    DESCRIPTION:
    The experimenter may delete a movement onset/offset time pair. This could be done if, by looking at the 
    zoomed-in relevant hand trajectories above, the experimenter determines that the movement onset/offset pair
    is not in the correct location. The experimenter will be prompted which movement type to delete, and will
    then be prompted to input the index number of that movement within the list onset/offset pairs of that 
    particular movement.
    
    INPUT VARIABLES:
    block_id:               [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:                   [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_intermediates:      [string]; [string]; Intermediates directory where relevant information is stored.
    movement_onsetsoffsets: [dictionary (key: string (movement); Value: list > list [t_onset, t_offset] > floats)]; The 
                            dictionary containing all movement onset and offset times for each movement type.
    patient_id:             [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:                   [string]; Type of task that was run.
    """
    
    # COMPUTATION:
    
    # Extracting the list movement types from the dictionary.
    movement_types = list(movement_onsetsoffsets.keys())

    # Display all movement types.
    print('Movement types: ', movement_types)

    # Prompt the experimenter to enter for which movement an onset/offset time pair will be deleted.
    movement_delete = input('For which movement would you like to delete an onset/offset time pair?')

    # Initializing the logical flag for the while loop to ensure that the experimenter does not enter
    # an invalid movement type.
    move_flag = True

    # While loop to ensure experimenter does not enter an invalid movement type.
    while move_flag:

        # Only exit the loop if the experimenter entered a valid movement.
        if movement_delete in movement_types:
            move_flag = False

        # If the movement type was not entered correctly, prompt the experimenter to enter it again.
        if movement_delete not in movement_types:
            movement_delete = input('Invalid entry. For which movement would you like to delete an onset/offset time pair?')

    # Extracting the movement onset/offset times for the selected movement and computing the total 
    # number of movements.
    this_movement_onsetsoffsets = movement_onsetsoffsets[movement_delete]
    n_onsets_offsets            = len(this_movement_onsetsoffsets)

    # Prompting the experimenter to enter the index of the movement onset/offset pair to be deleted
    # based on the output of the figure in the zoomed-in plot.
    delete_ind = input('Please enter the onset/offset time index within the list of movement indices.')

    # Converting the string input to a int.
    delete_ind = int(delete_ind)

    # Initializing the logical flag for the while loop to ensure that the experimenter does not enter
    # an invalid movement index.
    ind_flag = True

    # While loop to ensure that the experimenter does not enter an index that's outside the total
    # number of indices for the selected movement.
    while ind_flag:

        # If the experimenter entered an index that's larger than the total number of indices for the
        # selected movement, the experimenter must re-enter the value.
        if delete_ind >= n_onsets_offsets:
            delete_ind = input('You entered an index greater than the total number of onset/offset pairs. Please re-enter: ')
            delete_ind = int(delete_ind)

        # Only exit the loop if the index is within the total number of onset/offset indices.
        else:
            ind_flag = False

    # Removing the onset/offset pair corresponding to the experimenter-specified index.
    del this_movement_onsetsoffsets[delete_ind]    


    # Updating the dictionary with the movement onset/offset times for the selected movement without
    # the onset/offset times that were just removed.
    movement_onsetsoffsets[movement_delete] = this_movement_onsetsoffsets


    # SAVING:

    # Creating the directory and filename to the movement onsets/offsets dictionary.
    dir_onsetsoffsets      = dir_intermediates + patient_id + '/' + task + '/MovementOnsetsAndOffsets/' + date + '/'
    filename_onsetsoffsets = 'dict_OnsetOffset_' + block_id

    # Creating the pathway to the movement onset/offset dictionary.
    path_onsetsoffsets_dict = dir_onsetsoffsets + filename_onsetsoffsets

    # Saving the updated dictionary.
    with open(path_onsetsoffsets_dict, "wb") as fp: pickle.dump(movement_onsetsoffsets, fp)

    return None





def extracting_relevant_trajectories(hand_trajectories_ref, relevant_hand_landmarks):
    """
    DESCRIPTION:
    For each movement type, the experimenter enters the most relevant hand landmarks for visualization. The experimenter
    creates a relevant_hand_landmarks dictionary where the keys of the dictionary are the possible movement classes and
    the value for each key is a list of the most relevant hand landmarks to that class. The plotting cells above should
    be used to determine these landmarks. Then for each movement type a dictionary, hand_trajectories_relevant is created
    where for each movement, only the relevant hand trajectories are stored.

    INPUT VARIABLES:
    hand_trajectories_ref:   [xarray (landmarks x time samples) > floats]; The trajectories of the x- and y-coordinates
                             for each landmark. These are referenced in the x- and y-dimensions according to the reference
                             landmarks. The time dimension is in units of seconds. 
    relevant_hand_landmarks: [dictionary (key: string (movement type); Value: list > strings (hand landmarks))]; Each
                             movement holds a list of the most useful landmarks used to detect the corresponding 
                             movement type.
    
    OUTPUT VARIABLES:
    hand_trajectories_relevant: [dictionary (Key: string (movement type); Value: xarray (relevant landmarks x time samples)
                                > floats]; For each movement type, only the relevant hand trajectories are stored. The time
                                dimension of each xarray is in units of s.
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





def load_click_information(block_id, date, dir_intermediates, patient_id, task):
    """
    DESCRIPTION:
    Loading the click information dictionary. Note that the click information is curtailed between the block
    start and stop times.
    
    INPUT VARIABLES:
    block_id:          [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:              [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_intermediates: [string]; [string]; Intermediates directory where relevant information is stored.
    patient_id:        [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:              [string]; Type of task that was run.
    
    OUTPUT VARIABLES:
     click_info: [dict (key: string ('backspace','keyboard','stimcolumn'); Values: below)];
        data:       [xarray (1 x time samples) > strings];  For each  time sample of the array of each key there
                    is a 'no_click' string or a click-string specific to that xarray. For example, the 'backspace'
                    key of the dictionary has an array where each element is a string named either 'no_click' or 
                    'backspace_click'. The 'backspace_click' elements do not occur consecutively and describe the 
                    instance a click on the backspace key occured. For the 'keyboard' and 'stimcolumn' keys, similar
                    rules apply. Time dimension is in units of s.
        plotcolor:  [string]; Color corresponding to the type of click for plotting.
    """
    
    # COMPUTATION:
    
    # Creating the directory and filename for the click information.
    dir_clickinfo      = dir_intermediates + patient_id + '/' + task + '/ClickDetections/'  + date + '/Curtailed/' 
    filename_clickinfo = date + '_' + block_id + '_click_highlights'
    
    # Pathway for the click detections.
    path_clickinfo = dir_clickinfo + filename_clickinfo
    
    # Loading the click information.
    click_info = np.load(path_clickinfo, allow_pickle=True)
    
    return click_info





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
    hand_trajectories: [xarray (landmarks x time samples) > floats]; The time traces of the x- and y-coordinates 
                       for each landmark. The time domain is in units of seconds. 
    """
    
    # COMPUTATION:
    
    # Creating the directory and filename for the hand trajectories.
    dir_handtrajectories      = dir_intermediates + patient_id + '/' + task + '/HandTrajectories/'  + date + '/Curtailed/'
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
    Each hand landmark is referenced according to experimenter-specified landmarks. Make sure that the landmarks that are
    selected will not be used for further analysis as they will get normalized out to 0.
    
    INPUT VARIABLES:
    hand_trajectories: [xarray (landmarks x time samples) > floats]; The time traces of the x- and y-coordinates for each
                       landmark. The time domain is in units of seconds. 
    ref1_x:            [string]; First horizontal reference landmark
    ref2_x:            [string]; Second horizontal reference landmark
    refa_y:            [string]; First vertical reference landmark
    refb_y:            [string]; Second vertical reference landmark
    
    OUTPUT VARIABLES:
    hand_trajectories_ref: [xarray (landmarks x time samples) > floats]; The trajectories of the x- and y-coordinates for each
                           landmark. These are referenced in the x- and y-dimensions according to the reference landmarks. The
                           time domain is in units of seconds. 
    """
    
    # COMPUTATION:
    
    # Initializing the xarray that holds the normalized hand trajectories. Just deep-copying the 
    # un-normalized version.
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





def plotting_landmarks_and_clicks(click_info, hand_trajectories, landmark_trajectories_plotting):
    """
    DESCRIPTION:
    Plotting the experimenter-specified hand landmarks and the click information across the entirety of
    the spelling block.
    
    INPUT VARIABLES:
    click_info:                     [dict (key: string ('backspace','keyboard','stimcolumn'); Values: below)];
        data:                       [xarray (1 x time samples) > strings];  For each  time sample of the array of each key there
                                    is a 'no_click' string or a click-string specific to that xarray. For example, the 'backspace'
                                    key of the dictionary has an array where each element is a string named either 'no_click' or 
                                    'backspace_click'. The 'backspace_click' elements do not occur consecutively and describe the 
                                    instance a click on the backspace key occured. For the 'keyboard' and 'stimcolumn' keys, similar
                                    rules apply. Time dimension is in units of s.
        plotcolor:                  [string]; Color corresponding to the type of click for plotting.
    hand_trajectories:              [xarray (landmarks x time samples) > floats]; The time traces of the x- and y-coordinates 
                                    for each landmark. The time domain is in units of seconds. 
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
        
        # Initialize the dictionary.
        movement_onsetsoffsets = {}
        
        # Iterate across each movement.
        for movement in relevant_hand_landmarks.keys():
            
            # Initializing a dictionary key for each movement, which will store all corresponding movement onset/offset times.
            movement_onsetsoffsets[movement] = []
    
    return movement_onsetsoffsets





def zooming_in(click_info, hand_trajectories_relevant, movement_colors, movement_onsetsoffsets, t_end_zoom, t_start_zoom):
    """
    DESCRIPTION:
    The experimenter inputs a start and an end time between which to zoom in to view the relevant hand trajectories
    for each movement and click information. The hand landmark trajectories are shown for each movement in a 
    separate plot and should be used to inform determining the movement onset and offset times. If there already
    exists in the movement onset/offset times dictionary onset and offset times within the zoomed-in region for a 
    particular movement, these will also be displayed as well as their numerical cardinality (as a list).
    
    INPUT VARIABLES:
    click_info:                 [dict (key: string ('backspace','keyboard','stimcolumn'); Values: below)];
        data:                   [xarray (1 x time samples) > strings];  For each  time sample of the array of each key there
                                is a 'no_click' string or a click-string specific to that xarray. For example, the 'backspace'
                                key of the dictionary has an array where each element is a string named either 'no_click' or 
                                'backspace_click'. The 'backspace_click' elements do not occur consecutively and describe the 
                                instance a click on the backspace key occured. For the 'keyboard' and 'stimcolumn' keys, similar
                                rules apply. Time dimension is in units of s.
        plotcolor:              [string]; Color corresponding to the type of click for plotting.
    hand_trajectories_relevant: [dictionary (Key: string (movement type); Value: xarray (relevant landmarks x time samples)
                                > floats]; For each movement type, only the relevant hand trajectories are stored. The time
                                dimension of each xarray is in units of s.
    movement_colors:            [dictionary (key: string (movement); Value: string (color))]; There is a color associated
                                with each movement for plotting.
    movement_onsetsoffsets:     [dictionary (key: string (movement); Value: list > list [t_onset, t_offset] > floats)]; The 
                                dictionary containing all movement onset and offset times for each movement type.
    t_end_zoom:                 [int (units: s)]; The ending time point for the zoomed in window. To set as the last time
                                point, leave as empty list [].
    t_start_zoom:               [int (units: s)]; The starting time point for the zoomed in window. To set as the first
                                time point, leave as empty list [].    
    """
    
    # COMPUTATION:
    
    # Extracting the time array from the first key of the relevant hand trajectories dictikonary. This is
    # the same time array that is used for the click information.
    t_seconds = hand_trajectories_relevant[list(hand_trajectories_relevant.keys())[0]].time_seconds.values

    # Defaulting to the first and last time points if the experimenter-specified starting and ending time
    # points are left empty ( [] ).
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

        # Extracting the relevent hand trajectories for the current movement and zoomed-in interval.
        # Transposing is necessary for plotting.
        these_hand_trajectories_zoomed = hand_trajectories_relevant[this_movement][:,zoom_bool].transpose()

        # Extracting the onset and offsets times for the current movement.
        this_movement_onsets_offset_times = movement_onsetsoffsets[this_movement]

        # Initializing lists of movement onset and offset times as well as correspnding indices that fall
        # within the zoomed-in period.
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
                this_onset_location = ["{} {}".format(ind1,ind2) for (ind1,val1) in enumerate(this_movement_onsets_offset_times) for (ind2,val2) in enumerate(val1) if val2 == this_onset_time]
                this_onset_location = this_onset_location[0].split()

                # Add the index of the current movement onset time to the list of movement onsets indices within
                # the zoomed-in period.
                move_onset_inds.append(int(this_onset_location[0]))

            # The current movement offset time falls within the zoomed-in period.
            if offset_in_zoom:

                # Add the current movement offset time to the list of movement offsets within the zoomed-in period.
                move_offset_times.append(this_offset_time)

                # Find the index location for the current movement offset time.
                this_offset_location = ["{} {}".format(ind1,ind2) for (ind1,val1) in enumerate(this_movement_onsets_offset_times) for (ind2,val2) in enumerate(val1) if val2 == this_offset_time]
                this_offset_location = this_offset_location[0].split()

                # Add the index of the current movement offset time to the list of movement offsets indices within
                # the zoomed-in period.
                move_offset_inds.append(int(this_offset_location[0]))

        # Printing movement times that fall in the zoomed-in region as well as the cardinality of the onset/offset
        # pairs within the zoomed-in region.
        print('Movement: ', this_movement)
        print('\nMovement Onset Times: ', move_onset_times)
        print('Movement Onset Inds: ', move_onset_inds)
        print('\nMovement Offset Times: ', move_offset_times)
        print('Movement Offset Inds: ', move_offset_inds)

        
        # For the current movement, initializing a plot the movement onsets and offsets as well as the 
        # click information.
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
        # ax1.set_title(this_movement + ' trajectories')
        ax1.set_title('Grasp Traces')
        ax1.grid()
        ax2.set_ylabel("Click Type",fontsize = 14, color = "black")
        ax2.tick_params(axis="y", colors = "black")
    
    return None