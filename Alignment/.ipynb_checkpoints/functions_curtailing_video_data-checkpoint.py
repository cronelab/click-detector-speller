
# IMPORTING LIBRARIES
import copy
import numpy as np
import os
import pickle
import xarray as xr

from pprint import pprint

import matplotlib.pyplot as plt





def curtailing(click_info, hand_trajectories, t_start, t_stop):
    """
    DESCRIPTION:
    Curtailing the hand trajectories and the click information according to the block start and stop times. For reference see 
    computing_block_start_stop_times.ipynb.
    
    INPUT VARIABLES:
    click_info:        [dict (key: string ('backspace','keyboard','stimcolumn'); Values: below)];
        data:          [xarray (1 x time samples) > strings];  For each  time sample of the array of each key there is a 'no_click'
                       string or a click-string specific to that xarray. For example, the 'backspace' key of the dictionary has an
                       array where each element is a string named either 'no_click' or 'backspace_click'. The 'backspace_click' 
                       elements do not occur consecutively and describe the instance a click on the backspace key occured. For the
                       'keyboard' and 'stimcolumn' keys, similar rules apply. Time dimension is in units of s.
        plotcolor:     [string]; Color corresponding to the type of click for plotting.
    hand_trajectories: [xarray (landmarks x time samples) > floats]; The trajectories of the x- and y-coordinates
                       for each landmark. The time domain is in units of seconds. 
    t_start:           [float (units: s)]; True starting time of the block.
    t_stop:            [float (units: s)]; True ending time of the block.
                                
    NECESSARY FUNCTIONS:
    curtailing_click_info
    curtailing_hand_trajectories
    
    OUTPUT VARIABLES:
    click_info_curt:        [dict (key: string ('backspace','keyboard','stimcolumn'); Values: data, plotcolor)]; Same as click_info, but 
                            all xarrays are curtailed within the start and stop times.
    hand_trajectories_curt: [xarray (landmarks x time samples) > floats]; Same as above but curtailed within the start and
                            stop times.                   
    """
    
    # COMPUTATION:
    
    # Curtailing all the click information.
    click_info_curt = curtailing_click_info(click_info, t_start, t_stop)
    
    # Curtailing all the hand trajectories.
    hand_trajectories_curt = curtailing_hand_trajectories(hand_trajectories, t_start, t_stop)
    
    return click_info_curt, hand_trajectories_curt





def curtailing_click_info(click_info, t_start, t_stop):
    """
    DESCRIPTION:
    Curtailing all the data xarrays containing 'no_click' or click strings according to the block start and stop times.
    
    INPUT VARIABLES:
    click_info: [dict (key: string ('backspace','keyboard','stimcolumn'); Values: below)];
        data:      [xarray (1 x time samples) > strings];  For each  time sample of the array of each key there is a 'no_click'
                   string or a click-string specific to that xarray. For example, the 'backspace' key of the dictionary has an
                   array where each element is a string named either 'no_click' or 'backspace_click'. The 'backspace_click' 
                   elements do not occur consecutively and describe the instance a click on the backspace key occured. For the
                   'keyboard' and 'stimcolumn' keys, similar rules apply. Time dimension is in units of s.
        plotcolor: [string]; Color corresponding to the type of click for plotting.
    t_start:    [float (units: s)]; True starting time of the block.
    t_stop:     [float (units: s)]; True ending time of the block.
    
    OUTPUT VARIABLES:
    click_info_curt: [dict (key: string ('backspace','keyboard','stimcolumn'); Values: data, plotcolor)]; Same as click_info, but 
                     all xarrays are curtailed within the start and stop times.
    """
    
    # COMPUTATION:
    
    # Initializing the curtailed click information dictionary.
    click_info_curt = copy.deepcopy(click_info)
        
    # Iterating across all types of clicks
    for this_click_type in click_info.keys():
                
        # Extracting the xarray from the click_info dictionary.
        this_click_type_data = click_info[this_click_type]['data']
                
        # Extracting the time array of the xarray.
        time_seconds = this_click_type_data.time_seconds
        
        # Creating the boolean array of time points between the starting and stopping times.
        curt_bool = np.logical_and(time_seconds >= t_start, time_seconds <= t_stop)
        
        # Curtailing the data xarray according to the boolean array
        this_click_type_data_curt = this_click_type_data[curt_bool]
        
        print(this_click_type_data_curt.time_seconds.values)
        
        # Updating the curtailed click information.
        click_info_curt[this_click_type]['data'] = this_click_type_data_curt
    
    return click_info_curt





def curtailing_hand_trajectories(hand_trajectories, t_start, t_stop):
    """
    DESCRIPTION:
    Curtailing the hand trajectories according to the block start and stop times.
    
    INPUT VARIABLES:
    hand_trajectories: [xarray (landmarks x time samples) > floats]; The trajectories of the x- and y-coordinates
                       for each landmark. The time domain is in units of seconds. 
    t_start:           [float (units: s)]; True starting time of the block.
    t_stop:            [float (units: s)]; True ending time of the block.
                       
    OUTPUT VARIABLES:
    hand_trajectories_curt: [xarray (landmarks x time samples) > floats]; Same as above but curtailed within the start and
                            stop times. 
    """
    
    # COMPUTATION:
    
    # Extracting the time array of the xarray.
    time_seconds = hand_trajectories.time_seconds
    
    print(time_seconds[0])
    
    # Creating the boolean array of time points between the starting and stopping times.
    curt_bool = np.logical_and(time_seconds >= t_start, time_seconds <= t_stop)
    
    # Creating the curtailed hand trajectories xarray.
    hand_trajectories_curt = hand_trajectories[:,curt_bool]
    
    print(hand_trajectories_curt.time_seconds.values)
    
    return hand_trajectories_curt





def load_click_information(block_id, date, dir_intermediates, patient_id, task):
    """
    DESCRIPTION:
    Loading the click information dictionary.
    
    INPUT VARIABLES:
    block_id:          [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:              [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_intermediates: [string]; Intermediates directory where relevant information is stored.
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
    dir_clickinfo      = dir_intermediates + patient_id + '/' + task + '/ClickDetections/'  + date + '/Xarrays/' 
    filename_clickinfo = date + '_' + block_id + '_click_highlights'
    
    # Pathway for the click detections.
    path_clickinfo = dir_clickinfo + filename_clickinfo
    
    # Loading the click information.
    click_info = np.load(path_clickinfo, allow_pickle=True)
    
    return click_info





def load_hand_trajectories(block_id, date, dir_intermediates, patient_id, task):
    """
    DESCRIPTION:
    Importing the xarray of hand trajectories.
    
    INPUT VARIABLES:
    block_id:          [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:              [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_intermediates: [string]; Intermediates directory where relevant information is stored.
    patient_id:        [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:              [string]; Type of task that was run.
    
    OUTPUT VARIABLES:
    hand_trajectories: [xarray (landmarks x time samples) > floats]; The time traces of the x- and y-coordinates 
                       for each landmark. The time domain is in units of seconds. 
    """
    
    # COMPUTATION:
    
    # Creating the directory and filename for the hand trajectories.
    dir_handtrajectories      = dir_intermediates + patient_id + '/' + task + '/HandTrajectories/'  + date + '/Xarrays/'
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





def load_start_stop_times(block_id, date, dir_intermediates, patient_id):
    """
    DESCRIPTION:
    Loading the true starting and stopping times for the current block. 
    
    INPUT VARIABLES:
    block_id:          [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:              [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_intermediates: [string]; Intermediates directory where relevant information is stored.
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





def saving_curtailed_info(block_id, click_info_curt, date, dir_intermediates, hand_trajectories_curt, patient_id, task):
    """
    DESCRIPTION:
    Saving the curtailed hand trajectories and click information.
    
    INPUT VARIABLES:
    block_id:          [String (BlockX, where X is an int))]; Block ID of the task that was run.
    click_info:        [dict (key: string ('backspace','keyboard','stimcolumn'); Values: below)];
        data:          [xarray (1 x time samples) > strings];  For each  time sample of the array of each key there is a 'no_click'
                       string or a click-string specific to that xarray. For example, the 'backspace' key of the dictionary has an
                       array where each element is a string named either 'no_click' or 'backspace_click'. The 'backspace_click' 
                       elements do not occur consecutively and describe the instance a click on the backspace key occured. For the
                       'keyboard' and 'stimcolumn' keys, similar rules apply. All xarrays are curtailed between the startng and 
                       stopping times. Time dimension is in units of s. 
        plotcolor:     [string]; Color corresponding to the type of click for plotting.
    date:              [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_intermediates: [string]; Intermediates directory where relevant information is stored.
    hand_trajectories: [xarray (landmarks x time samples) > floats]; The trajectories of the x- and y-coordinates
                       for each landmark. Curtailed between the starting and stopping times. The time domain is in units of seconds. 
    patient_id:        [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:              [string]; Type of task that was run.
    """
    
    # SAVING:
    
    # Creating the directory and filename for the curtailed click information.
    dir_clickinfo      = dir_intermediates + patient_id + '/' + task + '/ClickDetections/'  + date + '/Curtailed/' 
    filename_clickinfo = date + '_' + block_id + '_click_highlights'

    # Creating the directory and filename for the curtailed hand trajectories.
    dir_handtrajectories      = dir_intermediates + patient_id + '/' + task + '/HandTrajectories/'  + date + '/Curtailed/'
    filename_handtrajectories = date + '_' + block_id + '_hand_trajectories.nc'

    # Pathway for the hand trajectories click detections.
    path_handtrajectories = dir_handtrajectories + filename_handtrajectories
    path_clickinfo        = dir_clickinfo + filename_clickinfo
    
    # If an xarray for the curtailed hand trajectories already exists in the file pathway, it must first be deleted before 
    # writing a new one to that location. This has to do with some property of netCDF4.
    path_handtrajectories_exists = os.path.exists(path_handtrajectories)
    if path_handtrajectories_exists:
        os.remove(path_handtrajectories)

    # Saving the curtailed hand trajectories.
    hand_trajectories_curt.to_netcdf(path_handtrajectories) 

    # Saving the curtailed click highlights.
    with open(path_clickinfo, "wb") as fp: pickle.dump(click_info_curt, fp)
    
    return None