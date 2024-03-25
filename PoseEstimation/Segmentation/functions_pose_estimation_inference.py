
# IMPORTING LIBRARIES:
import collections
import copy
import numpy as np
import os
import pickle
import shutil
import tensorflow as tf
import xarray as xr

import matplotlib.pyplot as plt

from pprint import pprint



def save_script_backup():
    
    """
    Automatically saving this entire script immediately when it's called.
    """
    
    # Creating the original and target file directories.
    original = r'/home/dan/Projects/PseudoOnlineTests_for_RTCoG/Scripts/Recent/PoseEstimation/functions_pose_estimation_inference.py'
    target   = r'/mnt/shared/danprocessing/BACKUP/Projects/PseudoOnlineTests_for_RTCoG/Scripts/PoseEstimation/functions_pose_estimation_inference.py'

    # Saving.
    shutil.copyfile(original, target)
    
# Immediately saving script.   
save_script_backup()





def concatenating_historical_features(features_no_history, fps):
    """
    DESCRIPTION:
    Based on the experimenter-specified time history (t_history) the number of historical time points are calculated. An
    xarray with dimensions (history, features, time) is created, where each coordinate in the history dimension represents
    how much the features were shifted in time. For example, consider one coordinate in the feature array, and suppose a
    time length of 10 samples and a total time history of 3 samples. For this feature, the resulting xarray would look like:

    historical time shifts
         n=2 shifts      [[0.000, 0.000, 0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101],
         n=1 shift        [0.000, 0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521],
         n=0 shifts       [0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521, 0.882]]
                            t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7    t=8    t=9     (time samples)   

    and the resulting dimensions of this array are (history=3, features=1, time=10).
    
    INPUT VARIABLES:
    features_no_history: [dictionary (Key: string (task ID); Value: xarray (landmarks x time samples) > floats)]; The time 
                         traces of the x- and y-coordinates for each landmark.
    fps:                 [int (30 or 60)]; Frames per second of of the video feed.
    
    GLOBAL PARAMETERS:
    t_history: [float (unit: s)]; Amount of time history used as features.
                           
    OUTPUT VARIABLES:
    features_all_history: [xarray (time history, features, time) > floats]; Array of historical time features.
    """
    
    # COMPUTATION:
    
    # Computing the frame duration given the FPS.
    t_frame = (1/fps) * 1000 # 1/(/s) * ms/s = s * ms/s = ms
    
    # Computing the total number of historical time features.
    n_history = int(t_history/t_frame)
    
    # Extracting the number of trajectories.
    n_features = features_no_history.shape[0]

    # Extracting the time array and corresponding number of time frames in the current block.
    t_seconds = features_no_history.time_seconds.values
    n_samples = t_seconds.shape[0]

    # Initializing a feature array which will contain all historical time features.
    features_all_history = np.zeros((n_history, n_features, n_samples))

    # Iterating across all historical time shifts. The index, n, is the number of samples back in time that will be shifted.
    for n in range(n_history):

        # If currently extracting historical time features (time shift > 0)
        if n >= 1:

            # Extracting the historical time features for the current time shift.
            these_features_history = features_no_history[:,:-n]

            # Creating a zero-padded array to make up for the time curtailed from the beginning of the features array.
            zero_padding = np.zeros((n_features, n))

            # Concatenating the features at the current historical time point with a zero-padded array.
            these_features = np.concatenate((zero_padding, these_features_history), axis=1)

        # If extracting the first time set of features (time shift = 0). 
        else:
            these_features = features_no_history

        # Assigning the current historical time features to the xarray with all historical time features.
        features_all_history[n,:,:] = these_features

    # Converting the historical trajectory features for this task to xarray.
    features_all_history = xr.DataArray(features_all_history, 
                                        coords={'history': np.arange(n_history), 'feature': np.arange(n_features), 'time_seconds': t_seconds}, 
                                        dims=["history", "feature", "time_seconds"])

    return features_all_history





def creating_onsetoffset_dictionary(predictions):
    """
    DESCRIPTION:
    Creating the dictionary of attempted movement onsets and offsets times.
    
    INPUT VARIABLES:
    predictions: [xarray > strings (1 x samples)]; The xarray of most likely outputs for each sample. The time 
                 dimension is in units of seconds.
                 
    NECESSARY FUNCTIONS:
    unique_value_index_finder
    
    OUTPUT VARIABLES:
    movement_onsetsoffsets: [dictionary (key: string (movement); Value: list > list [t_onset, t_offset] > floats)]; The 
                            dictionary containing all movement onset and offset times for each movement type.
    """
    
    # COMPUTATION:
    
    # Initializing the dictionary for movement onsets and offsets.
    movement_onsetsoffsets = {}

    # Extracting the time array from the predictions.
    t_seconds = predictions.time_seconds.values

    # Extracting the starting and ending indices of attempted movement.
    _, _, _, start_end_inds = unique_value_index_finder(predictions)

    # Extracting the unique classes.
    unique_classes = list(start_end_inds.keys())

    # Iterating across each class.
    for this_class in unique_classes:
        
        # Excluding the rest class.
        if this_class != 'rest':

            # Extracting the starting and ending indices for the current class.
            start_end_inds_this_class = start_end_inds[this_class]

            # Computing the total number of onsets and offsets for the current class.
            n_onsetsoffsets_this_class = len(start_end_inds_this_class)

            # Initializing the list of attemptedmovement onset and offset times.
            this_class_onsets_offsets = [None] * n_onsetsoffsets_this_class

            # Iterating across the onset and offset indices for the current class.
            for n, (this_onset_idx, this_offset_idx) in enumerate(start_end_inds_this_class):

                # Computing the onset and offset times.
                t_onset  = t_seconds[this_onset_idx]
                t_offset = t_seconds[this_offset_idx] 

                # Updating the list of onset and offset times.
                this_class_onsets_offsets[n] = [t_onset, t_offset]

            # Updating the dictionary with all the onset and offset times for the current class.
            movement_onsetsoffsets[this_class] = this_class_onsets_offsets

    return movement_onsetsoffsets





def data_upload(block_id, date, dir_intermediates, patient_id, task):
    """
    DESCRIPTION:
    Uploading the click information hand trajectories from the experimenter-specified date and block.
    
    INPUT VARIABLES:
    block_id:          [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:              [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_intermediates: [string]; Intermediates directory where relevant information is stored.
    patient_id:        [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:              [string]; Type of task that was run.
                   
    NECESSARY FUNCTIONS:
    load_click_information
    load_hand_trajectories
    
    OUTPUT VARIABLES:
    data_dict: [dictionary (key/value pairs below)];
        click_info: [dict (key: string ('backspace','keyboard','stimcolumn'); Values: below)];
            data:      [xarray (1 x time samples) > strings];  For each  time sample of the array of each key there
                       is a 'no_click' string or a click-string specific to that xarray. For example, the 'backspace'
                       key of the dictionary has an array where each element is a string named either 'no_click' or 
                       'backspace_click'. The 'backspace_click' elements do not occur consecutively and describe the 
                       instance a click on the backspace key occured. For the 'keyboard' and 'stimcolumn' keys, similar
                       rules apply. Time dimension is in units of s.
            plotcolor: [string]; Color corresponding to the type of click for plotting.
        trajectories:  [xarray (landmarks x time samples) > floats]; The time traces of the x- and y-coordinates 
                       for each landmark. The time domain is in units of seconds. 
    """
    # COMPUTATION:
    
    # Initializing a dictionary that will hold the hand trajectories and click information.
    data_dict = {}

    # Loading the hand trajectories.
    this_block_hand_trajectories = load_hand_trajectories(block_id, date, dir_intermediates, patient_id, task)

    # Loading the click information.
    this_block_click_info = load_click_information(block_id, date, dir_intermediates, patient_id, task)

    # Updating the data dictionary with the hand trajectories and click and click information.
    data_dict['trajectories'] = this_block_hand_trajectories
    data_dict['click_info']   = this_block_click_info
    
    return data_dict





def index_advancer(indices, shift):
    
    """
    DESCRIPTION:
    This function is used to update the indices of a vector X which is being repeatedly filled with smaller vectors A, B, 
    C, D, of a particular size, shift. For example, if vector X is initialized as A = zeros(1,20) and vectors A-D are 
    defined as follows:
    A = 1:5, B = 1:5, C = 12:16, D = 7:11;
    Then we may wish to fill the first 5 elements of X with A and so the inputs are: indices = [0,0], shift = length(A),
    and the output is indices = [1,5], We may wish to fill the next 5 elements with B, and so the inputs are 
    indices = [1,5], shift = length(B), and the output is indieces = [6,10].

    INPUT VARIABLES:
    indices: [array (1 x 2) > int]; The beginning and ending indices within the array that is being updated. The first
             time this function is called, indices = [0,0]. 
    shift:   [int]; Difference between the starting and ending indices. The length of the vector that is being placed
             inside the array for which this function is used.

    OUTPUT VARIABLES:
    indices: [array (1 x 2) > int]; The beginning and ending indices within the array that is being updated.
    """

    # COMPUTATION:
    indices[0] = indices[1];
    indices[1] = indices[0] + shift; 
    
    # Converting values inside indices array to integers.
    indices = indices.astype(int)
    
    return indices





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

    # PRINTING:
    
    # Printing all the hand landmarks.
    print('\nHAND LANDMARKS LIST:')
    pprint(list(hand_trajectories.landmarks.values))

    return hand_trajectories





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
    training_data_mean: [xarray (history x features) > floats]; Mean power of each feature of  only the 0th time
                        shift. This array is repeated for each historical time point.
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





def lstm_arrange_data(data, time_history):
    """
    DESCRIPTION:
    Rearrange the data so that each training sample's features from the time domain are organized into another dimension for LSTM use.

    INPUT VARIABLES:
    data:         [array (samples x features) > floats]; Training or testing data which will be rearranged.
    time_history: [list > int (units: ms)]; Historical time points which will be included during classification of each data sample. Specifically, if
                  time_history = [0,50,100], then the classification at time point t will occur with features from times t, t-50ms and t-100ms. If
                  no time history will be used, set time_history as an empty list []. Note: Do not make this time history list at higher resolution
                  than the spectral window shift.

    NECESSARY FUNCTIONS:
    index_advancer

    OUTPUT VARIABLES:
    data_lstm: [array (samples x time points x features) > floats]; Rearranged data fit for the LSTM.
    """
    
    # COMPUTATION:
    
    # Extracting the number of time points, training samples, and features from the current fold to organize the features in 
    # preparation for the LSTM.
    N_time_history = len(time_history)
    if N_time_history == 0:
        N_time_history = 1
        
    # Computing the number of samples and features.
    N_samples        = data.shape[0]
    N_features_all_t = data.shape[1]
    
    # Computing the number of features per historical time point.
    N_features_no_t = int(N_features_all_t / N_time_history)
    
    # Initializing the re-arranged data array. 
    data_lstm = np.zeros((N_samples, N_time_history, N_features_no_t))
    
    # Iterating across all time samples.
    for n in range(N_samples):
        
        # Extracting the current training sample.
        this_sample = data[n, :]
        
        # Initializing the time samples x feature array for the current time sample.
        this_sample_time_vs_feat = np.zeros((N_time_history, N_features_no_t))
        
        # Iterating across all time points.
        ind = np.zeros((2,))
        for t in range(N_time_history):
            
            # Extracting the indices of the current time feature.
            ind = index_advancer(ind, N_features_no_t)
            
            # Populating the time vs feature array with the features of the current time point.
            this_sample_time_vs_feat[t,:] = this_sample[ind[0]:ind[1]]

        # Populating the LSTM data with the current sample's rearranged temporal information.
        data_lstm[n,:,:] = this_sample_time_vs_feat

    return data_lstm





def mean_centering(data, data_mean):
    """
    DESCRIPTION:
    Mean-centering all features at all historical time points by subtracting the data mean, averaged across time.
        
    INPUT VARIABLES:
    data:      [xarray (history x features x time samples) > floats ]; Historical power features across time
               samples. Time samples are in units of seconds.
    data_mean: [xarray (history x features) > floats ]; Mean power of each feature of only the 0th time shift.
               This array is repeated for each historical time point.
    
    OUTPUT VARIABLES:
    data_centered: [xarray (history x features x time samples) > floats]; Mean-centered historical power features
                   across time samples. Time samples are in units of seconds.
    """
    
    # COMPUTATION:
    
    # Extracting the number of historical time points, features, and time samples from the data xarray.
    n_history  = data.history.shape[0]
    n_features = data.feature.shape[0]
    
    # Extracting the time array and total number of samples..
    t_seconds = data.time_seconds.values
    n_samples = t_seconds.shape[0]
    
    # Converting the data mean xarray to a regular array and repeating the means for each sample.
    data_mean = np.array(data_mean)
    data_mean = np.expand_dims(data_mean, axis=2)
    data_mean = np.tile(data_mean, (1, 1, n_samples))
    
    # Converting the data mean array back into an xarray
    data_mean = xr.DataArray(data_mean, 
                             coords={'history': np.arange(n_history), 'feature': np.arange(n_features), 'time_seconds': t_seconds}, 
                             dims=["history", "feature", "time_seconds"])
    
    # Computing the mean-centered data.
    data_centered = data - data_mean
    
    return data_centered





def model_inference(trajectories_processed):
    """
    DESCRIPTION:
    Classification of the trajectory features for each sample. 
    
    INPUT VARIABLES:
    trajectories_processed: [xarray (time samples x history x features) > floats]; The processed hand trajectories.
                            The time dimension is in units of seconds.
                            
    GLOBAL PARAMETERS:
    model:         [classification model]; The classification model.
    model_classes: [list > strings]; List of all the classes to be used in the classifier.
    model_type:    [string ('SVM','LSTM')]; The model type that will be used for classification.
    
    OUTPUT VARIABLES:
    predictions: [xarray > strings (1 x samples)]; The xarray of most likely outputs for each sample. The time 
                 dimension is in units of seconds.
    """
    
    # COMPUTATION:
    
    # Extracting the time array and total number of samples.
    t_seconds = trajectories_processed.time_seconds.values
    n_samples = t_seconds.shape[0]

    # Initializing a list of predicted labels.
    predictions = [None] * n_samples

    # SVM
    if model_type == 'SVM':
        pass

    # LSTM
    if model_type == 'LSTM':

        # Computing the predicted probabilities for all samples..
        probabilities = np.squeeze(model.predict(trajectories_processed.values))
        
        # Iterating across all samples.
        for n in range(n_samples):

            # Extracting the probabilities of the current sample.
            this_sample_probs = probabilities[n,:]

            # Classifying the current sample according to the predicted probabilities.
            classification_ind  = int(np.argmax(this_sample_probs))
            predictions[n]      = model_classes[classification_ind]
                
    # Converting the predictions list into an xarray.
    predictions = xr.DataArray(predictions,
                               coords={'time_seconds': t_seconds},
                               dims=["time_seconds"])

    return predictions





def pc_transform(data, eigenvectors):
    """
    DESCRIPTION:
    Transforming the data into PC space by multiplying the features at each historical time point by the same eigenvectors.
    
    INPUT VARIABLES:
    data:         [xarray (features x time samples) > floats];
    eigenvectors: [array (features x pc features) > floats]; Array in which columns consist eigenvectors which explain the
                  variance in features in descending order. Time samples are in units of seconds.
    
    OUTPUT VARIABLES:
    data_pc: [xarray (pc features x time samples) > floats (units: PC units)]; Reduced-dimensionality data. Time dimension is
             in units of seconds.
    """
    
    # COMPUTATION:
    
    # Extracting the number of historical time shifts.
    n_history = data.history.shape[0]
    
    # Extracting the time array and number of samples.
    t_seconds = data.time_seconds.values
    n_samples = t_seconds.shape[0]
    
    # Computing the number of PC features.
    n_features_pc = eigenvectors.shape[1]
    
    # Initializing an xarray of PC data.
    data_pc = xr.DataArray((np.zeros((n_history, n_features_pc, n_samples))),
                            coords={'history': np.arange(n_history), 'feature': np.arange(n_features_pc), 'time_seconds': t_seconds},
                            dims=["history", "feature", "time_seconds"])
    
    # Iterating across all historical time shifts and multiplying the data at each historical time shift by the same
    # eigenvectors.
    for n in range(n_history):
        
        # Extracting the data at the nth time shift.
        this_history_data = np.asarray(data.loc[n,:,:])
        
        # Transforming hte data of the nth time shift to PC space.
        data_pc.loc[n,:,:] = np.matmul(this_history_data.transpose(), eigenvectors).transpose()   
    
    return data_pc





def plotting_landmarks_and_clicks(data_dict, landmark_trajectories_plotting):
    """
    DESCRIPTION:
    Plotting the experimenter-specified hand landmarks and the click information across the entirety of
    experimenter-specified date and block.
    
    INPUT VARIABLES:
    data_dict: [dictionary (key/value pairs below)];
        click_info: [dict (key: string ('backspace','keyboard','stimcolumn'); Values: below)];
            data:      [xarray (1 x time samples) > strings];  For each  time sample of the array of each key there
                       is a 'no_click' string or a click-string specific to that xarray. For example, the 'backspace'
                       key of the dictionary has an array where each element is a string named either 'no_click' or 
                       'backspace_click'. The 'backspace_click' elements do not occur consecutively and describe the 
                       instance a click on the backspace key occured. For the 'keyboard' and 'stimcolumn' keys, similar
                       rules apply. Time dimension is in units of s.
            plotcolor: [string]; Color corresponding to the type of click for plotting.
        trajectories:  [xarray (landmarks x time samples) > floats]; The time traces of the x- and y-coordinates 
                       for each landmark. The time domain is in units of seconds. 
    landmark_trajectories_plotting: [list > strings]; Possible landmarks to display.
    """
    
    # PLOTTING
    
    # Extracting the click and hand trajectory information.
    click_info        = data_dict['click_info']
    hand_trajectories = data_dict['trajectories']
    
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





def preprocessing_features(trajectories, fps):
    """
    DESCRIPTION:
    Preprocessing the trajectories for classification.
    
    INPUT VARIABLES:
    trajectories: [xarray (landmarks x time samples) > floats]; The time traces of the x- and y-coordinates 
                  for each landmark. The time domain is in units of seconds. 
                  
    GLOBAL PARAMETERS:
    eigenvectors:       [array (features x pc features) > floats]; Array in which columns consist eigenvectors 
                        which explain the variance in features in descending fashion. 
    model_type:         [string ('SVM','LSTM')]; The model type that will be used for classification.
    training_data_mean: [xarray (history x features) > floats]; Mean power of each feature of  only the 0th time
                        shift. This array is repeated for each historical time point.
                   
    NECESSARY FUNCTIONS:
    concatenating_historical_features
    mean_centering
    pc_transform
    rearranging_features
    
    OUTPUT VARIABLES:
    trajectories_processed: [xarray (time samples x history x features) > floats]; The processed hand trajectories.
                            The time dimension is in units of seconds.
    """
    
    # COMPUTATION:
    
    # Creating feature array historical time shifts.
    trajectories = concatenating_historical_features(trajectories, fps)
    
    # Mean-centering the trajectories to the mean from the model training data.
    trajectories = mean_centering(trajectories, training_data_mean)

    # Computing the reduced-dimension PC training data.
    trajectories = pc_transform(trajectories, eigenvectors)

    # Rearranging features to fit the appropriate model type.
    trajectories_processed = rearranging_features(trajectories, model_type)
    
    return trajectories_processed





def rearranging_features(data, model_type):
    """
    DESCRIPTION:
    Rearranging the data dimensions as necessary to fit the experimenter-determined model.
    
    INPUT VARIABLES:
    data:       [xarray (time history x features x time samples) > floats]; Array of historical time features.
    model_type: [string ('SVM','LSTM')]; The model type that will be used to fit the data.
    
    OUTPUT VARIABLES:
    data_rearranged: [xarray (dimensions vary based on model type) > floats]; Rearranged data. 
    """
    # COMPUTATION:
    
    # Extracting the dimension sizes of the current features array.
    n_history  = data.history.shape[0]
    n_features = data.feature.shape[0]
    
    # Extracting the time array and total number of samples.
    t_seconds = data.time_seconds.values
    n_samples = t_seconds.shape[0]
    
    # If the model type is a SVM.
    if model_type == 'SVM':
        
        # NOTE: This script doesn't have a SVM model structure for training. Feel free to write one. The data is rearranged
        # for it.

        # Concatenating all the historical time features into one dimension.
        data_rearranged = np.asarray(data).reshape(n_history*n_features, n_samples)
        data_rearranged = data_rearranged.transpose()

        # Converting the rearranged features back into an xarray.
        data_rearranged = xr.DataArray(data_rearranged, 
                                       coords={'time_seconds': t_seconds, 'feature': np.arange(n_history*n_features)}, 
                                       dims=["time_seconds", "feature"])

    # If the model type is an LSTM.
    if model_type == 'LSTM':

        # Don't rearrange the features. They already have the correct dimensionality for LSTM.
        data_rearranged = data.transpose("time_seconds","history","feature")
    
    return data_rearranged





def referencing_hand_trajectories(data_dict, ref1_x, ref2_x, refa_y, refb_y):
    """
    DESCRIPTION:
    Each hand landmark is referenced according to experimenter-specified landmarks. Make sure that the landmarks that are
    selected will not be used for further analysis as they will get normalized out to 0.
    
    INPUT VARIABLES:
    data_dict: [dictionary (relevant key/value pairs below)];
        trajectories:  [xarray (landmarks x time samples) > floats]; The time traces of the x- and y-coordinates 
                       for each landmark. The time domain is in units of seconds. 
    ref1_x:            [string]; First horizontal reference landmark
    ref2_x:            [string]; Second horizontal reference landmark
    refa_y:            [string]; First vertical reference landmark
    refb_y:            [string]; Second vertical reference landmark

    OUTPUT VARIABLES:
    data_dict: [dictionary (relevant key/value pairs below)];
        trajectories:  [xarray (landmarks x time samples) > floats]; The time traces of the x- and y-coordinates 
                       for each landmark. The time domain is in units of seconds. 
    """
    
    # COMPUTATION:
    
    # Extracting the hand trajectories from the data dictionary.
    hand_trajectories = data_dict['trajectories']

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

    # Assigning the referenced hand trajectories back to data dictionary.
    data_dict['trajectories'] = hand_trajectories_norm
    
    return data_dict

                
                
                
                
def saving(block_id, date, dir_intermediates, movement_onsetsoffsets, patient_id, task):
    """
    DESCRIPTION:
    Saving the dictionary of movement onsets and offsets.
    
    INPUT VARIABLES:
    block_id:               [String (BlockX, where X is an int))]; Block ID of the task that was run.
    date:                   [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_intermediates:      [string]; Intermediates directory where relevant information is stored.
    movement_onsetsoffsets: [dictionary (key: string (movement); Value: list > list [t_onset, t_offset] > floats)]; The 
                            dictionary containing all movement onset and offset times for each movement type.
    patient_id:             [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:                   [string]; Type of task that was run.
    """
    
    # COMPUTATION:
    
    # Creating the directory and filename to the movement onsets/offsets dictionary.
    dir_onsetsoffsets      = dir_intermediates + patient_id + '/' + task + '/MovementOnsetsAndOffsets/' + date + '/'
    filename_onsetsoffsets = 'dict_OnsetOffset_' + block_id

    # Creating the pathway to the movement onset/offset dictionary.
    path_onsetsoffsets_dict = dir_onsetsoffsets + filename_onsetsoffsets

    # Saving the updated dictionary.
    with open(path_onsetsoffsets_dict, "wb") as fp: pickle.dump(movement_onsetsoffsets, fp)
    
    return None      





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
    foldername_history_params,\
    foldername_model_params,\
    foldername_pc_params = [name for name in sorted(os.listdir(dir_model_config))]
                
    # Loading each set of parameters.
    loading_history_params(foldername_history_params, dir_model_config)
    loading_model(foldername_model_params, dir_model_config)
    loading_model_params(foldername_model_params, dir_model_config)
    loading_pc_params(foldername_pc_params, dir_model_config)





def unique_value_index_finder(stepwise_sequence):
    """
    DESCRIPTION: 
    This function finds useful indexing information about step-wise sequences of numbers.

    INPUT VARIABLES:
    stepwise_sequence: [list > (ints/strings)]; List of integers or strings in which steps are formed by clusters of
                       one integer or string.
        
    OUTPUT VARIABLES:
    unique_vals:     [list > (ints/strings)] Individual values (at different "heights") of all the steps in the vector. 
                     For example, the unique_vals of 
                     ex_stepwise_sequence = ['a','a','a','a','b','b','b','b','a','a','a','a','c','c','c','c','a','a','a','a'] 
                     would be [a,b,c]
    n_steps_per_val: [dictionary (Key: ints/strings (step names); Value: ints (number of occurrence per step name)]; The 
                     number of steps for each unique value. For example n_steps_per_val of ex_stepwise_sequence would be:
                     {a: 3, b: 1, c: 1} because the "a" step occurs 3 times, whereas "b" and "c" steps occur only once.
    unique_val_inds: [dictionary (Key: ints/strings (step names); Value: list > ints (array indices))]; All the indices within 
                     the stepwise_sequence list of where a specific step occurs. For example, the unique_vals_inds of 
                     ex_stepwise_sequence would be {a: [0,1,2,3,8,9,10,11,16,17,18,19], b: [4,5,6,7], c: [12,13,14,15]}.

    start_end_inds:  [dictionary (Key: ints/strings (step names); Value: list > list > ints (array indices))]; The indices 
                     within the stepwise_sequence list where individual steps start and end. For exaple, the start_end_inds of
                     ex_stepwise_sequence would be {a: [[0,3],[8,11],[16,19]], b: [[4,7]], c: [[12,15]]}.
    """
        
    # COMPUTATION:
    
    # Find the unique sorted elements of stepwise_sequence.
    unique_vals = np.unique(stepwise_sequence)
    
    # Initiate the list of indices.
    unique_val_inds = {}
    start_end_inds  = {}
    n_steps_per_val = {}
    
    # Iterate across each value in unique_vals.
    for this_val in unique_vals:
                
        this_val_inds = [i for i, x in enumerate(stepwise_sequence) if x == this_val]
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
    
        # Updating the dictionaries.
        n_steps_per_val[this_val] = len(inds)
        unique_val_inds[this_val] = this_val_inds
        start_end_inds[this_val]  = inds
        
    # Converting the unique values to a list.
    unique_vals = list(unique_vals)
    
    return unique_vals, n_steps_per_val, unique_val_inds, start_end_inds