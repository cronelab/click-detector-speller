
# IMPORTING LIBRARIES:
import collections
import copy
import numpy as np
import os
import pickle
import seaborn as sns
import tensorflow as tf
import xarray as xr

import matplotlib.pyplot as plt

from pprint import pprint
from random import sample

from sklearn.metrics import confusion_matrix
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam





def computing_eigenvectors(data, n_pc_thr, percent_var_thr):
    """
    DESCRIPTION:
    Computing the eigenvectors of the data array. The eigenvectors will be curtailed according to one of two
    experimenter-specified criteria:
        
    1) Only keep the first eiggenvectors that that in total explain less than or equal variance to percent_var_thr
       ... or ... 
    2) Only keep the first n_pc_thr eigenvectors.
        
    INPUT VARIABLES:
    data:            [array (features x samples) > floats];
    n_pc_thr:        [int]; The number of principal components to which the user wishes to reduce the data set. Set to 'None' if
                     percent_var_thr is not 'None', or set to 'None' along with percent_var_thr if all of the variance will be used
                     (no PC transform).
    percent_var_thr: [float]; The percent variance which the user wishes to capture with the principal components. Will compute the
                     number of principal components which capture this explained variance as close as possible, but will not surpass
                     it. Set to 'None' if n_pc_thr is not 'None', or set to 'None' along with n_pc_thr if all of the variance will be
                     used (no PC transform).
    OUTPUT VARIABLES:
    eigenvectors: [array (features x pc features) > floats]; Array in which columns consist of eigenvectors which explain the
                  variance of the data in descending order. 
    """
    
    # COMPUTATION:
    
    # Computing the PCs of the training data if the experimenter has entered a value for the expected percentage variance explained
    # or for the number of PCs to be computed.
    if (percent_var_thr != 'None') or  (n_pc_thr != 'None'):
        
        # Computing the number of samples in the data.
        n_samples = data.shape[1]
        
        # Computing the covariance matrix of the data.
        C = np.matmul(data, data.transpose())
        
        # Computing the eigenvectors and eigenvalues of the covariance matrix. 
        [D,V] = np.linalg.eig(C)
        
        # If there are more features than samples, curtail the the eigenvalues and eigenvectors to exclude the indices past 
        # the number of samples. Due to Python's numerical approximation, when there are more features than samples, the 
        # eigen-decomposition returns complex eigenvalues and eigenvectors. This shouldn't ever mathematically happen because
        # the covariance matrix is symmetric, and therefore only has real eigenvalues and eigenvectors.
        D = D[:n_samples].real
        V = V[:,:n_samples].real
        
        # Eigenvectors and eigenvalues are sorted and flipped to the descending order of the Eigenvalues.
        eig_values_dsc_inds = np.argsort(-D)
        eig_vals            = D[eig_values_dsc_inds]
        eig_vecs            = V[:,eig_values_dsc_inds]
        
        # Calculating the percentage of variance explained from each eigenvector by cumulatively summing the eigenvalues.
        percent_var = np.cumsum(eig_vals)/np.sum(eig_vals);
        
        # Depending on the experimenter-input, the eigenvectors will be extracted based on the percent variance explained threshold
        # or the number of principal components threshold. 
        if percent_var_thr != 'None':
            eigenvectors = eig_vecs[:,percent_var <= percent_var_thr];
            
            # If the size of the eigenvectors array is 0, this means that the first eigenvector explained more than the threshold
            # percent variance, and so it wasn't captured. Manually setting the eigenvectors variable to only the first eigenvector.
            if eigenvectors.size == 0:
                eigenvectors = eig_vecs[:,0];
            
        if n_pc_thr != 'None':
            eigenvectors = eig_vecs[:,0:n_pc_thr];
       

        # If there is only one eigenvector, the second dimension of the array must be expanded from nothing to 1.
        if len(list(eigenvectors.shape)) == 1:
            eigenvectors = np.expand_dims(eigenvectors[n],axis=1)   
            
        # Enforcing the rule that the first element of each eigenvector should be positive, and if it is not, then the entire 
        # eigenvector will be multiplied by a -1. This is to ensure eigenvector similarity across multiple data uploads.
        n_eig = eigenvectors.shape[1]
        for n in range(n_eig):
            this_eigenvector  = eigenvectors[:,n]
            element0_sign     = np.sign(this_eigenvector[0])
            eigenvectors[:,n] = element0_sign*this_eigenvector

        # Computing the number of PCs and percent variance explained. 
        n_pc                  = eigenvectors.shape[1];
        percent_var_explained = percent_var[n_pc-1]*100;
        
        
    # No PCs will be computed. The experimenter has entered 'None' for both percent_var_thr and for n_pc_thr.
    else:
        
        # Computing the total number of features.
        n_features = data.shape[0]
        
        # Creating the eigenvectors simply as an identity matrix.
        eigenvectors = np.identity(n_features)
        
        # Computing the number of PCs and percent variance explained. 
        n_pc                  = [];
        percent_var_explained = 100;
    
    
    # PRINTING 
    
    # Printing out the number of PCs and explained variance.
    print('\nNUMBER OF PRINCIPAL COMPONENTS: ', n_pc)
    print('\nPERCENT VARIANCE EXPLAINED: ', percent_var_explained)
    
    return eigenvectors





def computing_predicted_labels(fold_models, valid_data_folds, valid_labels_folds):
    """
    DESCRIPTION:
    For each fold of validation data, the approrpirate fold model is used to determine the predicted label for each sample.
    Predicted labels across all folds are then concatenated together in list. Corresponding true labels for each fold are
    also concatenated together.
    
    INPUT VARIABLES:
    fold_models:        [dictionary (key: string (fold ID); Value: model)]; Models trained for each training fold.
    valid_data_folds:   [dict (key: string (fold ID); Value: xarray (dimensions vary based on model type) > floats)];
                        Data across all validation tasks per fold. Equal number of samples per class. PC features. Rearranged 
                        according to the type of model that will be trained.
    valid_labels_folds: [dict (key: string (fold ID); Value: xarray (1 x time samples) > strings (labels))]; Labels across
                        all validation tasks per fold. Equal number of labels per class.
                    
    NECESSARY FUNCTIONS:
    index_advancer
    
    OUTPUT VARIABLES:
    pred_labels: [list > strings]; The predicted labels for each sample across all tasks for each validation folds. 
    true_labels: [list > strings]; The true labels for each sample across all tasks for each validation folds. 
    """
    
    # COMPUTATION:
    
    # Computing the number of samples across all folds.
    n_samples_all_folds = 0
    
    # Iterating across all folds to update the total number of samples across all folds.
    for this_fold in valid_data_folds.keys():
        
        # Computing the number of samples in the current fold.
        n_samples_this_fold = valid_labels_folds[this_fold].shape[0]
        
        # Updating the number of samples.
        n_samples_all_folds += n_samples_this_fold
    
    # Initializing the lists of true and predicted labels across all folds.
    true_labels = [None] * n_samples_all_folds
    pred_labels = [None] * n_samples_all_folds
    
    # Initializing the array of validation indices.
    valid_inds = np.zeros((2,))
    
    # Iterating across all folds.
    for this_fold in valid_data_folds.keys():
        
        # Extracting the validation data and labels from the current fold. Transforming data back into an array because tensorflow
        # models produce warnings when given xarrays.
        this_fold_validation_data   = np.asarray(valid_data_folds[this_fold])
        this_fold_validation_labels = valid_labels_folds[this_fold]
        
        # Extracting the sorted model classes from the validation data.
        model_classes = np.sort(np.unique(this_fold_validation_labels)).tolist()
        
        # Initializing the list of predicted labels for the current fold.
        this_fold_predicted_labels = [None] * n_samples_this_fold
    
        # Extracting the corresponding model to use on the current fold of validation data.
        this_fold_model = fold_models[this_fold]
            
        # Computing the predicted probabilities for each label in the current fold.
        this_fold_predicted_probs = this_fold_model.predict(this_fold_validation_data);
        
        # Creating the predicted labels array by finding the index of the maximum probability for each sample.
        this_fold_max_probs_inds   = np.argmax(this_fold_predicted_probs, axis=1) 
        this_fold_predicted_labels = [model_classes[n] for n in this_fold_max_probs_inds]
        
        # Computing the number of samples for the current fold.
        n_samples_this_fold = this_fold_validation_labels.shape[0]
    
        # Updating the validation indices.
        valid_inds = index_advancer(valid_inds, n_samples_this_fold)
    
        # Updating the array of predicted and true labels. 
        true_labels[valid_inds[0]:valid_inds[1]] = this_fold_validation_labels.values
        pred_labels[valid_inds[0]:valid_inds[1]] = this_fold_predicted_labels
    
    return pred_labels, true_labels





def concatenating_all_data_and_labels(features_dict, labels_dict):
    """
    DESCRIPTION:
    Concatenating features and labels across all tasks in the sample dimension.
    
    INPUT VARIABLES:
    features_dict: [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats (units: V^2/Hz))]
                   Array of historical time features. 
    labels_dict:   [dictionary (Key: string (task ID); Value: xarray > strings ('grasp'/'rest'))]; For each time sample in each
                   task, there exists a rest or grasp label depending on the experimenter-specified onset and offset of
                   modulation as well as the per-trial shift from the AW model. 
                   
    OUTPUT VARIABLES:
    training_data:   [xarray (dimensions vary based on model type) > floats (units: V^2/Hz)]; For each task, feature xarrays are 
                     concatenated in the sample dimension.
    training_labels: [xarray (1 x time samples) > strings ('grasp'/'rest')]; For each task, label xarrays are concatenated in the
                     sample dimension.
    """
    
    # COMPUTATION:
    
    # Initialize the task flag.
    task0_flag = True
        
    # Iterating across all tasks.
    for n, this_task in enumerate(features_dict.keys()):
        
        # Extracting the training data and labels of the current task.
        this_task_data   = features_dict[this_task]
        this_task_labels = labels_dict[this_task]

        # If the training task counter is 0, intiailize the training data and labels xarrays. If not, concatenate
        # them with data and labels from another task.
        if task0_flag:            
            training_data   = this_task_data
            training_labels = this_task_labels
            
            # Setting the flag to False to never enter this IF statement again.
            task0_flag = False
            
        else:             
            training_data   = xr.concat([training_data, this_task_data], dim="sample")
            training_labels = xr.concat([training_labels, this_task_labels], dim='sample')

        # Reassigning the sample coordinates to the training data and labels xarrays.
        training_data   = training_data.assign_coords(sample=np.arange(training_data.sample.shape[0]))
        training_labels = training_labels.assign_coords(sample=np.arange(training_labels.sample.shape[0]))
    
    return training_data, training_labels





def concatenating_historical_features(features_no_history, fps, t_history):
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
    t_history:           [float (unit: s)]; Amount of time history used as features.
                           
    OUTPUT VARIABLES:
    trajectories_all_history: [dictionary (Key: string (task ID); Value: xarray (time history, features, time) > floats)]; 
                              Array of historical time features.
    """
    
    # COMPUTATION:
    
    # Initializing a dictionary of historical time features.
    trajectories_all_history = {}
    
    # Computing the frame duration given the FPS.
    t_frame = (1/fps) * 1000 # 1/(/s) * ms/s = s * ms/s = ms
    
    # Computing the total number of historical time features.
    n_history = int(t_history/t_frame)
    
    # Extracting the number of trajectories. Computing from the first date+block pair, as the number of features should be the 
    # same across all tasks.
    n_features = features_no_history[list(features_no_history.keys())[0]].shape[0]

    # Iterating across all date+block pairs.
    for i, this_date_block_id in enumerate(features_no_history.keys()):
        
        # Extracting the data from the current task.
        this_task_trajectories = features_no_history[this_date_block_id]

        # Extracting the time array and corresponding number of time frames in the current block.
        t_seconds = this_task_trajectories.time_seconds.values
        n_frames  = t_seconds.shape[0]
        
        # Initializing a feature array which will contain all historical time features.
        this_task_trajectories_all_history = np.zeros((n_history, n_features, n_frames))
                
        # Iterating across all historical time shifts. The index, n, is the number of samples back in time that will be shifted.
        for n in range(n_history):
            
            # If currently extracting historical time features (time shift > 0)
            if n >= 1:
                
                # Extracting the historical time features for the current time shift.
                these_features_history = this_task_trajectories[:,:-n]
                
                # Creating a zero-padded array to make up for the time curtailed from the beginning of the features array.
                zero_padding = np.zeros((n_features, n))
                
                # Concatenating the features at the current historical time point with a zero-padded array.
                these_features = np.concatenate((zero_padding, these_features_history), axis=1)
                
            # If extracting the first time set of features (time shift = 0). 
            else:
                these_features = this_task_trajectories
                        
            # Assigning the current historical time features to the xarray with all historical time features.
            this_task_trajectories_all_history[n,:,:] = these_features
        
        # Converting the historical trajectory features for this task to xarray.
        this_task_trajectories_all_history = xr.DataArray(this_task_trajectories_all_history, 
                                                          coords={'history': np.arange(n_history), 'feature': np.arange(n_features), 'time': t_seconds}, 
                                                          dims=["history", "feature", "time"])
         
        # Creating task ID.
        task_id = 'task' + str(i)
                
        # Adding the historical time features for the current date+block pair to the dictionary.
        trajectories_all_history[task_id] = this_task_trajectories_all_history
    
    return trajectories_all_history





def confusion_matrix_display(cm, model_classes, suppress_figs='No'):
    """
    DESCRIPTION:
    Given the user-input confusion matrix array and corresponding labels, the confusion matrix is displayed.

    INPUT VARIABLES:
    cm:            [array > float]; Confusion matrix holding the accuracy of the true (vertical axis) and predicted (horizontal axis) labels.
    model_classes: [list > strings]; Class labels for the confusion matrix.
    """
    
    # COMPUTATION:
    
    # Computing the normalized confusion matrix for display. Rounding the values to the nearest integer.
    cm_norm       = np.divide(cm.transpose(), np.sum(cm, axis=1)).transpose()*100
    cm_norm_round = np.round(cm_norm, 0)    
    
    # Computing the accuracy using the diagonal. 
    acc = np.round(np.trace(cm) / sum(sum(cm))*100, 2)
    
    # Showing the confusion matrix. 
    if suppress_figs == 'No':
        
        fig, ax = plt.subplots(figsize = (15, 10));
        ax = sns.heatmap(cm_norm_round, annot = True, cmap = 'Blues', xticklabels = model_classes,\
                         yticklabels = model_classes, vmin = 0, vmax = 100, annot_kws={"size": 40},\
                         cbar_kws={'label': 'Accuracy (%)'})
        
        ax.figure.axes[-1].yaxis.label.set_size(20)
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 18, rotation = 90)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 18, rotation = 0)
        ax.set_xlabel('Predicted Class', fontsize=25)
        ax.set_ylabel('True Class', fontsize=25)
        for t in ax.texts: t.set_text(t.get_text() + " %")
        
        fig_title = 'Balanced frame-wise Accuracy ('+str(acc)+'%)'
        plt.text(0.5, 1.04, fig_title,
                 horizontalalignment='center',
                 fontsize=30,
                 transform = ax.transAxes)
        
    if suppress_figs == 'Yes':
        fig = []
        
    return None





def creating_features(data_dict, fps, t_history):
    """
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
    
    Note: Instead of date+block ID, the nomenclature will switch to task ID in the output.
    
    INPUT VARIABLES:
    data_dict:         [dictionary (key: string (date+block ID); value: dictionary (relevant key/value pairs below))];
        trajectories:  [xarray (landmarks x time samples) > floats]; The time traces of the x- and y-coordinates for 
                       each landmark. The time domain is in units of seconds.      
    fps:               [int (30 or 60)]; Frames per second of of the video feed.
    t_history:         [float (unit: s)]; Amount of time history used as features.
    
    NECESSARY FUNCTIONS:
    concatenating_historical_features
                           
    OUTPUT VARIABLES:
    features_dict: [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats)]; Array of 
                   historical time features.
    """
    
    # COMPUTATION:
    
    # Creating a features dictionary.
    features_dict = {}

    # Iterating across all date+block pairs.
    for this_date_block_id in data_dict.keys():

        # Copying the trajectories (with all features, not only relevant) to the features dictionary.
        features_dict[this_date_block_id] = data_dict[this_date_block_id]['trajectories']

    # Creating the historical time features.
    features_dict = concatenating_historical_features(features_dict, fps, t_history) 
    
    return features_dict





def creating_labels(data_dict):
    """
    DESCRIPTION:
    Labeling each video frame across all blocks according to the experimenter-specified attempted movement onsets 
    and offsets.
    
    Note: Instead of date+block ID, the nomenclature will switch to task ID in the output.
    
    INPUT VARIABLES:
    data_dict:         [dictionary (key: string (date+block ID); value: dictionary (key/value pairs below))];
        onsetsoffsets: [list > list [t_onset, t_offset] > floats (units: s)]; The dictionary containing all movement 
                       onset and offset times for each movement type.
        trajectories:  [xarray (landmarks x time samples) > floats]; The time traces of the x- and y-coordinates for 
                       each landmark. The time domain is in units of seconds.      
                           
    OUTPUT VARIABLES:
    labels_dict: [dictionary (Key: string (task ID); Value: xarray > strings (labels))]; For each time frame in each
                 task, there exists a rest or movement-type label depending on the experimenter-specified onset and
                 offset of attempted movements.
    """
    
    # COMPUTATION:
    
    # Initializing a dictionary of labels.
    labels_dict = {}

    # Iterating across all date+block pairs.
    for n, this_date_block_id in enumerate(data_dict.keys()):

        # Extracting the data from the current date and block.
        this_data = data_dict[this_date_block_id]

        # Flag to initialize the labels array.
        flag_init_labels = True

        # Extracting the onsets and offsets for the current block.
        these_onsets_offsets = this_data['onsetsoffsets']

        # Iterating across all types of movements.
        for this_movement in these_onsets_offsets.keys():

            # Extracting the movement onset and offset times for the current movement.
            this_movemnent_onsets_offsets = these_onsets_offsets[this_movement]

            # Initialize the labels array only once such that it is not re-initialized with every movement.
            if flag_init_labels:

                # Extracting the time array and number of frames.
                t_seconds = this_data['trajectories_relevant'][this_movement].time_seconds.values            
                n_frames  = t_seconds.shape[0]

                # Iniitalizing the array of labels for the current block.
                this_date_block_labels = ['rest']*n_frames

                # Switch the flag to false to not enter the IF statement until the next date+block pair.
                flag_init_labels = False

            else:
                pass

            # Iterating across all pairs of movement onsets and offsets for the current movement.
            for this_onset_time, this_offset_time in this_movemnent_onsets_offsets:

                # Determining the onset and offset indices.
                onset_idx  = np.argmin(np.abs(t_seconds - this_onset_time))
                offset_idx = np.argmin(np.abs(t_seconds - this_offset_time))
                
                # Computing the total number of indices for the current individual movement.
                n_inds = offset_idx - onset_idx

                # Updating the labels for the current block.
                this_date_block_labels[onset_idx:offset_idx] = [this_movement] * n_inds

        # After iterating across all movements, convert the labels array into an xarray.
        this_date_block_labels = np.asarray(this_date_block_labels)
        this_date_block_labels = xr.DataArray(this_date_block_labels, 
                                              coords={'time': t_seconds}, 
                                              dims=["time"])
        
        # Creating task ID.
        task_id = 'task' + str(n)

        # Updating the labels dictionary.
        labels_dict[task_id] = this_date_block_labels
    
    return labels_dict





def data_upload(date_block_dict, dir_intermediates, patient_id, task):
    """
    DESCRIPTION:
    Uploading the click, hand trajectories and movement onsets and offsets from each experimenter-input block in
    the date_block_dict.
    
    INPUT VARIABLES:
    date_block_dict:   [dictionary (key: string (YYYY_MM_DD); value: list > strings (block IDs)]; The keys and vlaues of
                       the dictionary correspond to dates and block IDs respectively that will be used to train the 
                       movement onset and offset detector.
    dir_intermediates: [string]; Intermediates directory where relevant information is stored.
    patient_id:        [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:              [string]; Type of task that was run.
                   
    NECESSARY FUNCTIONS:
    load_click_information
    load_hand_trajectories
    load_movement_onsetsoffsets
    
    OUTPUT VARIABLES:
    data_dict: [dictionary (key: string (date+block ID); value: below)];
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
    
    # Initializing a dictionary that will hold the movement onsets and offsets, the hand trajectories, and click 
    # information for each of the blocks.
    data_dict = collections.defaultdict(dict)

    # Iterating across all keys and values of the dictionary.
    for this_date in date_block_dict.keys():
        for this_block in date_block_dict[this_date]:

            # Uploading the dictionary with movement onset and offsets.
            this_block_onsets_offsets = load_movement_onsetsoffsets(this_block, this_date, dir_intermediates, patient_id, task)

            # Loading the hand trajectories.
            this_block_hand_trajectories = load_hand_trajectories(this_block, this_date, dir_intermediates, patient_id, task)

            # Loading the click information.
            this_block_click_info = load_click_information(this_block, this_date, dir_intermediates, patient_id, task)

            # Creating the date+block ID
            date_block_id = this_date + '_' + this_block

            # Updating the data dictionary with the movement onsets and offsets, hand trajectories, and click 
            # information from the current block.
            data_dict[date_block_id]['onsetsoffsets'] = this_block_onsets_offsets
            data_dict[date_block_id]['trajectories']  = this_block_hand_trajectories
            data_dict[date_block_id]['click_info']    = this_block_click_info
    
    return data_dict





def equalizing_samples_per_class(features_dict, labels_dict):
    """
    DESCRIPTION:
    According to the label array, it is unlikely that there are an equal number of samples per class. For whichever
    class there are the msot samples, the indices of this class will be extracted to create a smaller subset of 
    indices in number equal to that of the underrepresented classes. For example, consider the following labels array:
    
    labels:  ['rest', 'rest', 'rest', 'rest', 'rest, 'grasp', 'grasp', 'grasp', 'rest', 'rest', 'rest', 'rest']
    indices:    0       1       2       3       4       5        6        7       8       9       10      11

    In the example, there are 3 samples with grasp labels, while there are 9 samples with rest labels. The indices
    corresponding to the rest labels are: [0, 1, 2, 3, 4, 8, 9, 10, 11]. These indices will be randomly subsampled
    such that they are equal in number to the grasp class. For example: [0, 1, 2] or [1, 4, 10] or [3, 9, 11]. As
    such the downsampled labels (and corresponding features) will use the indices:

    labels downsampled:  ['rest', 'grasp', 'grasp', 'grasp', 'rest', 'rest']
    indices downsampled:    3        5        6        7       9       11

    INPUT VARIABLES:
    features_dict: [dictionary (Key: string (date+block ID); Value: xarray (time history x features x time) > floats)]; Array of 
                   historical time features.
    labels_dict:   [dictionary (Key: string (date+block ID); Value: xarray > strings (labels))]; For each time frame in each
                   block, there exists a rest or movement-type label depending on the experimenter-specified onset and
                   offset of attempted movements.
                   
    NECESSARY FUNCTIONS:
    index_advancer
                   
    OUTPUT VARIABLES:
    features_dict: [dictionary (Key: string (date+block ID); Value: xarray (time history x features x time) > floats)]; Array of 
                   historical time features. Time samples reduced such that there are an equal number of features per class.
    labels_dict:   [dictionary (Key: string (date+block ID); Value: xarray > strings (labels))]; For each time frame in each
                   block, there exists a rest or movement-type label depending on the experimenter-specified onset and
                   offset of attempted movements. Time samples reduced such that there are an equal number of features per
                   class.
    """
    
    # COMPUTATION:
    
    # Iterating across all date+block pairs.
    for this_date_block_id in labels_dict.keys():
        
        print(this_date_block_id)

        # Extracting the features and labels for the current date+block pair.
        this_date_block_features = features_dict[this_date_block_id].values
        this_date_block_labels   = labels_dict[this_date_block_id].values

        # Extracting the unique classes in the labels array.
        unique_classes = np.unique(this_date_block_labels).tolist()

        # Initializing a dictionary that will hold the total number of samples per class.
        n_samples_per_class = {}

        # Iterating across all unique classes.
        for this_class in unique_classes:

            # Counting the number of samples for the current class.
            n_samples_this_class = np.argwhere(this_date_block_labels == this_class).shape[0]

            # Updating the dictionary with the number of samples for the current class.
            n_samples_per_class[this_class] = n_samples_this_class

        # Minimum number of samples per class.
        n_samples_min = n_samples_per_class[min(n_samples_per_class)]

        # Initializing a dictionary which will holds equal numbers of indices per class, with all indices
        # randomly chosen.
        downsampled_inds_per_class = {}

        # Iterating across all unique labels.
        for this_class in unique_classes:

            # Extracting the indices where the current class appears in the labels array.
            this_class_inds = np.squeeze(np.argwhere(this_date_block_labels == this_class))

            # Randomly downsampling the indices of the overrepresented class.
            inds_this_class_downsampled = sample(this_class_inds.tolist(), n_samples_min)

            # Sorting indices. Not really necessary, but helps when printing.
            inds_this_class_downsampled.sort()

            # Assigining the downsampled indices to the dictionary.
            downsampled_inds_per_class[this_class] = inds_this_class_downsampled

        # Computing the total number of samples with equal number of samples per class.
        n_unique_classes      = len(unique_classes)
        n_samples_downsampled = n_samples_min * n_unique_classes

        # Computing the total number of historical time points and features.
        n_history  = this_date_block_features.shape[0]
        n_features = this_date_block_features.shape[1]

        # Initializing the array of downsampled features and labels across all tasks.
        this_date_block_features_downsampled = np.zeros((n_history, n_features, n_samples_downsampled))
        this_date_block_labels_downsampled   = [None] * n_samples_downsampled

        # Initializing the index array which will be used to index the downsampled samples. 
        inds = np.zeros((2,))

        # Iterating across all unique classes.
        for this_class in unique_classes:

            # Updating the indices.
            inds = index_advancer(inds, n_samples_min)

            # Extracting the downsampled indices for the current class.
            this_class_inds = downsampled_inds_per_class[this_class]

            # Updating the downsampled features and labels.
            this_date_block_features_downsampled[:,:,inds[0]:inds[1]] = this_date_block_features[:,:,this_class_inds]
            this_date_block_labels_downsampled[inds[0]:inds[1]]       = [this_class] * n_samples_min


        # Converting the features and labels arrays into xarrays.
        this_date_block_features_downsampled = xr.DataArray(this_date_block_features_downsampled, 
                                                            coords={'history': np.arange(n_history), 'feature': np.arange(n_features), 'sample': np.arange(n_samples_downsampled)}, 
                                                            dims=["history", "feature", "sample"])

        this_date_block_labels_downsampled = xr.DataArray(this_date_block_labels_downsampled, 
                                                          coords={'sample': np.arange(n_samples_downsampled)}, 
                                                          dims=["sample"])

        # Updating the features and labels dictionaries with the downsampled corresponding arrays.
        features_dict[this_date_block_id] = this_date_block_features_downsampled
        labels_dict[this_date_block_id]   = this_date_block_labels_downsampled
    
    return features_dict, labels_dict





def evaluating_model_accuracy(fold_models, valid_data_folds, valid_labels_folds):
    """
    DESCRIPTION:
    Evaluating model accuracy by computing and displaying the confusion matrix of all predicted vs. true labels from
    all folds.
    
    INPUT VARIABLES:
    fold_models:        [dictionary (key: string (fold ID); Value: model)]; Models trained for each training fold.
    valid_data_folds:   [dict (key: string (fold ID); Value: xarray (dimensions vary based on model type) > floats)];
                        Data across all validation tasks per fold. Equal number of samples per class. PC features. Rearranged 
                        according to the type of model that will be trained.
    valid_labels_folds: [dict (key: string (fold ID); Value: xarray (1 x time samples) > strings (labels))]; Labels across
                        all validation tasks per fold. Equal number of labels per class.
    
    NECESSARY FUNCTIONS:
    computing_predicted_labels
    confusion_matrix_display
    """
    
    # COMPUTATION:
    
    # Across all folds, computing the predicted labels and extracting the true labels.
    pred_labels, true_labels = computing_predicted_labels(fold_models, valid_data_folds, valid_labels_folds)
    
    # Extracting the model classes.
    model_classes = np.sort(np.unique(true_labels)).tolist()
    
    # Computing the confusion matrix between rest and grasp.
    this_confusion_matrix = confusion_matrix(true_labels, pred_labels, labels = model_classes);
    
    # Plotting the confusion matrix
    confusion_matrix_display(this_confusion_matrix, model_classes)





def extracting_relevant_trajectories(data_dict, relevant_hand_landmarks):
    """
    DESCRIPTION:
    For each movement type, the experimenter enters the most relevant hand landmarks for training. The experimenter
    creates a relevant_hand_landmarks dictionary where the keys of the dictionary are the possible movement classes 
    and the value for each key is a list of the most relevant hand landmarks to that class. The plotting cells above
    should be used to determine these landmarks. Then for each movement type a dictionary, hand_trajectories_relevant
    is created where for each movement, only the relevant hand trajectories are stored.

    INPUT VARIABLES:
    data_dict: [dictionary (key: string (date+block ID); value: dictionary (key/value pairs below))];
        click_info: [dict (key: string ('backspace','keyboard','stimcolumn'); Values: below)];
            data:      [xarray (1 x time samples) > strings];  For each  time sample of the array of each key there
                       is a 'no_click' string or a click-string specific to that xarray. For example, the 'backspace'
                       key of the dictionary has an array where each element is a string named either 'no_click' or 
                       'backspace_click'. The 'backspace_click' elements do not occur consecutively and describe the 
                       instance a click on the backspace key occured. For the 'keyboard' and 'stimcolumn' keys, similar
                       rules apply. Time dimension is in units of s.
            plotcolor: [string]; Color corresponding to the type of click for plotting.
        onsetsoffsets: [list > list [t_onset, t_offset] > floats (units: s)]; The dictionary containing all
                       movement onset and offset times for each movement type.
        trajectories:  [xarray (landmarks x time samples) > floats]; The time traces of the x- and y-coordinates 
                       for each landmark. The time domain is in units of seconds. 
    relevant_hand_landmarks: [dictionary (key: string (movement type); Value: list > strings (hand landmarks))]; Each
                             movement holds a list of the most useful landmarks used to detect the corresponding 
                             movement type.
    
    OUTPUT VARIABLES:
    data_dict: Same as above with the additional key/value pair:
        trajectories_relevant: [dictionary (Key: string (movement type); Value: xarray (relevant landmarks x time samples)
                                > floats]; For each movement type, only the relevant hand trajectories are stored. The time
                                dimension of each xarray is in units of s.
    """
    
    # COMPUTATION:

    # Iterating across all date+block pairs of the data dictionary.
    for this_date_block_id in data_dict.keys():
        
        # Extracting the hand trajectories from the current date+block pair.
        these_hand_trajectories = data_dict[this_date_block_id]['trajectories']

        # Initializing the dictionary of relevant hand trajectories per movement for the current date+block pair.
        these_hand_trajectories_relevant = {}

        # Iterating across all movement types:
        for this_movement in relevant_hand_landmarks.keys():

            # Extracting the relevant landmarks for the current movement.
            this_movement_relevant_landmarks = relevant_hand_landmarks[this_movement]

            # Extracting only the trajectories of the relevant landmarks for the current movement.
            this_movement_hand_trajectories = these_hand_trajectories.loc[this_movement_relevant_landmarks,:]

            # Assigning the hand trajectorires specific to this movement to the dictionary.
            these_hand_trajectories_relevant[this_movement] = this_movement_hand_trajectories

        # Updating the date_dict with the relevant trajectories.
        data_dict[this_date_block_id]['trajectories_relevant'] = these_hand_trajectories_relevant
    
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

    # PRINTING
    # print('HAND TRAJECTORIES ARRAY')
    # print(hand_trajectories)
    
    # Printing all the hand landmarks.
    print('\nHAND LANDMARKS LIST:')
    pprint(list(hand_trajectories.landmarks.values))

    return hand_trajectories





def load_movement_onsetsoffsets(block_id, date, dir_intermediates, patient_id, task):
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
            
        # # Print the dictionary.
        # pprint(movement_onsetsoffsets)
        
    # The onsets/offsets dictionary does not exist in the specified pathway.
    else:
        print('Dictionary does not exist. Please create.')
    
    return movement_onsetsoffsets





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
    n_samples  = data.sample.shape[0]
    
    # Converting the data mean xarray to a regular array and repeating the means for each sample.
    data_mean = np.array(data_mean)
    data_mean = np.expand_dims(data_mean, axis=2)
    data_mean = np.tile(data_mean, (1, 1, n_samples))
    
    # Converting the data mean array back into an xarray
    data_mean = xr.DataArray(data_mean, 
                             coords={'history': np.arange(n_history), 'feature': np.arange(n_features), 'sample': np.arange(n_samples)}, 
                             dims=["history", "feature", "sample"])
    
    # Computing the mean-centered data.
    data_centered = data - data_mean
    
    return data_centered





def mean_centering_all_folds(data_folds, data_fold_means):
    """
    DESCRIPTION:
    Using the data mean to mean center the data at each fold.

    INPUT VARIABLES:
    data_folds:      [dict (key: string (fold ID); Value: xarray (time history x features x time samples) > floats)];
                     For each fold, feature xarrays are concatenated in the sample dimension.
    data_fold_means: [dict (Key: string (fold ID); Value: xarray (time history x features) > floats)]; The mean,
                     averaged across samples, for each fold. 
                     
    NECESSARY FUNCTIONS:
    mean_centering

    OUTPUT VARIABLES:
    data_folds_centered: [dict (key: string (fold ID); Value: xarray (time history x features x time samples) > floats)];
                         For each fold, the features are mean-centered according to means from the respective folds.
    """
    
    # COMPUTATION:
    
    # Initializing a dictionary with the mean-centered data for each fold.
    data_folds_centered = {}
    
    # Iterating across each fold.
    for this_fold in data_folds.keys():
        
        # Extracting the data and corresponding data means.
        this_fold_data       = data_folds[this_fold]
        this_fold_data_means = data_fold_means[this_fold]
        
        # Centering the data for the current fold.
        data_folds_centered[this_fold] = mean_centering(this_fold_data, this_fold_data_means)
        
    return data_folds_centered





def mean_compute(data):
    """
    DESCRIPTION:
    Computing the mean across all time samples for each feature. Note that though the data array has a history dimension, 
    only the first time history coordinate (time shift = 0) is used. As the mean is computed for potential PC reduction,
    the PCs will only be based off of the non-shifted features.
    
    INPUT VARIABLES:
    data: [xarray (history x features x samples) > floats]; Historical power features across time samples.
    
    OUTPUT VARIABLES:
    data_mean_history: [xarray (history x features) > floats ]; Mean power of each feature of only the 0th time shift.
                       This array is repeated for each historical time point.
    """
    # COMPUTATION:
    
    # Extracting the number of historical time points and features from task0. These number are the same across all tasks.
    n_features = data.feature.shape[0]
    n_history  = data.history.shape[0]
    
    # Extracting only the first historical time feature (no time shift). Eventually, in PC reduction, only this slice will 
    # be used to compute the eigenvectors.
    data = data.loc[0,:,:]
    
    # Computing the mean of the data across the time samples dimension.
    data_mean = np.asarray(data.mean(dim='sample'))
    
    # Creating an array of concatenated rows where each row is the data mean for each feature. There are as many rows as there
    # are historical time points. 
    data_mean_history = np.tile(data_mean, (n_history,1))

    # Converting the concatenated array into an xarray.
    data_mean_history = xr.DataArray(data_mean_history, 
                                     coords={'history': np.arange(n_history), 'feature': np.arange(n_features)}, 
                                     dims=["history", "feature"])
    
    return data_mean_history





def mean_compute_all_folds(data_folds):
    """
    DESCRIPTION:
    Computing the mean across all dimensions of each fold. Note that only the mean of the first historical time point is computed
    with the intent of only computing the PCs from those features. This mean will be repeated in an array for as many historical
    time features that there are.

    INPUT VARIABLES:
    data_folds: [dict (key: string (fold ID); Value: xarray (time history x features x time samples) > floats)]; For each fold, 
                feature xarrays are concatenated in the sample dimension.
                
    NECESSARY FUNCTIONS:
    mean_compute

    OUTPUT VARIABLES:
    data_fold_means: [dict (Key: string (fold ID); Value xarray (time history x features) > floats)]; The mean, averaged across
                     samples, for each fold. 
    """
    
    # COMPUTATION:
    
    # Initializing a dictionary of time-averaged data for each fold.
    data_fold_means = {}
    
    # Extracting the number of historical time points and features from task0. These numbers are the same across all tasks.
    n_features = data_folds['fold0'].feature.shape[0]
    n_history  = data_folds['fold0'].history.shape[0]
    
    # Iterating across each fold.
    for this_fold in data_folds.keys():
        
        # Extracting the data from the current fold.
        this_data = data_folds[this_fold]
                
        # Computing the mean of the current fold's data. The resulting array has concatenated rows where each row is the data mean
        # for each feature. There are as many rows as there are historical time points.
        this_data_mean_history = mean_compute(this_data)
        
        # Computing the mean of the data across the time dimension.
        data_fold_means[this_fold] = this_data_mean_history

    return data_fold_means





def model_training_lstm(training_data, training_labels, validation_data, validation_labels):
    """
    DESCRIPTION:
    Training and validating an LSTM.
    
    INPUT VARIABLES:
    training_data:     [xarray (history, features, training time samples] > floats]; 
    training_labels:   [xarray (1 x training samples) > strings (labels)]
    validation_data:   [xarray (history, features, validation time samples] > floats]; 
    validation_labels: [xarray (1 x validation samples) > strings (labels)]
    
    OUTPUT VARIABLES:
    model:         [classification model];
    model_classes: [list > strings]; Unique model classes.
    """
    
    # COMPUTATION
    
    # Extracting the hyperparameters.
    alpha         = 0.001
    batch_size    = 45
    dropout_rate  = 0.3
    epochs        = 2 # 10
    n_hidden_lstm = 25
    
    # Converting the training and validation data and labels from xarrays to arrays.
    training_data     = np.array(training_data)
    training_labels   = np.array(training_labels)
    validation_data   = np.array(validation_data)
    validation_labels = np.array(validation_labels)
    
    # Extracting the model classes and sorting them in a list.
    model_classes = np.sort(np.unique(validation_labels)).tolist()
    
    # Computing the number of classes.
    n_model_classes = np.unique(validation_labels).shape[0]
    
    # Extracting the number of training samples, time points, and features from the training data in current fold.
    n_samples_train = training_data.shape[0]
    n_time_history  = training_data.shape[1]
    n_features      = training_data.shape[2]
    
    # Extracting the total number of validation samples.
    n_samples_val = validation_data.shape[0]
    
    # Initializing overall model parameters.
    initializer = initializers.he_normal()
    model       = Sequential()
    
    # Creating the model architecture
    model.add(LSTM(n_hidden_lstm, input_shape = (n_time_history, n_features), activation = 'tanh', recurrent_activation = 'sigmoid',\
                   kernel_initializer = initializer, dropout = dropout_rate, recurrent_dropout = dropout_rate, return_sequences = True,\
                   unroll = True))
    model.add(Flatten())
    model.add(Dense(10, activation = 'elu', kernel_initializer = initializer))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_model_classes, activation = 'softmax', kernel_initializer = initializer))

    # Categorical cross-entropy for computing the error between true and predicted labels of each batch and updating the weights using
    # adaptive moment optimization (Adam optimizer)
    opt     = Adam(learning_rate = alpha)
    loss_fn = 'categorical_crossentropy' 
    model.compile(loss = loss_fn, optimizer = opt, metrics = ['accuracy'])
    print(model.summary())
        
    
    # Extracting the number of training and validation samples in this fold.
    n_training_samples   = training_labels.shape[0]
    n_validation_samples = validation_labels.shape[0]
    
    # Initializing the one-hot class training and validation labels.
    training_labels_1hot   = np.zeros((n_training_samples, n_model_classes))
    validation_labels_1hot = np.zeros((n_validation_samples, n_model_classes))
    

    # Iterating across all the training time samples.
    for n in range(n_training_samples):
        
        # Extracting the current class and class index.
        this_class       = training_labels[n]
        this_class_index = model_classes.index(this_class)
        
        # Creating the one-hot label for the current training sample.
        training_labels_1hot[(n, this_class_index)] = 1

    # Iterating across all the validation time samples.
    for n in range(n_validation_samples):
        
        # Extracting the current class and class index.
        this_class       = validation_labels[n]
        this_class_index = model_classes.index(this_class)
        
        # Creating the one-hot label for the current training sample.
        validation_labels_1hot[(n, this_class_index)] = 1
        
                
    # Updating the history of the model with each batch.
    history = model.fit(training_data, training_labels_1hot, epochs = epochs, batch_size = batch_size,\
                        validation_data = (validation_data, validation_labels_1hot)) # , verbose=0
    
    # Summary plot for accuracy.
    fig = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # Summary plot for loss.
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
        
    return model





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
    
    # Extracting the number of historical time shifts and number of samples from the data.
    n_history = data.history.shape[0]
    n_samples = data.sample.shape[0]
    
    # Computing the number of PC features.
    n_features_pc = eigenvectors.shape[1]
    
    # Initializing an xarray of PC data.
    data_pc = xr.DataArray((np.zeros((n_history, n_features_pc, n_samples))),
                            coords={'history': np.arange(n_history), 'feature': np.arange(n_features_pc), 'sample': np.arange(n_samples)},
                            dims=["history", "feature", "sample"])
    
    # Iterating across all historical time shifts and multiplying the data at each historical time shift by the same
    # eigenvectors.
    for n in range(n_history):
        
        # Extracting the data at the nth time shift.
        this_history_data = np.asarray(data.loc[n,:,:])
        
        # Transforming hte data of the nth time shift to PC space.
        data_pc.loc[n,:,:] = np.matmul(this_history_data.transpose(), eigenvectors).transpose()   
    
    return data_pc





def pc_transform_all_folds(n_pc_thr, percent_var_thr, train_data_folds, valid_data_folds):
    """
    DESCRIPTION: 
    Transforming the training and validation data into PC space for each fold. The eigenvectors for the PC transformation
    are computed from the 0th point in time history. For example consider an array of dimensions (h x f x t) where h, f
    and t correspond to the historical time sequence, number of features and number of time samples, respectively. 
    Using a historical time features of the last 1 second with 100 ms shifts, this means that h can be any integer from
    0 to 9. The PCs are computed from the 0th time history coordinate of the data array: (0 x f x t). This can be 
    dimensionally reduced to an array of dimensions (f x t). The resulting PCs of dimensions (p x t) where p <= f are
    applied to the (f x t) array at each historical time point. 
    
    The number of eigenvectors are selected based on one of two experimenter-specified criteria:
    
    1) Only keep the first eiggenvectors that that in total explain less than or equal variance to percent_var_thr
       ... or ... 
    2) Only keep the first n_pc_thr eigenvectors.
    

    INPUT VARIABLES:                        
    n_pc_thr:         [int]; The number of principal components to which the user wishes to reduce the data set. Set to 'None' if
                      percent_var_thr is not 'None', or set to 'None' along with percent_var_thr if all of the variance will be used
                      (no PC transform).
    percent_var_thr:  [float]; The percent variance which the user wishes to capture with the principal components. Will compute the
                      number of principal components which capture this explained variance as close as possible, but will not surpass
                      it. Set to 'None' if n_pc_thr is not 'None', or set to 'None' along with n_pc_thr if all of the variance will be
                      used (no PC transform).
    train_data_folds: [dict (key: string (fold ID); Value: xarray (time history x features x time samples) > floats)];
                      The training data across all training tasks for each fold. 
    valid_data_folds: [dict (key: string (fold ID); Value: xarray (time history x features x time samples) > floats)];
                      The validation data across all training tasks for each fold. 
    
    NECESSARY FUNCTIONS:
    computing_eigenvectors
    pc_transform

    OUTPUT VARIABLES: 
    train_data_folds  [dict (Key: string (fold ID); Value: xarray (history x pc features x time samples) > floats (units: PC units)];
                      Reduced dimensionality version of the training data arrays.
    valid_data_folds: [dict (Key: string (fold ID); Value: xarray (history x pc features x time samples) > floats (units: PC units)];
                      Reduced dimensionality version of the validation data arrays.
    """
    
    # COMPUTATION:
    
    # Iterating across each fold in the training data.
    for this_fold in train_data_folds.keys():
        
        # Extracting the training and validation data from the current fold.
        this_fold_train_data = train_data_folds[this_fold]
        this_fold_valid_data = valid_data_folds[this_fold]
        
        # Extracting only the training data from this fold corresponding to the 0th historical time shift.
        train_data_history0 = np.asarray(this_fold_train_data.loc[0,:,:])
        
        # Computing the eigenvectors for the current fold, using only the historical features corresponding to the 0th shift.
        this_fold_eigenvectors = computing_eigenvectors(train_data_history0, n_pc_thr, percent_var_thr)
        
        # Replacing the training and validation data with the PC transformed for the current fold.
        train_data_folds[this_fold] = pc_transform(this_fold_train_data, this_fold_eigenvectors)
        valid_data_folds[this_fold] = pc_transform(this_fold_valid_data, this_fold_eigenvectors)
    
    return train_data_folds, valid_data_folds





def plotting_landmarks_and_clicks(block_id, date, data_dict, landmark_trajectories_plotting):
    """
    DESCRIPTION:
    Plotting the experimenter-specified hand landmarks and the click information across the entirety of
    experimenter-specified date and block.
    
    INPUT VARIABLES:
    block_id: [String (BlockX, where X is an int))]; Block ID of the task whose click, hand trajectory and movement
              onset and offset will be plotted.
    date:     [string (YYYY_MM_DD)]; Date on which the block was run.
    data_dict: [dictionary (key: string (date+block ID); value: dictionary (key/value pairs below))];
        click_info: [dict (key: string ('backspace','keyboard','stimcolumn'); Values: below)];
            data:      [xarray (1 x time samples) > strings];  For each  time sample of the array of each key there
                       is a 'no_click' string or a click-string specific to that xarray. For example, the 'backspace'
                       key of the dictionary has an array where each element is a string named either 'no_click' or 
                       'backspace_click'. The 'backspace_click' elements do not occur consecutively and describe the 
                       instance a click on the backspace key occured. For the 'keyboard' and 'stimcolumn' keys, similar
                       rules apply. Time dimension is in units of s.
            plotcolor: [string]; Color corresponding to the type of click for plotting.
        onsetsoffsets: [list > list [t_onset, t_offset] > floats (units: s)]; The dictionary containing all
                       movement onset and offset times for each movement type.
        trajectories:  [xarray (landmarks x time samples) > floats]; The time traces of the x- and y-coordinates 
                       for each landmark. The time domain is in units of seconds. 
    landmark_trajectories_plotting: [list > strings]; Possible landmarks to display.
    """
    
    # PLOTTING
    
    # Creating the date+block key.
    date_block_id = date + '_' + block_id
    
    # Extracting the click and hand trajectory information.
    click_info        = data_dict[date_block_id]['click_info']
    hand_trajectories = data_dict[date_block_id]['trajectories']
    

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
    n_samples  = data.sample.shape[0]
    
    # If the model type is a SVM.
    if model_type == 'SVM':
        
        # NOTE: This script doesn't have a SVM model structure for training. Feel free to write one. The data is rearranged
        # for it.

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
    

    
    
    
def rearranging_features_all_folds(features_dict, model_type):
    """
    DESCRIPTION:
    Depending on the experimenter-specified model type the features array will be rearranged corresponding to the
    dimensions required for model.fit() 
    
    INPUT VARIABLES:
    model_type:    [string ('SVM','LSTM')]; The model type that will be used to fit the data.
    features_dict: [dictionary (Key: string (fold ID); Value: xarray (time history x features x time samples) > floats)]
                   Array of historical time features. Time samples reduced such that there are an equal number of features
                   per class.
                   
    NECESSARY FUNCTIONS:
    rearranging_features
    
    OUTPUT VARIABLES:
    features_dict: [dictionary (key: string (fold ID): Value xarray (dimensions vary based on model type) > floats)];
                   Rearranged features for each fold. 
    """
    
    # COMPUTATION:
    
    # Iterating across each task.
    for this_fold in features_dict.keys():
        
        # Extracting the current task's features from the feature_dict.
        this_task_features = features_dict[this_fold]
        
        # Updating the features dictionary with the re-arranged features.
        features_dict[this_fold] = rearranging_features(this_task_features, model_type)
    
    return features_dict





def referencing_hand_trajectories(data_dict, ref1_x, ref2_x, refa_y, refb_y):
    """
    DESCRIPTION:
    Each hand landmark is referenced according to experimenter-specified landmarks. Make sure that the landmarks that are
    selected will not be used for further analysis as they will get normalized out to 0.
    
    INPUT VARIABLES:
    data_dict:         [dictionary (key: string (date+block ID); value: dictionary (relevant key/value pairs below))];
        trajectories:  [xarray (landmarks x time samples) > floats]; The time traces of the x- and y-coordinates 
                       for each landmark. The time domain is in units of seconds. 
    ref1_x:            [string]; First horizontal reference landmark
    ref2_x:            [string]; Second horizontal reference landmark
    refa_y:            [string]; First vertical reference landmark
    refb_y:            [string]; Second vertical reference landmark
    
    OUTPUT VARIABLES:
    data_dict:         [dictionary (key: string (date+block ID); value: dictionary (relevant key/value pairs below))];
        trajectories:  [xarray (landmarks x time samples) > floats]; The trajectories of the x- and y-coordinates for each
                       landmark. These are referenced in the x- and y-dimensions according to the reference landmarks. The
                       time domain is in units of seconds. 
    """
    
    # COMPUTATION:
    
    # Iterating across all date+block pairs of the data dictionary.
    for this_date_block_id in data_dict.keys():
        
        # Extracting the hand trajectories from the current date+block pair.
        hand_trajectories = data_dict[this_date_block_id]['trajectories']

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
            
        # Assigning the referenced hand trajectories back to the corret date+block pair.
        data_dict[this_date_block_id]['trajectories'] = hand_trajectories_norm
    
    return data_dict





def save_info(dictionary, directory, filename):
    """
    DESCRIPTION:
    Saving information from the dictionary in the directory under the filename.
    """
    
    # Creating the pathway where the contents of the dictionary will be saved.
    path = directory + filename
    
    # If the path doesn't exist, create it.
    if not os.path.exists(path):
        # print('HERE')
        os.makedirs(path)
    
    # Iterating across each item in the dictionary.
    for pair in dictionary.items():
        
        # Extracting the key and value of the current item. 
        key   = pair[0]
        value = pair[1]
        
        # Saving the value under the name of the key in the appropriate path.
        with open(path + '/' + key, 'wb') as (fp): pickle.dump(value, fp)


        
        
        
def save_model(directory, filename, model, model_type):
    """
    DESCRIPTION:
    Saving the model in the directory under the filename.
    """
    
    # COMPUTATION:
    
    # Creating the pathway where the contents of the dictionary will be saved.
    path = directory + filename
    
    # If the path doesn't exist, create it.
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Creating the model pathway.
    model_path = path + '/Model'

    # If saving an LSTM model.
    if model_type == 'LSTM':

        # Saving the model.
        tf.keras.models.save_model(model, model_path)          
        




def time_history_sample_adjustment(features_dict, fps, labels_dict, t_history):
    """
    DESCRIPTION:
    Adjusting the time dimension of the labels and features. Due to the time history, the time-shifted rows of the features
    arrays are zero padded. As such, all columns with leading zeros should be removed. If there are N time shfited columns,
    this means that N-1 columns should be removed. For example, consider N = 3:

    Features array:

    historical time shifts
        n=2 shifts      [[0.000, 0.000, 0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101],
        n=1 shift        [0.000, 0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521],
        n=0 shifts       [0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521, 0.882]]
                            t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7    t=8    t=9     (time samples)   

        Labels:             l0     l1     l2     l3     l4     l5     l6     l7     l8     l9

    After curtailing, the features and labels arrays would look like:

    historical time shifts
        n=2 shifts      [[0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101],
        n=1 shift        [0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521],
        n=0 shifts       [0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521, 0.882]]
                            t=2    t=3    t=4    t=5    t=6    t=7    t=8    t=9     (time samples)   

        Labels:             l2     l3     l4     l5     l6     l7     l8     l9

    where time points and labels will then be re-adjusted.

    INPUT VARIABLES:
    features_dict: [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats)]; Array of 
                   historical time features.
    fps:           [int (30 or 60)]; Frames per second of of the video feed.
    labels_dict:   [dictionary (Key: string (task ID); Value: xarray > strings (labels))]; For each time frame in each
                   block, there exists a rest or movement-type label depending on the experimenter-specified onset and
                   offset of attempted movements.
    t_history:     [float (unit: s)]; Amount of feature time history.

    OUTPUT VARIABLES:
    features_dict: [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats)]; Array of 
                   historical time features.  Number of time samples corresponding to time history are curtailed at the beginning
                   of the time array.
    labels_dict:   [dictionary (Key: string (task ID); Value: xarray > strings (labels))]; For each time frame in each
                   block, there exists a rest or movement-type label depending on the experimenter-specified onset and
                   offset of attempted movements. Number of time samples corresponding to time history are
                   curtailed at the beginning of the time array.
    """
    
    # COMPUTATION:
    
    # Iterating across each task.
    for this_task in features_dict.keys():
        
        # Computing the frame duration given the FPS.
        t_frame = (1/fps) * 1000 # 1/(/s) * ms/s = s * ms/s = ms
        
        # Extracting the features and labels for the current task.
        this_task_features = features_dict[this_task]
        this_task_labels   = labels_dict[this_task]
        
        # Computing the number of samples corresponding to the features.
        n_history = int(t_history/t_frame)
        
        # Extracting the features and labels corresponding to only the time samples after the first n_history samples.
        this_task_features_curt = this_task_features[:,:,n_history:]
        this_task_labels_curt   = this_task_labels[n_history:]

        # Updating the features and labels dictionary.
        features_dict[this_task] = this_task_features_curt
        labels_dict[this_task]   = this_task_labels_curt
    
    return features_dict, labels_dict 





def training_fold_models(model_type, train_data_folds, train_labels_folds, valid_data_folds, valid_labels_folds):
    """
    DESCRIPTION:
    This is only to be used to create a confusion matrix across all validation data folds using models trained on the 
    correpsonding training data folds. Used to assess classification performance before training a final model on all
    the data.

    INPUT VARIABLES:
    model_type:         [string ('SVM','LSTM')]; The model type that will be used to fit the data.
    train_data_folds:   [dict (key: string (fold ID); Value: xarray (dimensions vary based on model type) > floats)];
                        Data across all training tasks per fold. Equal number of samples per class. PC features. Rearranged 
                        according to the type of model that will be trained.
    train_labels_folds: [dict (key: string (fold ID); Value: xarray (1 x time samples) > strings (labels))]; Labels across
                        all training tasks per fold. Equal number of labels per class.
    valid_data_folds:   [dict (key: string (fold ID); Value: xarray (dimensions vary based on model type) > floats)];
                        Data across all validation tasks per fold. Equal number of samples per class. PC features. Rearranged 
                        according to the type of model that will be trained.
    valid_labels_folds: [dict (key: string (fold ID); Value: xarray (1 x time samples) > strings (labels))]; Labels across
                        all validation tasks per fold. Equal number of labels per class.
                             
    NECESSARY FUNCTIONS:
    model_training_lstm
    model_training_svm
    
    OUTPUT VARIABLES:
    fold_models: [dictionary (key: string (fold ID); Value: model)]; Models trained for each training fold.
    """
    # COMPUTATION:
    
    # Creating a dictionary to contain models for all validation folds.
    fold_models = {}
    
    # Iterating across each task.
    for this_fold in train_data_folds.keys():
        
        # Extracting the training and validation data and labels from the current fold.
        this_fold_train_data   = train_data_folds[this_fold]
        this_fold_train_labels = train_labels_folds[this_fold]
        this_fold_valid_data   = valid_data_folds[this_fold]
        this_fold_valid_labels = valid_labels_folds[this_fold]
        
        # If the model type is na SVM. Currently code for training SVM does not exist. Feel free to write a model_training_svm
        # function.
        if model_type == 'SVM':
            pass
        
        # If the model type is an LSTM
        if model_type == 'LSTM':
            
            # Creating a LSTM model for the current fold of training data.
            this_fold_model = model_training_lstm(this_fold_train_data, this_fold_train_labels,\
                                                  this_fold_valid_data, this_fold_valid_labels)
        
        # Updating the fold models dictionary with the current fold model.
        fold_models[this_fold] = this_fold_model
            
    return fold_models





def training_final_model(features_dict, labels_dict, model_type, n_pc_thr, percent_var_thr):
    """
    DESCRIPTION:
    Concatenating the data and labels from all tasks and training the final model on these concatenated arrays.

    INPUT VARIABLES:
    features_dict:   [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats )]
                     Array of historical time features. Time samples reduced such that there are an equal number of 
                     features per class.
    labels_dict:     [dictionary (Key: string (task ID); Value: xarray > strings (labels))]; For each time sample in 
                     each task, there exists a rest or grasp label depending on the experimenter-specified onset and
                     offset of modulation as well as the per-trial shift from the AW model. Time samples reduced such
                     that there are an equal number of features per class.
    model_type:      [string ('SVM','LSTM')]; The model type that will be used to fit the data.
    n_pc_thr:        [int]; The number of principal components to which the user wishes to reduce the data set. Set to
                     'None' if percent_var_thr is not 'None', or set to 'None' along with percent_var_thr if all of the
                     variance will be used (no PC transform).
    percent_var_thr: [float]; The percent variance which the user wishes to capture with the principal components. Will
                     compute the number of principal components which capture this explained variance as close as
                     possible, but will not surpass it. Set to 'None' if n_pc_thr is not 'None', or set to 'None' along
                     with n_pc_thr if all of the variance will be used (no PC transform).

    NECESSARY FUNCTIONS:
    computing_eigenvectors
    concatenating_all_data_and_labels
    mean_centering
    mean_compute
    model_training_lstm
    pc_transform
    rearranging_features

    OUTPUT VARIABLES:
    eigenvectors_final: [array (features x pc features) > floats]; Array in which columns consist of eigenvectors which
                        explain the variance of the data in descending order. 
    final_model:        [classification model]; Model trained with data from all tasks.
    model_classes:      [list > strings]; Class labels for the confusion matrix.
    training_data_mean: [xarray (history x features) > floats ]; Mean power of each feature of only the 0th time shift.
                        This array is repeated for each historical time point.
    """
    
    # COMPUTATION:
        
    # Concatenating the data and labels from all tasks for training.
    training_data, training_labels = concatenating_all_data_and_labels(features_dict, labels_dict)
    
    # Extracting the unique model classes.
    model_classes = np.sort(np.unique(training_labels)).tolist()
    
    # Computing the mean of the training data.
    training_data_mean = mean_compute(training_data)
    
    # Mean-centering the training data.
    training_data = mean_centering(training_data, training_data_mean)
    
    # Extracting only the training data corresponding to the 0th historical time shift.
    training_data_history0 = np.asarray(training_data.loc[0,:,:])

    # Computing the eigenvectors, using only the historical features corresponding to the 0th shift.
    eigenvectors = computing_eigenvectors(training_data_history0, n_pc_thr, percent_var_thr)
    
    # Computing the reduced-dimension PC training data.
    training_data = pc_transform(training_data, eigenvectors)

    # Rearranging features to fit the appropriate model type.
    training_data = rearranging_features(training_data, model_type)
    
    # If the model type is an LSTM
    if model_type == 'LSTM':

        # Creating a LSTM model for the current fold of training data. The third and fourth inputs in this
        # function take the spot of validation data and labels, but are meaningless here. Similarly, the 
        # validation accuracy should be ignored.
        final_model = model_training_lstm(training_data, training_labels, training_data, training_labels)
    
    return eigenvectors, final_model, model_classes, training_data_mean





def training_validation_split_tasks(features_dict):
    """
    DESCRIPTION:
    Creating dictionaries of training and validation folds where the tasks are split up per-fold.

    INPUT VARIABLES:
    features_dict: [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats (units: V^2/Hz))]
                   Array of historical time features. Time samples reduced such that there are an equal number of features per
                   class.

    OUTPUT VARIABLES:
    training_folds_tasks    [dict (key: string (fold ID); Value: list > strings (task IDs))]; List of all the training tasks
                            for each training fold.
    validation_folds_tasks: [dict (key: string (fold ID); Value: list > strings (task IDs))]; List of all the validation tasks
                            for each validation fold.
    """
    
    # COMPUTATION:
    
    # Initializing dictionaries for training and validation folds.
    train_folds_tasks = {}
    valid_folds_tasks = {}

    # Extracting a list of all the tasks.
    task_list = list(features_dict.keys())

    # Computing the number of folds
    n_folds = len(task_list)

    # Iterating across all folds.
    for f, this_task in enumerate(task_list):

        # Initializing the lists of training and validation tasks for the current fold.
        this_fold_training_tasks   = []
        this_fold_validation_tasks = []

        # Updating the list of validation tasks for the current fold with just the current task.
        this_fold_validation_tasks.append(this_task)

        # Updating the list of training tasks for the current fold with all other tasks aside from the current task.
        this_fold_training_tasks = [x for x in task_list if x != this_task]

        # Creating a fold ID
        fold_id = 'fold' + str(f)

        # Updating the training and validation tasks for the current training and validation folds.
        train_folds_tasks[fold_id] = this_fold_training_tasks
        valid_folds_tasks[fold_id] = this_fold_validation_tasks
    
    return train_folds_tasks, valid_folds_tasks





def training_validation_split(features_dict, labels_dict, train_folds_tasks, valid_folds_tasks):
    """
    DESCRIPTION:
    Splitting the data into training and validation blocks for building the fold-models. The validation folds will be tested
    on the models built on the corresponding training folds to confirm that the decoder is generalizable to unseen data. 

    INPUT VARIABLES:
    features_dict:     [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats (units: V^2/Hz))]
                       Array of historical time features. Time samples reduced such that there are an equal number of features per
                       class.
    labels_dict:       [dictionary (Key: string (task ID); Value: xarray > strings ('grasp'/'rest'))]; For each time sample in each
                       task, there exists a rest or grasp label depending on the experimenter-specified onset and offset of
                       modulation as well as the per-trial shift from the AW model. Time samples reduced such that there are an equal
                       number of features per class.
    train_folds_tasks  [dict (key: string (fold ID); Value: list > strings (task IDs))]; List of all the training tasks
                       for each training fold.
    valid_folds_tasks: [dict (key: string (fold ID); Value: list > strings (task IDs))]; List of all the validation tasks
                       for each validation fold.

    OUTPUT VARIABLES:
    train_data_folds:   [dict (key: string (fold ID); Value: xarray (time history x features x time samples) > floats (units: V^2/Hz))];
                        For each training fold, feature xarrays are concatenated in the sample dimension.
    train_labels_folds: [dict (key: string (fold ID); Value: xarray (1 x time samples) > strings ('grasp'/'rest'))]; For each training
                        fold, label xarrays are concatenated in the sample dimension.
    valid_data_folds:   [dict (key: string (fold ID); Value: xarray (time history x features x time samples) > floats (units: V^2/Hz))];
                        For each validation fold, feature xarrays are concatenated in the sample dimension.
    valid_labels_folds: [dict (key: string (fold ID); Value: xarray (1 x time samples) > strings ('grasp'/'rest'))]; For each validation
                        fold, label xarrays are concatenated in the sample dimension.
    """
    
    # COMPUTATION:
    
    # Initializing dictionaries of training and validation data and labels.
    train_data_folds   = {}
    train_labels_folds = {}
    valid_data_folds   = {}
    valid_labels_folds = {}

    # Extracting the fold list.
    fold_list = list(train_folds_tasks.keys())

    # Iterating across all folds.
    for fold_id in fold_list:

        # Extracting the training and validation task lists for the current fold.
        this_fold_training_tasks   = train_folds_tasks[fold_id]
        this_fold_validation_tasks = valid_folds_tasks[fold_id]

        # Initialize the training and validation task flags, which will help with initializing the arrays of 
        # training and validation data and labels for the current fold.
        training_task0_flag   = True
        validation_task0_flag = True
        
        # Iterating across all training tasks for the current fold.
        for this_task in this_fold_training_tasks:

            # Extracting the training data and labels of the current task.
            this_task_data   = features_dict[this_task]
            this_task_labels = labels_dict[this_task]

            # If the training task flag is True, intiailize the training data and labels xarrays. If not, concatenate
            # them with data and labels from another task.
            if training_task0_flag:
                these_training_data   = this_task_data
                these_training_labels = this_task_labels

                # Setting the flag to False to never enter this IF statement again.
                training_task0_flag = False

            else:
                these_training_data   = xr.concat([these_training_data, this_task_data], dim="sample")
                these_training_labels = xr.concat([these_training_labels, this_task_labels], dim='sample')
        
        # Iterating across all validatoin tasks for the current fold.
        for this_task in this_fold_validation_tasks:

            # Extracting the training data and labels of the current task.
            this_task_data   = features_dict[this_task]
            this_task_labels = labels_dict[this_task]

            # If the validation task flag is True, intiailize the validation data and labels xarrays. If not, concatenate
            # them with data and labels from another task.
            if validation_task0_flag:
                these_validation_data   = this_task_data
                these_validation_labels = this_task_labels

                # Setting the flag to False to never enter this IF statement again.
                validation_task0_flag = False

            else:
                these_validation_data   = xr.concat([these_validation_data, this_task_data], dim="sample")
                these_validation_labels = xr.concat([these_validation_labels, this_task_labels], dim='sample')
                
                
        # if this_fold_training_tasks:
        
        # Reassigning the sample coordinates to the training data and labels xarrays.
        these_training_data     = these_training_data.assign_coords(sample=np.arange(these_training_data.sample.shape[0]))
        these_training_labels   = these_training_labels.assign_coords(sample=np.arange(these_training_labels.sample.shape[0]))

        # Updating the training data and labels dictionaries with the appropriate training information.
        train_data_folds[fold_id]   = these_training_data
        train_labels_folds[fold_id] = these_training_labels

      
        # if this_fold_validation_tasks:
            
        # Reassigning the sample coordinates to the validation data and labels xarrays.
        these_validation_data   = these_validation_data.assign_coords(sample=np.arange(these_validation_data.sample.shape[0]))
        these_validation_labels = these_validation_labels.assign_coords(sample=np.arange(these_validation_labels.sample.shape[0]))

        # Updating the validation data and labels dictionaries with the appropriate validation information.
        valid_data_folds[fold_id]   = these_validation_data
        valid_labels_folds[fold_id] = these_validation_labels
        
    return train_data_folds, train_labels_folds, valid_data_folds, valid_labels_folds





def zooming_in(block_id, date, data_dict, movement_colors, t_end_zoom, t_start_zoom):
    """
    DESCRIPTION:
    The experimenter inputs a start and an end time between which to zoom in to view the relevant hand trajectories
    for each movement and click information for a specific date+block pair. The hand landmark trajectories are shown
    for each movement in a separate plot.
    
    INPUT VARIABLES:
    block_id:                  [String (BlockX, where X is an int))]; Block ID of the task whose click, hand trajectory and
                               movement onset and offset will be plotted.
    date:                      [string (YYYY_MM_DD)]; Date on which the block was run.
    data_dict:                 [dictionary (key: string (date+block ID); value: dictionary (key/value pairs below))];
        click_info:            [dict (key: string ('backspace','keyboard','stimcolumn'); Values: below)];
            data:              [xarray (1 x time samples) > strings];  For each  time sample of the array of each key there
                               is a 'no_click' string or a click-string specific to that xarray. For example, the 'backspace'
                               key of the dictionary has an array where each element is a string named either 'no_click' or 
                               'backspace_click'. The 'backspace_click' elements do not occur consecutively and describe the 
                               instance a click on the backspace key occured. For the 'keyboard' and 'stimcolumn' keys, similar
                               rules apply. Time dimension is in units of s.
            plotcolor:         [string]; Color corresponding to the type of click for plotting.
        onsetsoffsets:         [list > list [t_onset, t_offset] > floats (units: s)]; The dictionary containing all
                               movement onset and offset times for each movement type.
        trajectories:          [xarray (landmarks x time samples) > floats]; The time traces of the x- and y-coordinates 
                               for each landmark. The time domain is in units of seconds. 
        trajectories_relevant: [dictionary (Key: string (movement type); Value: xarray (relevant landmarks x time samples)
                               > floats]; For each movement type, only the relevant hand trajectories are stored. The time
                               dimension of each xarray is in units of s.              
    movement_colors:           [dictionary (key: string (movement); Value: string (color))]; There is a color associated
                               with each movement for plotting.
    t_end_zoom:                [int (units: s)]; The ending time point for the zoomed in window. To set as the last time
                               point, leave as empty list [].
    t_start_zoom:              [int (units: s)]; The starting time point for the zoomed in window. To set as the first
                               time point, leave as empty list [].    
    """
    
    # COMPUTATION:
    
    # Creating the date+block key.
    date_block_id = date + '_' + block_id
    
    # Extracting the click information, relevant hand trajectories and movement onsets and offsets from the date+block pair.
    click_info                 = data_dict[date_block_id]['click_info']
    movement_onsetsoffsets     = data_dict[date_block_id]['onsetsoffsets']
    hand_trajectories_relevant = data_dict[date_block_id]['trajectories_relevant']
    
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
        ax1.set_title(this_movement + ' trajectories')
        ax1.grid()
        ax2.set_ylabel("Click Type",fontsize = 14, color = "black")
        ax2.tick_params(axis="y", colors = "black")
    
    return None