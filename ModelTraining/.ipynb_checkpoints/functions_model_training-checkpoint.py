

# IMPORTING LIBRARIES:
import collections
import copy
import numpy as np
import os
import pandas as pd
import params_model_training
import pickle
import saliency_mapping_suite
import seaborn as sns
import shutil
import tensorflow as tf
import xarray as xr

import matplotlib.pyplot as plt
import scipy.signal as signal

from affinewarp import PiecewiseWarping, ShiftWarping
from ecogconf import ECoGConfig
from h5eeg import H5EEGFile
from itertools import chain
from PIL import Image, ImageDraw
from random import sample 

from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam





############################ UPLOADING GLOBAL PARAMETERS #########################
def uploading_global_parameters():
    """
    DESCRIPTION:
    Importing all of the parameters which will become global to this script.
    
    GLOBAL PARAMETERS:
    aw_model_type:   [string ('shift'/'piecewise')]; The type of affine warp transformation the experimenter wishes to perform. 
                     Leave empty [] if no AW-alignment will occur.
    car:             [bool (True/False)] Whether or not CAR filtering will be performed.
    calib_state_val: [int]; The state value from where to extract the appropriate calibration data.
    f_power_max:     [list > int (units: Hz)]; For each frequency band, maximum power band frequency.
    f_power_min:     [list > int (units: Hz)]; For each frequency band, minimum power band frequency.
    file_extension:  [string (hdf5/mat)]; The data file extension of the data.
    model_classes:   [list > strings]; List of all the classes to be used in the classifier.
    model_name:      [string]; Name that describes what data the model is trained on.
    model_type:      [string ('SVM','LSTM')]; The model type that will be used to fit the data.
    n_pc_thr:        [int]; The number of principal components to which the user wishes to reduce the data set. Set to 'None' if percent_var_thr
                     is not 'None', or set to 'None' along with percent_var_thr if all of the variance will be used (no PC transform).
    patient_id:      [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    percent_var_thr: [float]; The percent variance which the user wishes to capture with the principal components. Will compute the number
                     of principal components which capture this explained variance as close as possible, but will not surpass it. Set to 'None'
                     if n_pc_thr is not 'None', or set to 'None' along with n_pc_thr if all of the variance will be used (no PC transform).
    sampling_rate:   [int (samples/s)]; Sampling rate at which the data was recorded.
    sxx_shift:       [int (units: ms)]; Length of time by which sliding window (sxx_window) shifts along the time domain.
    sxx_window:      [int (units: ms)]; Time length of the window that computes the frequency power.
    """
    
    # COMPUTATION:
    
    # Making each of the following parameters global variables.
    global aw_model_type
    global car
    global calib_state_val
    global f_power_max
    global f_power_min
    global file_extension
    global model_classes
    global model_name
    global model_type
    global n_pc_thr
    global patient_id
    global percent_var_thr
    global sampling_rate
    global sxx_shift
    global sxx_window
    
    # Importing the parameters. 
    aw_model_type   = params_model_training.aw_model_type
    car             = params_model_training.car
    calib_state_val = params_model_training.calib_state_val
    f_power_max     = params_model_training.f_power_max
    f_power_min     = params_model_training.f_power_min
    file_extension  = params_model_training.file_extension
    model_classes   = params_model_training.model_classes
    model_name      = params_model_training.model_name
    model_type      = params_model_training.model_type
    n_pc_thr        = params_model_training.n_pc_thr
    patient_id      = params_model_training.patient_id
    percent_var_thr = params_model_training.percent_var_thr
    sampling_rate   = params_model_training.sampling_rate
    sxx_shift       = params_model_training.sxx_shift
    sxx_window      = params_model_training.sxx_window
    
    # PRINTING THE GLOBAL PARAMETERS.
    print('GLOBAL PARAMETERS TO functions_click_detector_final_model_training.py SCRIPT')
    print('AW MODEL TYPE:   ', aw_model_type)
    print('CAR FILTERING:   ', car)
    print('CALIB STATE VAL: ', calib_state_val)
    print('F-BAND MAX VALS: ', f_power_max, 'Hz')
    print('F-BAND MIN VALS: ', f_power_min, 'Hz')
    print('FILE EXTENSION:  ', file_extension)
    print('MODEL CLASSES:   ', model_classes)
    print('MODEL NAME:      ', model_name)
    print('MODEL TYPE:      ', model_type)
    print('NUMBER OF PCs:   ', n_pc_thr)
    print('PATIENT ID:      ', patient_id)
    print('PERCENT PC VAR:  ', percent_var_thr)
    print('SAMPLING RATE:   ', sampling_rate, 'Sa/s')
    print('SPECTRAL SHIFT:  ', sxx_shift, 'ms')
    print('SPECTRAL WINDOW: ', sxx_window, 'ms')
    
# Immediately uploading global parameters.
uploading_global_parameters()








def averaging_channel_importance_scores(ch_importance_per_sample):
    """
    DESCRIPTION:
    Averaging the mean saliency for each channel across all validation folds.
    
    INPUT VARIABLES:
    ch_importance_per_sample: [dictionary (key: string (fold); Value: array (time samples x channels))]; For each validation fold,
                              the importance score for each channel is computed for each sample of the experimenter-specified
                              saliency class.
    
    OUTPUT VARIABLES:
    ch_importance_scores: [array (1 x channels) > floats]; Mean importance score for each channel, averaged across all
                          time samples from all validation folds.
    """
    # COMPUTATION:
    
    
    # Initializing flag to inform whether first fold.
    first_fold_flag = True

    # Iterating across all validation folds.
    for this_fold in ch_importance_per_sample.keys():

        # If this is the first fold, initialize the array of all channel importance scores. 
        if first_fold_flag:
            ch_importance_all_folds = ch_importance_per_sample[this_fold]

            # Setting the flag to False so that this IF statement isn't entered more than once.
            first_fold_flag = False

        # Otherwise, concatenate this array of channel importance scores.
        else:
            ch_importance_all_folds = np.concatenate((ch_importance_all_folds, ch_importance_per_sample[this_fold]), axis=0)

    # Taking the mean of all importance scores across all samples (over all validation folds).
    ch_importance_scores = np.mean(ch_importance_all_folds, axis=0)
    
    return ch_importance_scores





def aw_model_building(aw_powerband, chs_alignment, grasp_bandpower_dict, ptr_all_trials, t_aw_end, t_aw_start):
    """
    DESCRIPTION:
    Each trial of the power trial rasters will be shifted in time be some trial-specific value such that the inter-trial
    correlation increases. This shifting will occur because of the natural variability in delay between the state ON onset
    and onset of neural activity. For example, this variability may arise from the natural variation in the participant's
    reaction delay when seeing a cue to make an attempted grasp.
    
    An affine warp (aw) model (see https://github.com/ahwillia/affinewarp.git) is created to compute these shifts for each
    trial. The aw model is trained on multiple channels (qualitatively selected) and on on power band from these channels.
    The experimenter must also specify the time bounds within the power trial rasters by which this model will be trained.

    INPUT VARIABLES:
    aw_powerband:         [string ('powerbandX')]; Name of the powerband that will be used for aligning the power trial rasters.
                          X is an integer (0, 1, 2,...).
    chs_alignment:        [list > strings]; The list of channels which will be used for affine warp. Leave as [] if all
                          channels will be used.
    grasp_bandpower_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_power:        [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each frequency band,
                          the band power is computed for each channel across every time point.
        sxx_power_z:      [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each standardized (to
                          calibration) frequency band, the band power is computed for each channel across every time point.
        sxx_states:       [xarray (1 x time samples) > strings ('state_ON'/'state_OFF'/'neutral')]; Stimulus array downsampled
                          to match time resolution of the signal spectral power. Time dimension is in units of seconds.
        power_trials_z:   [xarray (trials x channels x frequency bins x time samples) > floats (units: V^2/Hz)]; The spectral
                          information for each trial. Time dimension is in units of seconds.
    ptr_all_trials:       [xarray (trials x channels x powerbands x time samples) > floats (units: V^2/Hz)]; The spectral information
                          for  each trial (across all tasks). Time dimension is in units of seconds.
    t_aw_end:             [float (units: s)]; The ending time boundary for the HG onset period on which AW will be applied.
    t_aw_start:           [float (units: s)]; The starting time boundary for the HG onset period on which AW will be applied.
    
    GLOBAL PARAMETERS:
    aw_model_type: [string ('shift'/'piecewise')]; The type of affine warp transformation the experimenter wishes to perform. 
                   Leave empty [] if no AW-alignment will occur.
    
    OUTPUT VARIABLES:
    aw_model:                  [Affinewarp model]; Created using all trials.
    aw_shifts:                 [dictionary (Key: string (Task ID); Value: dictionary (Key/Value pairs below)];
        N (where N is an int): [int]; AW shift for the Nth trial in units of samples.
    """
    
    # COMPUTATION:
        
    # If the experimenter wishes to align the power trial rasters.
    if aw_model_type:
        
        # Extracting the time array of the power trial rasters.
        t_ptr = ptr_all_trials.time.values
                
        # Extracting the indices of the time samples the AW model will use to perform per-trial re-alignment.
        aw_inds = np.logical_and(t_ptr > t_aw_start, t_ptr < t_aw_end)
        
        # In addition to extracting the specific power band and indicies of time samples used for alignment, if the experimenter
        # specified a specific set of channels to be used for computing per-trial alignment, extract the power trial rasters 
        # information from only those channels
        if chs_alignment:            
            aw_ptr = ptr_all_trials.loc[:,chs_alignment, aw_powerband, aw_inds].values
        
        # If the experimenter has not specified channels to be used to compute the per-trial shifts, extracting only the specific
        # power band and indices of time samples used for alignment.
        else:
            aw_ptr = ptr_all_trials.loc[:,:, aw_powerband, aw_inds].values
            

        # Flipping the channels and time dimensions for the affine warp function.
        aw_ptr = np.moveaxis(aw_ptr, -2, -1)  # Dimensions: trials x time samples x channels
    
        # Creating the affine warp model type.
        if aw_model_type == 'shift':
            aw_model = ShiftWarping(maxlag = 0.3, smoothness_reg_scale = 10.0)

        if aw_model_type == 'piecewise':
            aw_model = PiecewiseWarping(n_knots = 0, warp_reg_scale = 1e-6, smoothness_reg_scale = 20.0)

        # Fit the AW model to the data.
        aw_model.fit(aw_ptr, iterations = 20)    
        
        # Creating a dictionary of AW shifts for all tasks.
        aw_shifts = {}
        
        # Initializing the trial counter corresponding to all tasks.
        tr_counter_all_tasks = 0
        
        # Iterating across all tasks.
        for this_task in grasp_bandpower_dict.keys():
            
            # Extracting the trial indices of the current task.
            trial_inds_this_task = copy.deepcopy(grasp_bandpower_dict[this_task]['power_trials_z'].trial.values)
            
            # Extracting the number of trials in the current task.
            n_trials_this_task = trial_inds_this_task.shape[0]

            # Initializing a dictionary of the trial shifts for the current task.
            this_task_shifts = {}
                        
            # Updating the trial indices for the current task to reflect their position in the aw_model.shifts array.
            trial_inds_this_task += tr_counter_all_tasks
            
            # Initializing the trial counter specific only for this task.
            tr_counter_this_task = 0
            
            # Iterating across all the trial indices corresponding to the current task found in the aw_model.
            for tr in trial_inds_this_task:
                
                # Adding the trial shift from the current task
                this_task_shifts[tr_counter_this_task] = aw_model.shifts[tr]
                
                # Updating the trial counter for the current task.
                tr_counter_this_task += 1
                
            # Updating the dictionary of AW shifts.
            aw_shifts[this_task] = this_task_shifts
            
            # Updating the trial counter with the trials from the current task.
            tr_counter_all_tasks += n_trials_this_task
       
    # If the experimenter does not wish to align the power trial rasters.
    else:
        aw_model = None
        aw_shifts = None
            
    return aw_model, aw_shifts





def aligning_power_trial_rasters(aw_model, ptr_all_trials):
    """
    DESCRIPTION:
    Aligning the power trial rasters (across all channels and powerbands) according to the affine warp model.
    
    INPUT VARIABLES:
    aw_model:       [Affinewarp model]; Created using all trials.
    ptr_all_trials: [xarray (trials x channels x powerbands x time samples) > floats (units: V^2/Hz)]; The spectral information
                    for each trial (across all tasks). Time dimension is in units of seconds.
           
    GLOBAL PARAMETERS:
    aw_model_type: [string ('shift'/'piecewise')]; The type of affine warp transformation the experimenter wishes to perform. 
                   Leave empty [] if no AW-alignment will occur.
    
    OUTPUT VARIABLES:
    ptr_aligned_all_trials: [xarray (trials x channels x powerbands x time samples) > floats (units: V^2/Hz)]; The spectral
                            information for each aligned trial (across all tasks). Time dimension is in units of seconds.
    """
    
    # COMPUTATION:
    
    # Extracting the trials, channels, powerbands, and time samples from the power trial rasters array.
    trials          = ptr_all_trials.trial.values
    channels        = ptr_all_trials.channel.values
    powerbands      = ptr_all_trials.powerband.values
    t_sxx_per_trial = ptr_all_trials.time.values
            
    # Extracting the number of trials, channels, powerbands, and time samples from the power trial rasters array.
    n_trials_all        = trials.shape[0]
    n_chs               = channels.shape[0]
    n_bands             = powerbands.shape[0]
    n_samples_per_trial = t_sxx_per_trial.shape[0]
        
    # Initializing an array of aligned power trial rasters.
    ptr_aligned_all_trials = xr.DataArray(np.zeros((n_trials_all, n_chs, n_bands, n_samples_per_trial)), 
                                          coords={'trial': np.arange(n_trials_all),'channel': channels, 'powerband': powerbands, 'time': t_sxx_per_trial}, 
                                          dims=["trial", "channel", "powerband", "time"])
    
    # Iterating over all powerbands.
    for powerband_id in powerbands:
        
        # Extracting the unaligned power trial rasters for the current power band.
        this_pwrband_ptr = ptr_all_trials.loc[:,:,powerband_id,:].values
        
        # Flipping the channels and time dimensions for the AW model.
        this_pwrband_ptr = np.moveaxis(this_pwrband_ptr, -2, -1)  # Dimensions: trials x time samples x channels
                
        # If an affine warp model was used, align the power trial rasters of the current power band appropriately.
        if aw_model_type:
        
            # Aligning the power trial rasters of the current band according to the AW model.
            this_pwrband_ptr_aligned = aw_model.transform(this_pwrband_ptr)
                    
        # If no affine warp model was used, keep the aligned power trial rasters the same as the unaligned power trial rasters.
        else:
            this_pwrband_ptr_aligned = this_pwrband_ptr
                    
        # Flipping the channels and time dimension back again for the aligned power trial rasters xarray.
        this_pwrband_ptr_aligned = np.moveaxis(this_pwrband_ptr_aligned, -2 ,-1) # Dimensions: trials x channels x time samples
        
        # Assigning the aligned power trial rasters array according to the appropriate powerband ID.
        ptr_aligned_all_trials.loc[:,:,powerband_id,:] = this_pwrband_ptr_aligned

    return ptr_aligned_all_trials





def calib_signals_relevant(calib_cont_dict):
    """
    DESCRIPTION:
    Only the signals and states where the state array equals the calibration state value are extracted.
    
    INPUT VARIABLES:
    calib_cont_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        signals:     [xarray (channels x time samples (units: s) > floats (units: microvolts)]; Array of continuous
                     voltage signals from the calibration tasks. Time dimension is in units of seconds.
        states:      [xarray (1 x time samples) > ints (0 or 1)]; Array of states at each time sample for the calibration
                     tasks. Time dimension is in units of seconds.
    
    GLOBAL PARAMETERS:
    calib_state_val: [int]; The state value from where to extract the appropriate calibration data.
    
    OUTPUT VARIABLES:
    calib_cont_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        signals:     [xarray (channels x time samples (units: s) > floats (units: microvolts)]; Array of only continuous
                     voltage signals from only the relevant time samples for the calibration tasks. Time dimension is in
                     units of seconds.
        states:      [xarray (1 x time samples) > ints (0 or 1)]; Array if states from only the relevant time samples for
                     the calibration tasks. Time dimension is in units of seconds.
    """
    
    # COMPUTATION:
    
    
    # Iterating across all tasks in the calibration data dictionary.
    for this_task in calib_cont_dict.keys():
        
        # Extracting the calibration signals and states from the current task.
        signals = calib_cont_dict[this_task]['signals']
        states  = calib_cont_dict[this_task]['states']
        
        # Find indices where the states array only equals the calibration state value.
        state_val_inds = states == calib_state_val
        
        # Extracting only signals and states of the state value indices.
        calib_cont_dict[this_task]['states']  = states[state_val_inds]
        calib_cont_dict[this_task]['signals'] = signals[:,state_val_inds]
    
    return calib_cont_dict





def car_filter(data_cont_dict, car_tag):   
    """
    DESCRIPTION:
    The signals from the included channels will be extracted and referenced to their common average at each time
    point (CAR filtering: subtracting the mean of all signals from each signal at each time point). The experimenter
    may choose to CAR specific subset of channels or to CAR all the channels together. If the experimenter wishes to
    CAR specific subsets of channels, each subset of channels should be written as a sublist, within a larger nested
    list. For example: [[ch1, ch2, ch3],[ch8, ch10, ch13]].
    
    INPUT VARIABLES:
    car_tag:        [string]; What type of data is being CAR-ed.
    data_cont_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        signals:    [xarray (channels x time samples (units: s) > floats (units: microvolts)]; Array of continuous 
                    voltage signals. Time dimension is in units of seconds.
        states:     [xarray (1 x time samples) > ints (0 or 1)]; Array of states at each time sample. Time dimension
                    is in units of seconds.
        
    GLOBAL PARAMETERS:
    car:        [bool (True/False)] Whether or not CAR filtering will be performed.
    patient_id: [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
        
    OUTPUT VARIABLES:
    data_cont_dict: Same as above, only the signals data may or may not be CAR filtered.
    """
    
    # Printing the type of data may be getting CAR-ed.
    print('\n'+car_tag)
    
    # If CAR filtering will happen.
    if car:
        
        # Iterating across all task blocks.
        for this_task in data_cont_dict.keys():
            
            # Printing for which individual task the CAR filter is applied.
            print(car_tag, ' ', this_task)
                        
            # Extracting the signals from the current task block.
            signals = data_cont_dict[this_task]['signals']

            # Extracting all of the channel and time coordinates from the signals xarray.
            chs_include = list(signals.channel.values) 
            t_seconds   = signals.time
            
            # Extracting the signal values from the signals xarray for speed.
            signals = signals.values

            # Loading the sets of channels over which to independently apply CAR.
            specific_car_chs = params_model_training.car_channels[patient_id]

            # Ensuring that the CAR channels are actually in the included channels list. If a particular channel is not in the 
            # included hannels list, it will be removed from the CAR channel list.
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

            # If the experimenter wishes to CAR only a specific subset of channels.
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

            # If the user wishes to CAR all channels together.
            else:
                # Iterating through all time samples to CAR all channels.
                for n in range(n_samples):
                    signals_car[:, n] = signals[:, n] - np.mean(signals[:, n])

            # Converting the CAR-ed signals back into an xarray.
            signals_car_xr = xr.DataArray(signals_car, 
                                          coords={'channel': chs_include, 'time': t_seconds}, 
                                          dims=["channel", "time"])

            # Updating the data dictionary with the CAR-ed signals.
            data_cont_dict[this_task]['signals'] = signals_car_xr  
            
    # If no CAR filtering of signals was applied. 
    else:
        pass
    
    return data_cont_dict





def channel_selector(eeglabels):
    """
    DESCRIPTION:
    This function extracts all the channels to include in further analysis and excludes the experimenter-determined
    channels for elimination.

    INPUT VARIABLES:
    eeglabels: [array > strings (eeg channel names)]: EEG channels extracted from the .hdf5 or .mat file.
    
    GLOBAL PARAMETERS:
    elim_channels: [dictionary (key: string (patient_id); Value: list > strings (bad channels))]; The list of
                   bad or non-neural channels to be exlucded from further analysis of the neural data.
    patient_id:    [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    
    OUTPUT VARIABLES:
    chs_exclude: [list > strings]; The list of channels to be excluded in further analysis.
    chs_include: [list > strings]; The list of channels to be included in further analysis.
    """
    
    # COMPUTATION:
    
    # Extracting the additional channels for elimination.
    chs_exclude = params_model_training.elim_channels[patient_id]
            
    # Based on the bad channels, computing the included and excluded channels.
    chs_include = [n for n in eeglabels if n not in chs_exclude]
    
    # PRINTING:
    print('\nEXCLUDED CHANNELS:')
    print(chs_exclude)
    print('\nINCLUDED CHANNELS:')
    print(chs_include)
    
    return chs_exclude, chs_include





def computing_channel_importance(aw_shifts, chs_include, grasp_bandpower_dict, saliency_class, saliency_powerband, t_history,\
                                 t_grasp_end_per_trial, t_grasp_start_per_trial):
    """
    DESCRIPTION:
    Computing the channel importance (saliency map) to a specific class (attemped movement) in a specified frequency 
    powerband. Since the saliency map is made from historical time features, for each channel, the L2 norm is taken.
    These channel importances are computed for each time sample corresponding to the specific class and then the 
    average is taken across all samples.
    
    INPUT VARIABLES:
    aw_shifts:                 [dictionary (Key: string (Task ID); Value: dictionary (Key/Value pairs below)];
        N (where N is an int): [int]; AW shift for the Nth trial in units of samples.
    chs_include:               [list > strings]; The list of channels to be included in further analysis.
    grasp_bandpower_dict:      [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_power:             [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each frequency band,
                               the band power is computed for each channel across every time point.
        sxx_power_z:           [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each standardized (to
                               calibration) frequency band, the band power is computed for each channel across every time point.
        sxx_states:            [xarray (1 x time samples) > strings ('state_ON'/'state_OFF'/'neutral')]; Stimulus array downsampled
                               to match time resolution of the signal spectral power. Time dimension is in units of seconds.
        power_trials_z:        [xarray (trials x channels x frequency bins x time samples) > floats (units: V^2/Hz)]; The spectral
                               information for each trial. Time dimension is in units of seconds.
    
    saliency_class:            [string]; The class for whose channel importance scores will be computed.
    saliency_powerband:        [string (powerbandX)]; The powerband for whose channel importances will be computed.
    t_grasp_end_per_trial:     [float (units: s)]; Time of trial end relative to stimulus onset.
    t_grasp_start_per_trial:   [float (units: s)]; Time of trial start relative to stimulus onset.
    t_history:                 [float (unit: ms)]; Amount of feature time history.
    
    NECESSARY FUNCTIONS:
    averaging_channel_importance_scores
    computing_channel_importance_per_sample
    
    OUTPUT VARIABLES:
    ch_importance_scores: [array (1 x channels) > floats]; Mean importance score for each channel, averaged across all
                          time samples from all validation folds.
    """
    # COMPUTATION:
    
    # Computing the importance score of each channel for each sample over multiple folds.
    ch_importance_per_sample = computing_channel_importance_per_sample(aw_shifts, chs_include, grasp_bandpower_dict,\
                                                                       saliency_class, saliency_powerband, t_history,\
                                                                       t_grasp_end_per_trial, t_grasp_start_per_trial)

    # Computing the mean importance for each channel across all samples of all folds.
    ch_importance_scores = averaging_channel_importance_scores(ch_importance_per_sample)
    
    return ch_importance_scores





def computing_channel_importance_per_sample(aw_shifts, chs_include, grasp_bandpower_dict, saliency_class, saliency_powerband,\
                                            t_history, t_tr_end_rel_stim_on, t_tr_start_rel_stim_on):
    """ 
    DESCRIPTION:
    Computing the channel importance (saliency maps) to a specific class (attempted movement) in a specified frequency
    powerband. Since the saliency map is made from historical time features, for each channel, the L2 norm is taken.
    
    INPUT VARIABLES:
    aw_shifts:                 [dictionary (Key: string (Task ID); Value: dictionary (Key/Value pairs below)];
        N (where N is an int): [int]; AW shift for the Nth trial in units of samples.
    chs_include:               [list > strings]; The list of channels to be included in further analysis.
    grasp_bandpower_dict:      [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_power:             [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each frequency band,
                               the band power is computed for each channel across every time point.
        sxx_power_z:           [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each standardized (to
                               calibration) frequency band, the band power is computed for each channel across every time point.
        sxx_states:            [xarray (1 x time samples) > strings ('state_ON'/'state_OFF'/'neutral')]; Stimulus array downsampled
                               to match time resolution of the signal spectral power. Time dimension is in units of seconds.
        power_trials_z:        [xarray (trials x channels x frequency bins x time samples) > floats (units: V^2/Hz)]; The spectral
                               information for each trial. Time dimension is in units of seconds.
    saliency_class:            [string]; Class whose saliency maps will be computed.
    saliency_powerband:        [string ('powerbandX')]; ID of the powerband whose saliency map will be computed.
    t_history:                 [float (unit: ms)]; Amount of feature time history.
    t_grasp_end_per_trial:     [float (units: s)]; Time of trial end relative to stimulus onset.
    t_grasp_start_per_trial:   [float (units: s)]; Time of trial start relative to stimulus onset.
    
    OUTPUT VARIABLES:
    ch_importance_per_sample: [dictionary (key: string (fold); Value: array (time samples x channels))]; For each validation fold,
                              the importance score for each channel is computed for each sample of the experimenter-specified
                              saliency class.
    """
    
    # COMPUTATION:
    
    # Extracting the number of historical time points and features.
    n_history  = int(t_history/sxx_shift)
    n_channels = len(chs_include)

    # Initializing the diciionary of importance scores for each fold.
    ch_importance_per_sample = {}

    # Extracting only the band power information of the powerband used for saliency mapping.
    grasp_bandpower_dict_saliencymap = extracting_saliencymap_powerband(grasp_bandpower_dict, saliency_powerband)

    # Assigning per-sample labels.
    labels_dict = creating_labels(aw_shifts, grasp_bandpower_dict_saliencymap, t_tr_end_rel_stim_on, t_tr_start_rel_stim_on)

    # Creating feature array with concatentated power bands and historical time shifts.
    features_dict = creating_features(grasp_bandpower_dict_saliencymap, t_history)

    # Adjusting the features and labels time dimensions to account for zero-padded features due to the historical features array. 
    features_dict, labels_dict = time_history_sample_adjustment(features_dict, labels_dict, t_history)

    # Downsampling the labels and corresponding features of the overrepresented class.
    features_dict, labels_dict = equalizing_samples_per_class(features_dict, labels_dict)

    # Splitting up the tasks into training and validation tasks per-fold.
    training_folds_tasks, validation_folds_tasks = training_validation_split_tasks(features_dict)

    # Splitting the training and validation data for multiple folds. 
    training_data_folds,\
    training_labels_folds,\
    validation_data_folds,\
    validation_labels_folds = training_validation_split(features_dict, labels_dict, training_folds_tasks, validation_folds_tasks)

    # Computing the means of the training folds.
    training_data_fold_means = mean_compute_all_folds(training_data_folds)

    # Centering the training and validation data folds according to the training data means.
    training_data_folds   = mean_centering_all_folds(training_data_folds, training_data_fold_means)
    validation_data_folds = mean_centering_all_folds(validation_data_folds, training_data_fold_means)

    # Computing the PC features of the training and validation data according to the eigenvectors of the training data.
    training_data_folds, validation_data_folds = pc_transform_all_folds(training_data_folds, validation_data_folds)

    # Rearranging the training and validation data according to the type of model that will be fit.
    training_data_folds   = rearranging_features_all_folds(training_data_folds)
    validation_data_folds = rearranging_features_all_folds(validation_data_folds)

    # Creating a model for each validation fold.
    fold_models = training_fold_models(training_data_folds, training_labels_folds, validation_data_folds, validation_labels_folds)

    # Iterating across all the folds.
    for this_fold in fold_models.keys():

        # Initializing baseline parameters for attribution mask..
        name_baseline_tensors = {
                                 "Baseline Image: Black": tf.zeros(shape=(n_history, n_channels)),
                                 "Baseline Image: Random": tf.random.uniform( shape=(n_history, n_channels), minval=0.0, maxval=1.0),
                                 "Baseline Image: White": tf.ones(shape=(n_history, n_channels)),
                                }

        # Extracting the model from the current fold.
        this_fold_model = fold_models[this_fold]

        # Extracting the validation data and labels corresponding to this fold.
        this_fold_validation_data   = validation_data_folds[this_fold]
        this_fold_validation_labels = validation_labels_folds[this_fold]

        # Extracting the indices of the saliency class.
        this_fold_saliency_class_inds = [i for i, x in enumerate(this_fold_validation_labels) if x == saliency_class]

        # Computing the total number of indices in the current fold of the saliency class.
        n_saliency_class_inds = len(this_fold_saliency_class_inds)

        # Initializing an array of the attribution masks of all the saliency class indices of the current fold.
        this_fold_importance_l2 = np.zeros((n_saliency_class_inds, n_channels))

        # Iterating across all the indices of the saliency class for the current fold.
        for n, ind in enumerate(this_fold_saliency_class_inds):

            print(ind)

            # Extracting the validation sample at the current index.
            this_sample = this_fold_validation_data[ind,:,:]

            # Creating the attribution mask for the current sample.
            this_sample_importance = saliency_mapping_suite.plot_img_attributions(
                                                                                  model=this_fold_model,    # model_final[0] or model_final_real @ at 1:57 AM, just used model_final[0]
                                                                                  img=this_sample.astype('float32'),
                                                                                  baseline=name_baseline_tensors["Baseline Image: Black"],
                                                                                  target_class_idx=0,
                                                                                  m_steps= 500,# 240, 
                                                                                  cmap=plt.cm.inferno,
                                                                                  overlay_alpha=0.4
                                                                                  )

            # Taking the L2 norm of the importance of all channels. 
            this_sample_importance_l2 = np.linalg.norm(this_sample_importance, ord=2, axis=0)

            # Updating the array of importance scores for each sample for the current fold.
            this_fold_importance_l2[n,:] = this_sample_importance_l2

        # Assigning the current fold's importance scores to the dictionary.
        ch_importance_per_sample[this_fold] = this_fold_importance_l2
    
    return ch_importance_per_sample



def computing_eigenvectors(data):
    """
    DESCRIPTION:
    Computing the eigenvectors of the data array. The eigenvectors will be curtailed according to one of two
    experimenter-specified criteria:
        
    1) Only keep the first eiggenvectors that that in total explain less than or equal variance to percent_var_thr
       ... or ... 
    2) Only keep the first n_pc_thr eigenvectors.
        
    INPUT VARIABLES:
    data: [array (features x samples) > floats (units: V^2/Hz)];
    
    GLOBAL PARAMETERS:
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
    valid_data_folds:   [dict (key: string (fold ID); Value: xarray (dimensions vary based on model type) > floats (units: V^2/Hz))];
                        Data across all validation tasks per fold. Equal number of samples per class. PC features. Rearranged 
                        according to the type of model that will be trained.
    valid_labels_folds: [dict (key: string (fold ID); Value: xarray (1 x time samples) > strings ('grasp'/'rest'))]; Labels across
                        all validation tasks per fold. Equal number of labels per class.
                    
    GLOBAL PARAMETERS:
    model_classes:  [list > strings]; List of all the classes to be used in the classifier.
    
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





def concatenating_historical_features(sxx_power_z_all_bands, t_history):
    """
    DESCRIPTION:
    Based on the experimenter-specified time history (t_history) and the time resolution (global variable sxx_shift),
    the number of historical time points are calculated (t_history/sxx_shift). An xarray with dimensions (history, features, time)
    is created, where each coordinate in the history dimension represents how much the features were shifted in time. For 
    example, consider one coordinate in the feature array, and suppose a time length of 10 samples and a total time history 
    of 3 samples. For this feature, the resulting xarray would look like:

    historical time shifts
         n=2 shifts      [[0.000, 0.000, 0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101],
         n=1 shift        [0.000, 0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521],
         n=0 shifts       [0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521, 0.882]]
                            t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7    t=8    t=9     (time samples)   

    and the resulting dimensions of this array are (history=3, features=1, time=10).
    
    INPUT VARIABLES:
    sxx_power_z_all_bands: [dictionary (Key: string (task ID); Value: xarray (features (chs x bands) x time samples) > floats (units: V^2/Hz))]; 
                           Concatenated band power features across all power bands. 
    t_history:             [float (unit: s)]; Amount of time history used as features.
                           
    GLOBAL PARAMETERS: 
    sxx_shift: [int (units: ms)]; Length of time by which sliding window (sxx_window) shifts along the time domain.
    
    OUTPUT VARIABLES:
    sxx_power_z_all_history: [dictionary (Key: string (task ID); Value: xarray (time history, features, time) > floats (units: V^2/Hz))]; 
                             Array of historical time features.
    """
    
    # COMPUTATION:
    
    # Initializing a dictionary of historical time features.
    sxx_power_z_all_history = {}
    
    # Computing the total number of historical time features.
    n_history = int(t_history/sxx_shift)
    
    # Extracting the number of features over all powerbands. Computing from the first task, as the number of features should be the 
    # same across all tasks.
    n_features = sxx_power_z_all_bands['task0'].feature.shape[0]

    # Iterating across all tasks.
    for this_task in sxx_power_z_all_bands.keys():
        
        # Extracting the data from the current task.
        this_task_sxx_power_z = sxx_power_z_all_bands[this_task]

        # Extracting the time array and corresponding number of time samples in the task.
        t_seconds = this_task_sxx_power_z.time
        n_samples = t_seconds.shape[0]
        
        # Initializing a feature array which will contain all historical time features.
        this_task_power_all_history = np.zeros((n_history, n_features, n_samples))
                
        # Iterating across all historical time shifts. The index, n, is the number of samples back in time that will be shifted.
        for n in range(n_history):
            
            # If currently extracting historical time features (time shift > 0)
            if n >= 1:
                
                # Extracting the historical time features for the current time shift.
                these_features_history = this_task_sxx_power_z[:,:-n]
                
                # Creating a zero-padded array to make up for the time curtailed from the beginning of the features array.
                zero_padding = np.zeros((n_features, n))
                
                # Concatenating the features at the current historical time point with a zero-padded array.
                these_features = np.concatenate((zero_padding, these_features_history), axis=1)
                
            # If extracting the first time set of features (time shift = 0). 
            else:
                these_features = this_task_sxx_power_z
                        
            # Assigning the current historical time features to the xarray with all historical time features.
            this_task_power_all_history[n,:,:] = these_features
        
        # Converting the historical power features for this task to xarray.
        this_task_power_all_history = xr.DataArray(this_task_power_all_history, 
                                                   coords={'history': np.arange(n_history), 'feature': np.arange(n_features), 'time': t_seconds}, 
                                                   dims=["history", "feature", "time"])
        
        # Adding the historical time features for the current task to the dictionary.
        sxx_power_z_all_history[this_task] = this_task_power_all_history
    
    return sxx_power_z_all_history





def concatenating_power_bands(grasp_bandpower_dict):
    """
    DESCRIPTION:
    For each task, features across all powerbands are concatenated into one feature dimension. For example, if a feature array
    from one task has dimensions (chs: 128, powerbands: 3, time samples: 4500), the powerband concatenation will result in an
    array of size: (chs x pwrbands: 384, time samples: 4500).
    
    INPUT VARIABLES:
    grasp_bandpower_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_power:        [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each frequency band,
                          the band power is computed for each channel across every time point.
        sxx_power_z:      [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each standardized (to
                          calibration) frequency band, the band power is computed for each channel across every time point.
        sxx_states:       [xarray (1 x time samples) > strings ('state_ON'/'state_OFF'/'neutral')]; Stimulus array downsampled
                          to match time resolution of the signal spectral power. Time dimension is in units of seconds.
        power_trials_z:   [xarray (trials x channels x frequency bins x time samples) > floats (units: V^2/Hz)]; The spectral
                          information for each trial. Time dimension is in units of seconds.
                        
    OUTPUT VARIABLES:
    features_dict: [dictionary (Key: string (task ID); Value: xarray (features (chs x bands), time samples) > floats (units: V^2/Hz)]; 
                   Concatenated band power features over all power bands. 
    """
    
    # COMPUTATION:
    
    # Initializing a dictionary of concatenated powerbands for all tasks.
    features_dict = {}
    
    # Iterating across each task.
    for this_task in grasp_bandpower_dict.keys():
        
        # Extracting the standardized power over all trials for the current task.
        this_task_power_all_trials = grasp_bandpower_dict[this_task]['sxx_power_z']
        
        # Extracting the dimension sizes of the current features array.
        n_channels = this_task_power_all_trials.shape[0]
        n_bands    = this_task_power_all_trials.shape[1]
        n_samples  = this_task_power_all_trials.shape[2]
        
        # Concatenating all powerbands and flipping the dimensions
        this_task_power_all_bands = np.asarray(this_task_power_all_trials).reshape(n_channels*n_bands, n_samples)
        
        # Converting the concatenated band features back into an xarray.
        this_task_power_all_bands = xr.DataArray(this_task_power_all_bands, 
                                                 coords={'feature': np.arange(n_channels*n_bands), 'time': np.arange(n_samples)}, 
                                                 dims=["feature", "time"])
        
        # Adding the array with concatenated power bands to the dictionary.
        features_dict[this_task] = this_task_power_all_bands
    
    return features_dict





def confusion_matrix_display(cm, suppress_figs='No'):
    """
    DESCRIPTION:
    Given the experimenter-input confusion matrix array and corresponding labels, the confusion matrix is displayed.

    INPUT VARIABLES:
    cm: [array > float]; Confusion matrix which contains the accuracy of the true (vertical axis) and predicted
        (horizontal axis) labels.
    
    GLOBAL PARAMETERS
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
        
        fig, ax = plt.subplots(figsize = (18, 12));
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
        
        
        # fig.savefig('Confusion_Matrix', bbox_inches='tight')
        # fig.savefig('Confusion_Matrix.svg', format = 'svg', bbox_inches='tight', dpi = 2000)
        
    if suppress_figs == 'Yes':
        fig = []
        
    return fig, acc





# def creating_labels(aw_shifts, grasp_bandpower_dict, t_tr_end_rel_stim_on, t_tr_start_rel_stim_on):
#     """
#     DESCRIPTION:
#     For each task, each time sample is labeled according to the experimenter-determined trial onset and offset and the per-trial shift.
    
#     INPUT VARIABLES:
#     aw_shifts: [dictionary (Key: string (Task ID); Value: dictionary (Key/Value pairs below)];
#         trial number (int): shift AW shift for the current trial in units of samples (int)
#     grasp_bandpower_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
#         sxx_power:      [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each frequency band, the band power is
#                         computed for each channel across every time point.
#         sxx_power_z:    [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each standardized (to calibration) 
#                         frequency band, the band power is computed for each channel across every time point.
#         sxx_stimuli:    [xarray (1 x time samples) > strings ('stim'/'no_stim')]; Stimulus array downsampled to match time resolution
#                         of the signal spectral power. Time dimension is in units of seconds.  
#         power_trials_z: [xarray (trials x channels x powerbands x time samples) > floats (units: V^2/Hz)]; The spectral
#                         information for each trial. Time dimension is in units of seconds.
#     t_tr_end_rel_stim_on:   [float (units: s)]; Time of trial end relative to stimulus onset.
#     t_tr_start_rel_stim_on: [float (units: s)]; Time of trial start relative to stimulus onset.
    
#     GLOBAL PARAMETERS:
#     model_classes:  [list > strings]; List of all the classes to be used in the classifier.
    
#     OUTPUT VARIABLES:
#     labels_dict: [dictionary (Key: string (task ID); Value: xarray > strings ('grasp'/'rest'))]; For each time sample in a task, there 
#                  exists a rest or grasp label depending on the experimenter-determined onset and offset of modulation as well as the 
#                  per-trial shift from the AW model.
#     """
    
#     # COMPUTATION:
    
#     # Initializing a dictionary of labels.
#     labels_dict = {}
    
#     # Extracting the default and active classes.
#     default_class = model_classes[0]
#     active_class  = model_classes[1]
    
#     # Iterating across all tasks. 
#     for this_task in grasp_bandpower_dict.keys():
        
#         print('LABELLING SAMPLES FOR: ',this_task )
        
#         # Extracting the array of stimulus labels.
#         this_task_stimuli = grasp_bandpower_dict[this_task]['sxx_stimuli']
        
#         # Extracting the time samples from the array of stimuli.
#         t_seconds = this_task_stimuli.time.values
        
#         # Computing the total number of samples in this task.
#         n_samples = t_seconds.shape[0]
                
#         # Initializing a list of labels.
#         this_task_labels = [default_class] * n_samples
        
#         # print(len(this_task_labels))
        
#         # Extracting the affine warp shifts for the current task.
#         this_task_aw_shifts = aw_shifts[this_task]
                
#         # Extracting the start and end indices of the stimulus onset and offset.        
#         _, _, _, this_task_start_end_inds = unique_value_index_finder(grasp_bandpower_dict[this_task]['sxx_stimuli'])
        
#         # Iterating across each stim onset/offset index pair of the current task.
#         for tr, (stim_onset_idx, _) in enumerate(this_task_start_end_inds['stim_ON']):
            
#             # Computing the trial onset and offset times by adding the experimenter-determined per-trial onset time (from average power-trial raster traces)
#             # to the stimulus onset time.
#             t_tr_onset  = t_seconds[stim_onset_idx] + t_tr_start_rel_stim_on
#             t_tr_offset = t_tr_onset + (t_tr_end_rel_stim_on - t_tr_start_rel_stim_on)
            
#             # Computing the trial onset and offset indices according to the closest index in the time array.
#             tr_onset_ind  = np.abs(t_seconds - t_tr_onset).argmin()
#             tr_offset_ind = np.abs(t_seconds - t_tr_offset).argmin()
            
#             # Computing the number of samples between the onset and offset inds.
#             n_diff_onset_offset = tr_offset_ind - tr_onset_ind
            
#             # If affine warp shifts exist, then shift each trial onset and offset by the trial shift.
#             if aw_shifts:
#                 tr_shift      = this_task_aw_shifts[tr]
#                 tr_onset_ind  += tr_shift
#                 tr_offset_ind += tr_shift
            
#             # Labeling the time between onset and offset with grasp.
#             this_task_labels[tr_onset_ind:tr_offset_ind] = [active_class]*n_diff_onset_offset
            
#         # In case the labels for the current class end up surpassing the total number of time samples due to the affine warp shifting, 
#         # curtail the task labels to this sample size.
#         this_task_labels = this_task_labels[:n_samples]
            
#         # Converting the list of task labels to an xarray.
#         this_task_labels = xr.DataArray(this_task_labels, 
#                                         coords={'time': t_seconds}, 
#                                         dims=["time"])
        
#         # Updating the labels dictionary with the current task's per-sample label.
#         labels_dict[this_task] = this_task_labels
        
#     return labels_dict





def creating_labels(aw_shifts, grasp_bandpower_dict, t_grasp_end_per_trial, t_grasp_start_per_trial):
    """
    DESCRIPTION:
    For each task, each time sample is labeled as grasp or rest according to the experimenter-specified trial onset and offset 
    and the per-trial shift.
    
    INPUT VARIABLES:
    aw_shifts:               [dictionary (Key: string (Task ID); Value: dictionary (Key/Value pairs below)];
        N (where N is an int): [int]; AW shift for the Nth trial in units of samples.
    grasp_bandpower_dict:    [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_power:           [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each frequency band,
                             the band power is computed for each channel across every time point.
        sxx_power_z:         [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each standardized (to
                             calibration) frequency band, the band power is computed for each channel across every time point.
        sxx_states:          [xarray (1 x time samples) > strings ('state_ON'/'state_OFF'/'neutral')]; Stimulus array downsampled
                             to match time resolution of the signal spectral power. Time dimension is in units of seconds.
        power_trials_z:      [xarray (trials x channels x frequency bins x time samples) > floats (units: V^2/Hz)]; The spectral
                             information for each trial. Time dimension is in units of seconds.
    t_grasp_end_per_trial:   [float (units: s)]; Time of trial end relative to stimulus onset.
    t_grasp_start_per_trial: [float (units: s)]; Time of trial start relative to stimulus onset.
    
    GLOBAL PARAMETERS:
    model_classes:  [list > strings]; List of all the classes to be used in the classifier.
    
    OUTPUT VARIABLES:
    labels_dict: [dictionary (Key: string (task ID); Value: xarray > strings ('grasp'/'rest'))]; For each time sample in each
                 task, there exists a rest or grasp label depending on the experimenter-specified onset and offset of
                 modulation as well as the per-trial shift from the AW model.
    """
    
    # COMPUTATION:
    
    # Initializing a dictionary of labels.
    labels_dict = {}
    
    # Extracting the default and active classes.
    rest_class  = model_classes[0]
    grasp_class = model_classes[1]
    
    # Iterating across all tasks. 
    for this_task in grasp_bandpower_dict.keys():
        
        print('LABELING SAMPLES FOR: ',this_task )
        
        # Initializing a list of labels as the state array.
        this_task_labels = copy.deepcopy(grasp_bandpower_dict[this_task]['sxx_states'])
        
        # Extracting the time samples from the states array.
        t_seconds = this_task_labels.time.values
        
        # Computing the total number of samples in this task.
        n_samples = t_seconds.shape[0]
                
        # Extracting the affine warp shifts for the current task.
        this_task_aw_shifts = aw_shifts[this_task]
                
        # Extracting the start and end indices of the states onset and offset.        
        _, _, _, this_task_start_end_inds = unique_value_index_finder(this_task_labels)
        
        # Iterating across each state onset/offset index pair of the current task.
        for tr, (state_onset_idx, _) in enumerate(this_task_start_end_inds['state_ON']):
            
            # Computing the onset and offset times of the grasp labels for the current trial. This si done by adding the 
            # experimenter-specified per-trial onset time (from average power-trial raster traces) to the onset time of
            # the ON state.
            t_grasp_label_onset  = t_seconds[state_onset_idx] + t_grasp_start_per_trial
            t_grasp_label_offset = t_grasp_label_onset + (t_grasp_end_per_trial - t_grasp_start_per_trial)
            
            # Computing the grasp onset and offset indices according to the closest index in the time array.
            grasp_onset_ind  = np.abs(t_seconds - t_grasp_label_onset).argmin()
            grasp_offset_ind = np.abs(t_seconds - t_grasp_label_offset).argmin()
            
            
            
            # If affine warp shifts exist, then shift each trial onset and offset by the respective trial shift.
            if aw_shifts:
                tr_shift         = this_task_aw_shifts[tr]
                grasp_onset_ind  += tr_shift
                grasp_offset_ind += tr_shift
                
            # If the shifted grasp onset or offset indices are less than 0 or go beyond the total number of samples in 
            # the current task.
            if grasp_onset_ind < 0:
                grasp_onset_ind = 0
            if grasp_offset_ind > n_samples:
                grasp_offset_ind = n_samples
                
            # Computing the number of samples between the grasp onset and offset inds.
            n_diff_onset_offset = grasp_offset_ind - grasp_onset_ind            
            
            # Labeling the time between grasp onset and offset with grasp.
            this_task_labels[grasp_onset_ind:grasp_offset_ind] = [grasp_class]*n_diff_onset_offset
            
        # In case the labels for the current task end up surpassing the total number of time samples due to the affine
        # warp shifting, curtail the task labels to this sample size.
        # this_task_labels = this_task_labels[:n_samples] # This should never happen due to the two previous IF statements.
        
        
        # Where the state_ON and state_OFF labels remain, replace with the default class, rest. We are ignoring any possible
        # neutral labels as those won't be used for training.
        remaining_state_on_inds  = np.squeeze(np.argwhere(this_task_labels.values == 'state_ON'))
        remaining_state_off_inds = np.squeeze(np.argwhere(this_task_labels.values == 'state_OFF'))        
        this_task_labels[remaining_state_on_inds]  = rest_class
        this_task_labels[remaining_state_off_inds] = rest_class
            
        # Converting the list of labels to an xarray for the current class.
        this_task_labels = xr.DataArray(this_task_labels, 
                                        coords={'time': t_seconds}, 
                                        dims=["time"])
        
        # Updating the labels dictionary with the current task's per-sample label.
        labels_dict[this_task] = this_task_labels
        
    return labels_dict





def creating_power_trial_rasters(grasp_bandpower_dict):
    """
    DESCRIPTION:
    Concatenating power trial rasters across all tasks.
    
    INPUT VARIABLES:
    grasp_bandpower_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_power:        [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each frequency band,
                          the band power is computed for each channel across every time point.
        sxx_power_z:      [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each standardized (to
                          calibration) frequency band, the band power is computed for each channel across every time point.
        sxx_states:       [xarray (1 x time samples) > strings ('state_ON'/'state_OFF'/'neutral')]; Stimulus array downsampled
                          to match time resolution of the signal spectral power. Time dimension is in units of seconds.
        power_trials_z:   [xarray (trials x channels x frequency bins x time samples) > floats (units: V^2/Hz)]; The spectral
                          information for each trial. Time dimension is in units of seconds.
    
    NECESSARY FUNCTIONS:
    index_advancer
    
    OUTPUT VARIABLES:
    ptr_all_trials: [xarray (trials x channels x powerbands x time samples) > floats (units: V^2/Hz)]; The spectral information
                    for each trial (across all tasks). Time dimension is in units of seconds.
    """

    # COMPUTATION:
    
    # Extracting the channels and frequency bins from task0. These are the same across all tasks.
    channels        = grasp_bandpower_dict['task0']['power_trials_z'].channel
    powerbands      = grasp_bandpower_dict['task0']['power_trials_z'].powerband
    t_sxx_per_trial = grasp_bandpower_dict['task0']['power_trials_z'].time
    
    # Computing the number of channels and frequency bins.
    n_chs               = channels.shape[0]
    n_bands             = powerbands.shape[0]
    n_samples_per_trial = t_sxx_per_trial.shape[0]
    
    # Initializing the total number of trials across all tasks.
    n_trials_all = 0
    
    # Computing the total number of trials.
    for this_task in grasp_bandpower_dict.keys():
        
        # Extracting the number of trials during the current task.
        n_trials_this_task = grasp_bandpower_dict[this_task]['power_trials_z'].trial.shape[0]
        
        # Updating the number of total trials.
        n_trials_all += n_trials_this_task
        
    # Initializing the xarray of spectrograms per-trial.
    ptr_all_trials = xr.DataArray(np.zeros((n_trials_all, n_chs, n_bands, n_samples_per_trial)), 
                                  coords={'trial': np.arange(n_trials_all),'channel': channels, 'powerband': powerbands, 'time': t_sxx_per_trial}, 
                                  dims=["trial", "channel", "powerband", "time"])
    
    # Initializing the trial indices.    
    tr_inds = np.zeros((2,))
    
    # Iterating across all tasks. 
    for this_task in grasp_bandpower_dict.keys():
        
        # Extracting the power trial rasters of all trials for the current task.
        this_task_powerband_trials = grasp_bandpower_dict[this_task]['power_trials_z']
        
        # Extracting the number of trials in this task.
        n_trials_this_task = this_task_powerband_trials.trial.shape[0]
        
        # Updating the trial indices.
        tr_inds = index_advancer(tr_inds, n_trials_this_task)

        # Updating the array of spectrogram information for all trials.
        ptr_all_trials.loc[tr_inds[0]:tr_inds[1]-1,:,:,:] = this_task_powerband_trials.values
    
    return ptr_all_trials





def creating_features(grasp_bandpower_dict, t_history):
    """
    DESCRIPTION:
    Power features across all powerbands are concatenated in one feature dimension. For example, if a feature array has
    dimensions (chs: 128, powerbands: 3, time samples: 4500), the powerband concatenation will result in an array of size:
    (chs x pwrbands: 384, time samples: 4500).

    Then, based on the experimenter-specified time history (t_history) and the time resolution (global variable sxx_shift),
    the number of historical time points are calculated (t_history/sxx_shift). An xarray with dimensions (history, features, time)
    is created, where each coordinate in the history dimension represents how much the features were shifted in time. For 
    example, consider one coordinate in the feature array, and suppose a time length of 10 samples and a total time history 
    of 3 samples. For this feature, the resulting xarray would look like:

    historical time shifts
         n=2 shifts      [[0.000, 0.000, 0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101],
         n=1 shift        [0.000, 0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521],
         n=0 shifts       [0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521, 0.882]]
                            t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7    t=8    t=9     (time samples)   

    and the resulting dimensions of this array are (history=3, features=1, time=10).
    
    INPUT VARIABLES:
    grasp_bandpower_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_power:        [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each frequency band,
                          the band power is computed for each channel across every time point.
        sxx_power_z:      [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each standardized (to
                          calibration) frequency band, the band power is computed for each channel across every time point.
        sxx_states:       [xarray (1 x time samples) > strings ('state_ON'/'state_OFF'/'neutral')]; Stimulus array downsampled
                          to match time resolution of the signal spectral power. Time dimension is in units of seconds.
        power_trials_z:   [xarray (trials x channels x frequency bins x time samples) > floats (units: V^2/Hz)]; The spectral
                          information for each trial. Time dimension is in units of seconds.
    t_history:            [float (unit: ms)]; Amount of feature time history.
    
    NECESSARY FUNCTIONS:
    concatenating_historical_features
    concatenating_power_bands
    
    OUTPUT VARIABLES:
    features_dict: [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats (units: V^2/Hz))]
                   Array of historical time features.
    """
    # COMPUTATION:
    
    # Concatenating all powerbands into one dimension.
    features_dict = concatenating_power_bands(grasp_bandpower_dict)

    # Creating the historical time features.
    features_dict = concatenating_historical_features(features_dict, t_history)
    
    return features_dict





def data_upload(chs_include, data_info_dict, eeglabels, state_info_dict):
    """
    DESCRIPTION:
    For each grasp-based or calibration block, the signals and states at the resolution of the continuous
    sampling rate are uploaded into data dictionaries.
    
    INPUT VARIABLES:
    chs_include:     [list > strings]; The list of channels to be included in further analysis.
    data_info_dict:  [dictionary (Key: string (date in YYYY_MM_DD format); Values: list > string (task names);
                     Values and keys correspond to grasp/calibration tasks and dates on which those tasks were run.
    eeglabels:       [list > strings (eeg channel names)]: EEG channels extracted from the .hdf5 or .mat file.
    state_info_dict: [dictionary (Key: string (date in YYYY_MM_DD format); Values: list > string (state names))];
                     Values and keys correspond to the relevant states for each grasp/calibraion task.
    
    GLOBAL PARAMETERS:
    file_extension: [string (hdf5/mat)]; The data file extension of the data.
    patient_id:     [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    sampling_rate:  [int (samples/s)]; Sampling rate at which the data was recorded.
    
    OUTPUT VARIABLES:    
    data_cont_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        signals:    [xarray (channels x time samples (units: s) > floats (units: microvolts)]; Array of continuous 
                    voltage signals. Time dimension is in units of seconds.
        states:     [xarray (1 x time samples) > ints (0 or 1)]; Array of states at each time sample. Time dimension
                    is in units of seconds.
    """
    
    # COMPUTATION:
    
    # Initializing a dictionary with the signals and state arrays recorded at the continuous sampling rate.
    data_cont_dict = collections.defaultdict(dict)

    # Initializing the task counter.
    n = 0

    # Iterate across all dates.
    for this_date in data_info_dict.keys():

        # Extracting the list of data blocks and corresponding state names for the current date.
        this_date_data_tasks  = data_info_dict[this_date]
        this_date_state_names = state_info_dict[this_date]

        # Iterating simultaneously across the data blocks and state names.
        for this_state, this_task in zip(this_date_state_names, this_date_data_tasks):

            # Creating the file pathway from where to upload the data.
            this_path = '/mnt/shared/ecog/' + patient_id + '/' + file_extension + '/' + this_date + '/' + this_task + '.' + file_extension

            # Uploading the eeg signals and states, depending on whether they come from a .hdf5 or .mat file.
            if file_extension == 'hdf5':

                # Uploading the .hdf5 data.
                h5file      = H5EEGFile(this_path)
                eeg         = h5file.group.eeg(); 
                eeg_signals = eeg.dataset[:]

                # Uploading the .hdf5 states.
                aux    = h5file.group.aux();
                states = aux[:, this_state]

            if file_extension == 'mat':

                # Uploading the .mat data.
                matlab_data = loadmat(this_path, simplify_cells=True)
                eeg_signals = matlab_data['signal']

                # Uploading the .mat states.
                states = matlab_data['states'][this_state]

            # Multiplying the EEG signals by a gain that's done downstream of the .dat collection in BCI2000.
            eeg_signals = eeg_signals * 0.25

            # Computing the total number of channels and time samples.
            n_samples  = eeg_signals.shape[0]
            n_channels = len(chs_include)

            # Creating the time array. Note that the first time sample is not 0, as the first recorded signal sample
            # does not correspond to a 0th time.
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


            # Creating the task ID for the current dataset.
            task_id = 'task' + str(n)

            # Updatign the task counter.
            n += 1

            # Populating the data dictionary with the signals and stimuli arrays.
            data_cont_dict[task_id]['signals'] = signals
            data_cont_dict[task_id]['states']  = states
    
    return data_cont_dict





# def data_upload(chs_include, eeglabels, path_list, stimulus_marker):
#     """
#     DESCRIPTION:
#     For each block of data, the signal and stimuli arrays are loaded into data dictionaries.
    
#     INPUT VARIABLES:
#     chs_include:     [list > strings]; The list of channels to be included in further analysis.
#     eeglabels:       [list > strings (eeg channel names)]: EEG channels extracted from the .hdf5 or .mat file.
#     path_list:       [list > strings (pathways)]; The list of file pathways from which the dataset(s) will be extracted.
#     stimulus_marker: [string]; Name of the stimulus marker for extracting GO and calibration task stimulus values.
    
#     GLOBAL PARAMETERS:
#     file_extension:  [string (hdf5/mat)]; The data file extension of the data.
#     sampling_rate:   [int (samples/s)]; Sampling rate at which the data was recorded.
    
#     OUTPUT VARIABLES:
#     raw_data_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
#         signals: [xarray (channels x time samples (units: s) > floats (units: microvolts)]; Array of raw voltage signals.
#                  Time dimension is in units of seconds.
#         stimuli: [xarray (1 x time samples) > ints (0 or 1)]; Array of time samples corresponding to the signals array. Time 
#                  dimension is in units of seconds.
#     """
    
#     # COMPUTATION:
    
#     # Initializing the dictionary with all the signals.
#     raw_data_dict = collections.defaultdict(dict)
    
#     # Iterating across all data paths.
#     for n, path in enumerate(path_list):
        
        
        
        
#         # NEED A WAY CLEANER WAY OF INCLUDING THIS FOR TASKS WITH DIFFERENT TYPES OF STIMULUS MARKERS.
#         if 'Speller' in path:
#             stimulus_marker = 'ControlClick'
            
            
            
            
        
#         # Extracting the eeg signals, depending on whether they come from an .hdf5 or .mat file.
#         if file_extension == 'hdf5':

#             # Extracting the .hdf5 data.
#             h5file      = H5EEGFile(path)
#             eeg         = h5file.group.eeg(); 
#             eeg_signals = eeg.dataset[:]

#             # Extracting the .hdf5 stimuli.
#             aux     = h5file.group.aux();
#             stimuli = aux[:, stimulus_marker]

#         if file_extension == 'mat':

#             # Extracting the .mat data.
#             matlab_data = loadmat(path, simplify_cells=True)
#             eeg_signals = matlab_data['signal']

#             # Extracting the .mat stimuli.
#             stimuli = matlab_data['states'][stimulus_marker]

#         # Multiplying our eeg signals by a gain that's done downstream of the .dat collection in BCI2000.
#         eeg_signals = eeg_signals * 0.25
        
#         # Computing the total number of channels and time samples.
#         n_samples  = eeg_signals.shape[0]
#         n_channels = len(chs_include)
        
#         # Creating the time array. Note that the first time sample is not 0, as the first recorded signal sample does not correspond to
#         # a 0th time.
#         t_seconds = (np.arange(n_samples) + 1)/sampling_rate    
        
        
#         if 'Speller' in path:
            
#             ################## LOADING TIME LAG ##############
#             # Extracitng the date and block ID from the path.
#             block_id = path[46:52]
#             date     = path[27:37]
            
#             # Creating the base path for the time lags.
#             dir_lag = '/mnt/shared/danprocessing/Projects/PseudoOnlineTests_for_RTCoG/Intermediates/' + patient_id +\
#                       '/Speller/LagsBetweenVideoAndBCI2000/' + date + '/' + block_id + '/'

#             # Creating the pathway for the lag for the current date+block pair.
#             path_lag = dir_lag + date + '_' + block_id + '.txt'

#             # Opening up the text file with the time lag from the pathway.
#             txt_file_t_lag = open(path_lag)

#             # Reading the content of the text file with the time lag.
#             text_file_lines = txt_file_t_lag.readlines()

#             # Extracting the time lag from the text file line.
#             t_lag = int(text_file_lines[0])
            
#             # Converting the lag into units of seconds.
#             t_lag_seconds = t_lag/1000 # ms * s/ms

#             # Extracting only audio time after lag.
#             bci2k_lag        = t_seconds - t_lag_seconds # units of seconds. 
#             bci2k_bool       = bci2k_lag > 0
#             bci2k_times_bool = bci2k_lag[bci2k_bool]
    
#             # Zero-ing the time from BCI2000. 
#             t_seconds = bci2k_times_bool - bci2k_times_bool[0] + t_seconds[0]
        
#             # Extracting the aligned signals and stimuli only.
#             stimuli     = stimuli[bci2k_bool]
#             eeg_signals = eeg_signals[bci2k_bool,:]
            
            
            
            
            
#             ################## LOADING START/STOP TIMES ##############
#             # Creating the path for the start and stop times.
#             dir_start_stop = '/mnt/shared/danprocessing/Projects/PseudoOnlineTests_for_RTCoG/Intermediates/' + patient_id +\
#                              '/Speller/BlocksStartAndStops/' + date + '/'
                
#             path_startstop = dir_start_stop + date + '_' + block_id +'_StartStop.txt'
            
#             # Opening up the block start-stop text file from the pathway.
#             txt_file_start_stop = open(path_startstop)

#             # Reading the content of the text file with the start and stop times.
#             text_file_lines = txt_file_start_stop.readlines()

#             # Reading in the strings with the starting and stopping times.
#             line_time_start = text_file_lines[5]
#             line_time_stop  = text_file_lines[6]

#             # Extracting the start and stop strings from the corresponding lines.
#             time_start = float(line_time_start[7:20])
#             time_end   = float(line_time_stop[7:20])
            
#             curt_bool = np.logical_and(t_seconds > time_start, t_seconds < time_end)
            
            
#             # Curtailing the stimuli and signals according to the block start and end times.
#             t_seconds   = t_seconds[curt_bool]
#             stimuli     = stimuli[curt_bool]
#             eeg_signals = eeg_signals[curt_bool,:]
    
#             # Redefining how many samples there are.
#             n_samples = eeg_signals.shape[0]
            
#             print('N SAMPLES: ', n_samples)
            
#             # Making all the stimuli zeros.
#             stimuli = np.zeros((n_samples,))
        
#         # Converting the stimuli array into an xarray.    
#         stimuli = xr.DataArray(stimuli,
#                                coords={'time': t_seconds},
#                                dims=["time"])
    
    
#         # Initializing an xarray for the signals.
#         signals = xr.DataArray(np.zeros((n_channels, n_samples)), 
#                                coords={'channel': chs_include, 'time': t_seconds}, 
#                                dims=["channel", "time"])
        
#         # Populating the signals xarray with the eeg signals.
#         for ch_name in chs_include:
            
#             # Extracting the appropriate channel index from the original eeglabels list and populating the
#             # signals xarray with the channel's activity.
#             eeg_ind              = eeglabels.index(ch_name)            
#             signals.loc[ch_name] = eeg_signals[:,eeg_ind]
        
        
#         # Creating the task ID for the current dataset.
#         task_id = 'task' + str(n)

#         # Populating the data dictionary with the signals and stimuli arrays.
#         raw_data_dict[task_id]['signals'] = signals
#         raw_data_dict[task_id]['stimuli'] = stimuli
        
#     return raw_data_dict





def draw_transparent_ellipse(img, xy, **kwargs):
    """ 
    DESCRIPTION:
    Draws an ellipse inside the given bounding box onto given image. Supports transparent colors
    """
    
    # COMPUTATION:
    
    transp = Image.new('RGBA', img.size, (0,0,0,0))  # Temp drawing image.
    draw = ImageDraw.Draw(transp, "RGBA")
    draw.ellipse(xy, **kwargs)
    
    # Alpha composite two images together and replace first with result.
    img.paste(Image.alpha_composite(img, transp))

    return None
    
    
    
    
    
# def equalizing_samples_per_class(features_dict, labels_dict):
#     """
#     DESCRIPTION:
#     According to the label array, there are likely not an equal number of samples for each class. For whichever class there are more
#     samples, this the indices of this class will be extracted to create a smaller subset of indices in number equal to that of the 
#     underrepresented class. For example, consider the following labels array:
    
#     labels:  ['rest', 'rest', 'rest', 'rest', 'rest, 'grasp', 'grasp', 'grasp', 'rest', 'rest', 'rest', 'rest']
#     indices:    0       1       2       3       4       5        6        7       8       9       10      11
    
#     There are 3 samples with grasp labels, while there are 9 samples with rest labels. The indices corresponding to the rest labels are:
#     [0, 1, 2, 3, 4, 8, 9, 10, 11]. These indices will be randomly subsampled such that they are equal in number to the grasp class. For
#     example: [0, 1, 2] or [1, 4, 10] or [3, 9, 11]. As such the downsampled labels (and correspondig features) will use the indices:
    
#     labels downsampled:  ['rest', 'grasp', 'grasp', 'grasp', 'rest', 'rest']
#     indices downsampled:    3        5        6        7       9       11
    
#     INPUT VARIABLES:
#     features_dict: [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats (units: V^2/Hz))]
#                    Array of historical time features.
#     labels_dict:   [dictionary (Key: string (task ID); Value: xarray > strings ('grasp'/'rest'))]; For each time sample in a task, there 
#                    exists a rest or grasp label depending on the experimenter-determined onset and offset of modulation as well as the 
#                    per-trial shift from the AW model.
                   
#     OUTPUT VARIABLES:
#     features_dict: [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats (units: V^2/Hz))]
#                    Array of historical time features. Time samples reduced such that there are an equal number of features per class.
#     labels_dict:   [dictionary (Key: string (task ID); Value: xarray > strings ('grasp'/'rest'))]; For each time sample in a task, there 
#                    exists a rest or grasp label depending on the experimenter-determined onset and offset of modulation as well as the 
#                    per-trial shift from the AW model. Time samples reduced such that there are an equal number of features per class.
#     """
    
#     # COMPUTATION
    
#     # Extracting the unique labels from task0. These unique labels should be the same across all tasks.
#     unique_labels = np.unique(labels_dict['task0'])
    
#     # Iterating across each task.
#     for this_task in labels_dict.keys():
        
#         # Extracting the features and labels for the current task.
#         this_task_features = np.asarray(features_dict[this_task])
#         this_task_labels   = np.asarray(labels_dict[this_task])
        
#         # Initializing an dictionary describing the number of samples per label.
#         samples_per_label = {}
        
#         # Iterating across all unique labels to determine the number of samples per label.
#         for this_label in unique_labels:
            
#             # Counting the number of samples for the current label and appropriately updating the dictionary.
#             samples_per_label[this_label] = sum(this_task_labels == this_label)
                
#         # Determining which class has the higher and lower number of samples.
#         class_low_samples  = min(samples_per_label, key=samples_per_label.get)
#         class_high_samples = max(samples_per_label, key=samples_per_label.get)
        
#         # Extracting the number of samples of the underrespresented class. This will be used to determine how many samples 
#         # of the overrepresented class to randomly extract.
#         n_samples_class_under = samples_per_label[class_low_samples] 
        
#         # Extracting the sample indices of the under and overrepresented class.
#         inds_class_under = np.argwhere(this_task_labels == class_low_samples).squeeze()
#         inds_class_over  = np.argwhere(this_task_labels == class_high_samples).squeeze()
        
#         # Randomly downsampling the indices of the overrepresented class.
#         inds_class_over_downsampled = sample(inds_class_over.tolist(), n_samples_class_under)
        
#         # Extracting the randomly downsampled features and labels of the overrepresented class.
#         features_class_over_downsampled = this_task_features[:,:,inds_class_over_downsampled]
#         labels_class_over_downsampled   = this_task_labels[inds_class_over_downsampled]
        
#         # Extracting the features and labels of the underrepresented class.
#         features_class_under = this_task_features[:,:,inds_class_under]
#         labels_class_under   = this_task_labels[inds_class_under]
        
#         # Concatenating the downsampled overrepresented class with the underrepresented class.
#         this_task_features_downsampled = np.concatenate((features_class_over_downsampled, features_class_under), axis=2)
#         this_task_labels_downsampled   = np.concatenate((labels_class_over_downsampled, labels_class_under), axis=0)
        
#         # Extracting the number of historical time points, features, and samples from the resulting downsampled feature array.
#         n_history  = this_task_features_downsampled.shape[0]
#         n_features = this_task_features_downsampled.shape[1]
#         n_samples  = this_task_features_downsampled.shape[2]
        
#         # Transforming the features and labels arrays into xarrays.
#         this_task_features_downsampled = xr.DataArray(this_task_features_downsampled, 
#                                                       coords={'history': np.arange(n_history), 'feature': np.arange(n_features), 'sample': np.arange(n_samples)}, 
#                                                       dims=["history", "feature", "sample"])
        
#         this_task_labels_downsampled = xr.DataArray(this_task_labels_downsampled, 
#                                                     coords={'sample': np.arange(n_samples)}, 
#                                                     dims=["sample"])
                
#         # Updating the features and labels dictionaries with the downsampled corresponding arrays.
#         features_dict[this_task] = this_task_features_downsampled
#         labels_dict[this_task]   = this_task_labels_downsampled
        
#     return features_dict, labels_dict





def equalizing_samples_per_class(features_dict, labels_dict):
    """
    DESCRIPTION:
    According to the label array, it is unlikely that there are an equal number of samples per class. For whichever
    class there are more samples, the indices of this class will be extracted to create a smaller subset of 
    indices in number equal to that of the underrepresented class. For example, consider the following labels array:

    labels:  ['rest', 'rest', 'rest', 'rest', 'rest, 'grasp', 'grasp', 'grasp', 'rest', 'rest', 'rest', 'rest']
    indices:    0       1       2       3       4       5        6        7       8       9       10      11

    In the example, there are 3 samples with grasp labels, while there are 9 samples with rest labels. The indices
    corresponding to the rest labels are: [0, 1, 2, 3, 4, 8, 9, 10, 11]. These indices will be randomly subsampled
    such that they are equal in number to the grasp class. For example: [0, 1, 2] or [1, 4, 10] or [3, 9, 11]. As
    such the downsampled labels (and corresponding features) will use the indices:

    labels downsampled:  ['rest', 'grasp', 'grasp', 'grasp', 'rest', 'rest']
    indices downsampled:    3        5        6        7       9       11

    INPUT VARIABLES:
    features_dict: [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats (units: V^2/Hz))]
                   Array of historical time features.
    labels_dict:   [dictionary (Key: string (task ID); Value: xarray > strings ('grasp'/'rest'))]; For each time sample in each
                   task, there exists a rest or grasp label depending on the experimenter-specified onset and offset of
                   modulation as well as the per-trial shift from the AW model.

    OUTPUT VARIABLES:
    features_dict: [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats (units: V^2/Hz))]
                   Array of historical time features. Time samples reduced such that there are an equal number of features per
                   class.
    labels_dict:   [dictionary (Key: string (task ID); Value: xarray > strings ('grasp'/'rest'))]; For each time sample in each
                   task, there exists a rest or grasp label depending on the experimenter-specified onset and offset of
                   modulation as well as the per-trial shift from the AW model. Time samples reduced such that there are an equal
                   number of features per class.
    """
    
    # COMPUTATION
    
    # Extracting the unique labels from task0. These unique labels should be the same across all tasks.
    unique_labels = np.unique(labels_dict['task0'])
    
    # Iterating across each task.
    for this_task in labels_dict.keys():
        
        # Extracting the features and labels for the current task.
        this_task_features = np.asarray(features_dict[this_task])
        this_task_labels   = np.asarray(labels_dict[this_task])
        
        # Initializing an dictionary describing the number of samples per label.
        samples_per_label = {}
        
        # Iterating across all unique labels to determine the number of samples per label (rest and grasp).
        for this_label in unique_labels:
            
            # Counting the number of samples for the current label and appropriately updating the dictionary.
            samples_per_label[this_label] = sum(this_task_labels == this_label)
            
        # Check if the neutral labels exists at all. If so, delete this dictionary key because these labels will not be used for
        # training a model.
        if 'neutral' in samples_per_label:
            del samples_per_label['neutral']
                
        # Determining which class has the higher and lower number of samples (probably going to be rest and grasp, respectively).
        class_low_samples  = min(samples_per_label, key=samples_per_label.get)
        class_high_samples = max(samples_per_label, key=samples_per_label.get)
                
        # Extracting the number of samples of the underrespresented class. This will be used to determine how many samples 
        # of the overrepresented class to randomly extract.
        n_samples_class_under = samples_per_label[class_low_samples] 
        
        # Extracting the sample indices of the under and overrepresented class.
        inds_class_under = np.argwhere(this_task_labels == class_low_samples).squeeze()
        inds_class_over  = np.argwhere(this_task_labels == class_high_samples).squeeze()
        
        # Randomly downsampling the indices of the overrepresented class.
        inds_class_over_downsampled = sample(inds_class_over.tolist(), n_samples_class_under)
        
        # Extracting the randomly downsampled features and labels of the overrepresented class.
        features_class_over_downsampled = this_task_features[:,:,inds_class_over_downsampled]
        labels_class_over_downsampled   = this_task_labels[inds_class_over_downsampled]
        
        # Extracting the features and labels of the underrepresented class.
        features_class_under = this_task_features[:,:,inds_class_under]
        labels_class_under   = this_task_labels[inds_class_under]
        
        # Concatenating the downsampled overrepresented class features and labels with those from the underrepresented class.
        this_task_features_downsampled = np.concatenate((features_class_over_downsampled, features_class_under), axis=2)
        this_task_labels_downsampled   = np.concatenate((labels_class_over_downsampled, labels_class_under), axis=0)
        
        # Extracting the number of historical time points, features, and samples from the resulting downsampled feature array.
        n_history  = this_task_features_downsampled.shape[0]
        n_features = this_task_features_downsampled.shape[1]
        n_samples  = this_task_features_downsampled.shape[2]
        
        # Converting the features and labels arrays into xarrays.
        this_task_features_downsampled = xr.DataArray(this_task_features_downsampled, 
                                                      coords={'history': np.arange(n_history), 'feature': np.arange(n_features), 'sample': np.arange(n_samples)}, 
                                                      dims=["history", "feature", "sample"])
        
        this_task_labels_downsampled = xr.DataArray(this_task_labels_downsampled, 
                                                    coords={'sample': np.arange(n_samples)}, 
                                                    dims=["sample"])
                
        # Updating the features and labels dictionaries with the downsampled corresponding arrays.
        features_dict[this_task] = this_task_features_downsampled
        labels_dict[this_task]   = this_task_labels_downsampled
        
    return features_dict, labels_dict





def evaluating_model_accuracy(fold_models, valid_data_folds, valid_labels_folds):
    """
    DESCRIPTION:
    Evaluating model accuracy by computing and displaying the confusion matrix of all predicted vs. true labels from
    all folds.
    
    INPUT VARIABLES:
    fold_models:        [dictionary (key: string (fold ID); Value: model)]; Models trained for each training fold.
    valid_data_folds:   [dict (key: string (fold ID); Value: xarray (dimensions vary based on model type) > floats (units: V^2/Hz))];
                        Data across all validation tasks per fold. Equal number of samples per class. PC features. Rearranged 
                        according to the type of model that will be trained.
    valid_labels_folds: [dict (key: string (fold ID); Value: xarray (1 x time samples) > strings ('grasp'/'rest'))]; Labels across
                        all validation tasks per fold. Equal number of labels per class.
                        
    GLOBAL PARAMETERS:
    model_classes:  [list > strings]; List of all the classes to be used in the classifier.
    
    NECESSARY FUNCTIONS:
    computing_predicted_labels
    confusion_matrix_display
    """
    
    # COMPUTATION:
    
    # Across all folds, computing the predicted labels and extracting the true labels.
    pred_labels, true_labels = computing_predicted_labels(fold_models, valid_data_folds, valid_labels_folds)
    
    # Computing the confusion matrix between rest and grasp.
    this_confusion_matrix = confusion_matrix(true_labels, pred_labels, labels = model_classes);
    
    # Plotting the confusion matrix
    confusion_matrix_display(this_confusion_matrix)


    
    
    
def extract_data_pathways(data_info_dict):
    """
    DESCRIPTION:
    Using the dictionary (grasp_info_dict or calib_info_dict) of dates and tasks, the appropriate data file pathways are 
    extracted and stored in a list.
    
    INPUT VARIABLES:
    data_info_dict: [dictionary (Key: string (date in YYYY_MM_DD format); Values: list > string (task names); Values and
                    keys correspond to grasp/calibration tasks and dates on which those tasks were run.
    
    GLOBAL PARAMETERS:
    file_extension: [string (hdf5/mat)]; The data file extension of the data.
    patient_id:     [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    
    OUTPUT VARIABLES:
    path_list: [list > strings (pathways)]; The list of file pathways from which the dataset(s) will be extracted.
    """
    
    # COMPUTATION:

    # Initializing a list of pathways in which the data from the data_info_dict is found.
    path_list = []

    # Iterating across all dictionary items to extract the date and task.
    for date, tasks in data_info_dict.items():

        # Iterating across each task in the task list for the particular date.
        for this_task in tasks:

            # Creating the path for the current date/task pair. Modify this variable as needed.
            path = '/mnt/shared/ecog/' + patient_id + '/' + file_extension + '/' + date + '/' + this_task + '.' + file_extension

            # Appending the datafile path list with the current datafile path.
            path_list.append(path)
    
    return path_list





def extracting_saliencymap_powerband(grasp_bandpower_dict, saliency_powerband):
    """
    DESCRIPTION:
    Extracting from the band power dictionary only the powerband used for saliency mapping.
    
    INPUT VARIABLES:
    grasp_bandpower_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_power:        [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each frequency band,
                          the band power is computed for each channel across every time point.
        sxx_power_z:      [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each standardized (to
                          calibration) frequency band, the band power is computed for each channel across every time point.
        sxx_states:       [xarray (1 x time samples) > strings ('state_ON'/'state_OFF'/'neutral')]; Stimulus array downsampled
                          to match time resolution of the signal spectral power. Time dimension is in units of seconds.
        power_trials_z:   [xarray (trials x channels x frequency bins x time samples) > floats (units: V^2/Hz)]; The spectral
                          information for each trial. Time dimension is in units of seconds.
    saliency_powerband:   [string ('powerbandX', where X is an int)]; Powerband ID for saliency map plotting.
    
    OUTPUT VARIABLES:
    grasp_bandpower_dict_saliencymap: Same as grasp_bandpower_dict with the only difference being that only the powerband for
                                      saliency mapping is included.
    """
    
    # COMPUTATION:
    
    # Initializing the bandpower dictionary for the saliency map
    grasp_bandpower_dict_saliencymap = copy.deepcopy(grasp_bandpower_dict)

    # Iterating across all tasks.
    for this_task in grasp_bandpower_dict.keys():
        
        # Extracting only the band power information corresponding to the saliency powerband.
        sxx_power_saliency      = grasp_bandpower_dict[this_task]['sxx_power'].loc[:,saliency_powerband,:]
        sxx_power_z_saliency    = grasp_bandpower_dict[this_task]['sxx_power_z'].loc[:,saliency_powerband,:]
        power_trials_z_saliency = grasp_bandpower_dict[this_task]['power_trials_z'].loc[:,:,saliency_powerband,:]

        # Replacing the powerband information in the dictionary with only the saliency powerband information.
        grasp_bandpower_dict_saliencymap[this_task]['sxx_power']      = np.expand_dims(sxx_power_saliency, axis=1)
        grasp_bandpower_dict_saliencymap[this_task]['sxx_power_z']    = np.expand_dims(sxx_power_z_saliency, axis=1)
        grasp_bandpower_dict_saliencymap[this_task]['power_trials_z'] = np.expand_dims(power_trials_z_saliency, axis=1)

    return grasp_bandpower_dict_saliencymap





def import_electrode_information(data_info_dict):
    """
    DESCRIPTION:
    The eeglabels and auxlabels lists will be populated with eeg channel names and auxilliary channel names respectively.
    These lists are created differently based on whether the data is extracted from a .hdf5 file or a .mat file.

    INPUT VARIABLES:
    data_info_dict: [dictionary (Key: string (date in YYYY_MM_DD format); Values: list > string (task names); Values and
                    keys correspond to grasp/calibration tasks and dates on which those tasks were run.
    
    GLOBAL PARAMETERS:
    file_extension: [string (hdf5/mat)]; The data file extension of the data.

    OUTPUT VARIABLES:
    auxlabels: [array > strings (aux channel names)]: Auxilliary channels extracted from the .hdf5 or .mat file.
    eeglabels: [list > strings (eeg channel names)]: EEG channels extracted from the .hdf5 or .mat file.
    """
    
    # COMPUTATION:
    
    # Defining the ecog object.
    ecog = ECoGConfig()
    
    # Extracting the data file pathway for the first task. We don't need information from any other files because the
    # eeglabels and auxlabels hould be the same across multiple dates as they are related to the same participant with
    # the hardware setup.
    date = list(data_info_dict.keys())[0]
    task = data_info_dict[list(data_info_dict.keys())[0]][0]
    path = '/mnt/shared/ecog/' + patient_id + '/' + file_extension + '/' + date + '/' + task + '.' + file_extension

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
    data_folds:      [dict (key: string (fold ID); Value: xarray (time history x features x time samples) > floats (units: V^2/Hz))];
                     For each fold, feature xarrays are concatenated in the sample dimension.
    data_fold_means: [dict (Key: string (fold ID); Value: xarray (time history x features) > floats (units: V^2/Hz))]; The mean,
                     averaged across samples, for each fold. 
                     
    NECESSARY FUNCTIONS:
    mean_centering

    OUTPUT VARIABLES:
    data_folds_centered: [dict (key: string (fold ID); Value: xarray (time history x features x time samples) > floats (units: V^2/Hz))];
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
    data: [xarray (history x features x samples) > floats (units: V^2/Hz)]; Historical power features across time samples.
    
    OUTPUT VARIABLES:
    data_mean_history: [xarray (history x features) > floats (units: V^2/Hz)]; Mean power of each feature of only the 
                       0th time shift. This array is repeated for each historical time point.
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
    data_folds: [dict (key: string (fold ID); Value: xarray (time history x features x time samples) > floats (units: V^2/Hz))];
                For each fold, feature xarrays are concatenated in the sample dimension.
                
    NECESSARY FUNCTIONS:
    mean_compute

    OUTPUT VARIABLES:
    data_fold_means: [dict (Key: string (fold ID); Value xarray (time history x features) > floats (units: V^2/Hz))]; The mean,
                     averaged across samples, for each fold. 
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
    training_data:     [xarray (history, features, training time samples] > floats (units: V^2/Hz)]; 
    training_labels:   [xarray (1 x training samples) > strings ('grasp'/'rest')]
    validation_data:   [xarray (history, features, validation time samples] > floats (units: V^2/Hz)]; 
    validation_labels: [xarray (1 x validation samples) > strings ('grasp'/'rest')]
    
    GLOBAL PARAMETERS:
    model_classes: [list > strings]; List of all the classes to be used in the classifier.
    
    OUTPUT VARIABLES:
    model:         [classification model];
    model_classes: [list > strings]; Unique model classes.
    """
    
    # COMPUTATION
    
    # Extracting the hyperparameters.
    alpha         = params_model_training.alpha 
    batch_size    = params_model_training.batch_size
    dropout_rate  = params_model_training.dropout
    epochs        = params_model_training.epochs 
    n_hidden_lstm = params_model_training.n_hidden_lstm
    
    # Converting the training and validation data and labels from xarrays to arrays.
    training_data     = np.array(training_data)
    training_labels   = np.array(training_labels)
    validation_data   = np.array(validation_data)
    validation_labels = np.array(validation_labels)
    
    # Computing the number of classes.
    n_model_classes = len(model_classes)
    
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
    data:         [xarray (features x time samples) > floats (units: V^2/Hz)];
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





def pc_transform_all_folds(train_data_folds, valid_data_folds):
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
    train_data_folds: [dict (key: string (fold ID); Value: xarray (time history x features x time samples) > floats (units: V^2/Hz))];
                      The training data across all training tasks for each fold. 
    valid_data_folds: [dict (key: string (fold ID); Value: xarray (time history x features x time samples) > floats (units: V^2/Hz))];
                      The validation data across all training tasks for each fold. 
    
    GLOBAL PARAMETERS:
    n_pc_thr:        [int]; The number of principal components to which the user wishes to reduce the data set. Set to 'None' if
                     percent_var_thr is not 'None', or set to 'None' along with percent_var_thr if all of the variance will be used
                     (no PC transform).
    percent_var_thr: [float]; The percent variance which the user wishes to capture with the principal components. Will compute the
                     number of principal components which capture this explained variance as close as possible, but will not surpass
                     it. Set to 'None' if n_pc_thr is not 'None', or set to 'None' along with n_pc_thr if all of the variance will be
                     used (no PC transform).
    
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
        this_fold_eigenvectors = computing_eigenvectors(train_data_history0)
        
        # Replacing the training and validation data with the PC transformed for the current fold.
        train_data_folds[this_fold] = pc_transform(this_fold_train_data, this_fold_eigenvectors)
        valid_data_folds[this_fold] = pc_transform(this_fold_valid_data, this_fold_eigenvectors)
    
    return train_data_folds, valid_data_folds





def plotting_channel_contributions(ch_importance_scores, chs_exclude, chs_include, marker_color):
    """
    DESCRIPTION:
    Mapping the channel importance scores at the coordinates of each electrode on an image of the research participant's.
    brain. Size and opaqueness of the markers on the image are proportional to that channel's importance in the neural
    network for classifying the experimenter-specified class. Channel importance scores are normalized between 0 and 1.
    
    INPUT VARIABLES:
    ch_importance_scores: [array (1 x channels) > floats]; Mean importance score for each channel, averaged across all
                          time samples from all validation folds.
    chs_exclude:          [list > strings]; The list of channels to be excluded in further analysis.
    chs_include:          [list > strings]; The list of channels to be included in further analysis.
    marker_color:         [list > ints (R, G, B)]; Marker color represented by RBG components. Each integer value ranges 
                          from 0 to 255.
    """
    
    # COMPUTATION:
    
    
    # Defining the directory for the information for mapping channel importance.
    dir_brain_mapping = 'SaliencyMapInfo/'
    
    # Extracting the image of the brain without electrodes.
    path_brain_no_electrodes = dir_brain_mapping + 'CC01_BrainWithoutGrids.jpg'

    # Extracting the coordinates of each channel on the brain image.
    path_electrode_coords_on_brain = dir_brain_mapping + 'ch_coords_on_brain'
    
    # Extracting the map of each channel to each coordinate.
    path_electrode_mapping_on_brain = dir_brain_mapping + 'ch_map_to_coords'
    
    # Opening up the brain image which does not contain electrode locations.
    brain_image = Image.open(path_brain_no_electrodes).convert('RGBA')
    
    # Read in electrode coordinates on the brain.
    with open(path_electrode_coords_on_brain,'rb') as fp:
        electrode_coords = pickle.load(fp)
    
    # Read in the electrode mappings to the coordinates on the brain.
    with open(path_electrode_mapping_on_brain,'rb') as fp:
        electrode_mapping = pickle.load(fp)
        
    
    # Sorting the list of included channels according to increasing channel importance scores.
    chs_include_sorted = [x for _, x in sorted(zip(ch_importance_scores, chs_include))] 
    
    # Combining the sorted included channels and excluded channels.
    chs_all_sorted = chs_exclude + chs_include_sorted 
    chs_all_sorted = [x for x in chs_all_sorted if 'chan' in x]
        
    # Extracting the maximum channel importance score from all channels.
    ch_importance_max = np.max(ch_importance_scores)
    
    # Initializing a dictionary containing all the normalized channel contributions.
    ch_importance_scores_norm = {}
        
    # Computing the normalized channel importance scores based on the only the maximum importance score from all channels.
    for this_channel in chs_include:
        
        # Extracting the channel index from chs_include list for pulling out the corresponding importance score from
        # the ch_importance_scores array.
        ch_idx = chs_include.index(this_channel)
        
        # Extracting the current channel's importance score.
        this_channel_importance = ch_importance_scores[ch_idx]
        # print('Channel Importance: ', this_channel_importance)
        
        # Updating the dictionary of normalized channel importance scores with the current channel's normalized 
        # importance score.
        ch_importance_scores_norm[this_channel] = (this_channel_importance-0)/(ch_importance_max-0) * 0.99
    

    # Sorting the list of included channels according to increasing channel contributions.
    
    # Printing out the importance scores of each channel in ascending order.
    ch_importance_scores_sorted = [np.round(ch_importance_scores_norm[ch], 5) for ch in chs_include_sorted]
    for this_ch, this_importance_score in zip(chs_include_sorted, ch_importance_scores_sorted):
        print(this_ch,':', this_importance_score)
    print('\n')
    
    
    # PLOTTING
    
    # Plotting the channel contributions on the brain image.
    fig = plt.figure()
    
    # Iterating across all sorted channels, both included and excluded.
    for this_channel in chs_all_sorted: 
        
        # Extracting the coordinates and radius for the current channel.
        ch_idx  = electrode_mapping.index(this_channel)
        x, y, r = electrode_coords[ch_idx,:]
            
        # If this is an included channel
        if this_channel in chs_include:

            # Extracting the current channel's importance score.
            this_channel_importance = ch_importance_scores_norm[this_channel]

            # Modifying the radius and transparency such that it is proportional to the channel importance.
            radius       = r * (1 + 2*this_channel_importance)
            transparency = int(135*this_channel_importance + 120)
            
            # Making the marker color appropriately transparent.
            this_channel_color = tuple(np.concatenate((marker_color, np.array([transparency])), axis=0))
    
        # If this is an excluded channel.
        if this_channel in chs_exclude:
            
            # Setting the radius simply to r.
            radius = r

            # Making the marker color white.
            this_channel_color = tuple(np.concatenate(([255, 255, 255], np.array([128])), axis=0))
            
        # Setting the bounds of the ellipse from the center of the channel marker.
        upper_left_x  = int(x-radius)
        upper_left_y  = int(y-radius)
        lower_right_x = int(x+radius)
        lower_right_y = int(y+radius)
        
        # Combining these ellipse bounds into a tuple.
        ellipse_bounds = (upper_left_x, upper_left_y, lower_right_x, lower_right_y)
        
        # Drawing the transparent ellipse on the brain image.
        draw_transparent_ellipse(brain_image, ellipse_bounds, fill=this_channel_color, outline=(54,69,79,255), width=9)
    
    # Plotting brain image with channel contribution markers
    plt.imshow(brain_image)
    plt.axis('off')
    
    # Transforming the image back to RGB 
    brain_image = brain_image.convert('RGB')
    
    # SAVING:
    # Saving the brain image with the channel contribution markers.
    # brain_image.save('brain_image_with_channel_contributions.jpg', 'JPEG', quality=95)

    return None
    
    
    
    
    
def plotting_states(calib_cont_dict, grasp_cont_dict, task_id):
    """
    DESCRIPTION:
    Plotting the stimuli of an experimenter-specified task such as to observe how many state values there are. This
    will inform the state mapping.
    
    INPUT VARIABLES:
    calib_cont_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        signals:     [xarray (channels x time samples (units: s) > floats (units: microvolts)]; Array of continuous
                     voltage signals from the calibration tasks. Time dimension is in units of seconds.
        states:      [xarray (1 x time samples) > ints (0 or 1)]; Array of states at each time sample for the calibration
                     tasks. Time dimension is in units of seconds.
    grasp_cont_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        signals:     [xarray (channels x time samples (units: s) > floats (units: microvolts)]; Array of continuous
                     voltage signals from grasp-based tasks. Time dimension is in units of seconds.
        states:      [xarray (1 x time samples) > ints (0 or 1)]; Array of states at each time sample for the grasp-based
                     tasks. Time dimension is in units of seconds.
    task_id:         [string (task ID)]; Task ID corresponding to the states that the experimenter wishes to see plotted.
    """
    
    # COMPUTATION:
    
    # Extracting the time points and states for the grasp task and corresponding calibration task.
    this_calib_task_states = calib_cont_dict[task_id]['states']
    this_calib_task_time   = this_calib_task_states.time
    this_grasp_task_states = grasp_cont_dict[task_id]['states']
    this_grasp_task_time   = this_grasp_task_states.time

    # PLOTTING
    
    # Plotting the states from the calibration task.
    fig = plt.figure(figsize=(20,5))
    plt.plot(this_calib_task_time, this_calib_task_states)
    plt.xlabel('Time (s)')
    plt.ylabel('State Values')
    plt.title('Calibration States')
    

    # Plotting the states from the grasp-based task.
    fig = plt.figure(figsize=(20,5))
    plt.plot(this_grasp_task_time[30000:40000], this_grasp_task_states[30000:40000])
    plt.xlabel('Time (s)')
    plt.ylabel('State Values')
    plt.title('Grasp-based Task States')

    return None





def post_state_neutral_labeling(grasp_cont_dict, t_neutral_interval):
    """
    DESCRIPTION:
    This function allows the experimenter to assign 'neutral' labels to any time period after a state (cue or click) occurs. Note that this
    function was written specifically for training decoders from real-time use (control clicks with speller), where two clicks may have 
    occurred one immediately after the other. In this situation, the data from the second click should not be used to train a click model 
    because of increased neural activity that may have not fallen to baseline since the first click occurred. In this situation, 'neutral' 
    labels would overwrite all time points from the first click to some time period past the second click. For example, consider the
    following:
      
    state:               0         0         1         1         0         1         1         0         0         0         1        1        0
    time sample:         0         1         2         3         4         5         6         7         8         9         10       11       12
    original labels: state_OFF state_OFF state_ON  state_ON  state_OFF state_ON  state_ON  state_OFF state_OFF state_OFF state_ON state_ON state_OFF
    modified labels: state_OFF state_OFF state_ON  state_ON  neutral   neutral   neutral   neutral   state_OFF state_OFF state_ON state_ON neutral
    
    In the above example (sampling resolution not to scale), the first click occurred at samples 2 and 3, while the second click occurred
    at samples 5 and 6. The second click occurred to soon after the first click. Therefore, after the end of the first click, the 'neutral'
    label is applied for some samples, which wipes away the 'state_ON' labels from the second click. Therefore, those 'state_ON' labels will
    not be used to train the model. The third click, occurring at samples 10 and 11, occurs after a longer period of time after the second
    click, but the 'neutral' labels are still applied at the end of the third click.
    
    The number of samples which will be replaced by 'neutral' corresponds to the experimenter-determined time, t_neutral_interval, which 
    is in units of seconds.
    
    INPUT VARIABLES:
    grasp_cont_dict:    [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        signals         [xarray (channels x time samples (units: s) > floats (units: microvolts)]; Array of continuous
                        voltage signals from grasp-based tasks. Time dimension is in units of seconds.
        states:         [xarray (1 x time samples) > strings ('state_ON'/'state_OFF')]; Array of states at each time sample for the
                        grasp-based tasks. Time dimension is in units of seconds.
    t_neutral_interval: [float (units: s)]; Amount of time after state offset that will be labeled as neutral.             
    
    NECESSARY FUNCTIONS:
    unique_value_index_finder
    
    OUTPUT VARIABLES:
    grasp_cont_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        signals:     [xarray (channels x time samples (units: s) > floats (units: microvolts)]; Array of continuous
                     voltage signals from grasp-based tasks. Time dimension is in units of seconds.
        states:      [xarray (1 x time samples) > strings ('state_ON'/'state_OFF'/'neutral')]; Array of states at each time sample for the
                     grasp-based tasks. Time dimension is in units of seconds.
    """
    
    # COMPUTATION:
    
    # Computing the number of samples in a neutral labeling time interval.
    n_samples_neutral = int(t_neutral_interval * sampling_rate)
    
    # If the experimenter uses a neutral labeling interval time greater than 0 s.
    if t_neutral_interval > 0:
            
        # Iterating across each task.
        for this_task in grasp_cont_dict.keys():
                    
            # Extracting the states xarray and times for the current task.
            this_task_states = grasp_cont_dict[this_task]['states']
            this_task_times  = this_task_states.time
            
            # Converting the states xarray to an array for faster processing when using unique_value_index_finder.
            this_task_state_array = np.asarray(this_task_states)
            
            # Computing the number of samples in the current task.
            n_samples_this_task = this_task_state_array.shape[0]
            
            # Extracting the onset and offset indices for the 'state_OFF' and 'state_ON' periods for the current task.
            _, _, _,\
            this_task_onset_offset_inds = unique_value_index_finder(this_task_state_array) 
            
            # Extracting the onset/offset indices of the state_ON elements.
            state_on_inds = this_task_onset_offset_inds['state_ON']
            
            # Computing the total number of state on periods.
            n_state_on = len(state_on_inds)
            
            # Initializing the offset time of the previous ON state which preceeded a neutral interval. See IF statements in the FOR loop.
            t_state_ON_offset_old = 0
            
            # Iterating across all state_ON indices, except for the last.    
            for n in range(n_state_on-1):
                
                # Extracting the 
                # onset_idx  = state_on_inds[n][0]
                
                
                # Extracting the onset and offset index of the current ON period.
                state_on_onset_idx  = state_on_inds[n][0]
                state_on_offset_idx = state_on_inds[n][1]
                
                # Extracting the onset index of the next ON state
                state_on_onset_idx_next = state_on_inds[n+1][0]
                
                
                # Extracting the onset and offset time of the current ON period.
                t_state_ON_onset  = this_task_times[state_on_onset_idx]
                t_state_ON_offset = this_task_times[state_on_offset_idx] 
                
                
                # Extracting the onset time of the next ON period.
                t_state_ON_onset_next = this_task_times[state_on_onset_idx_next]
                                
                    
                
                # at least after the neutral stimulus duration following the last stimulus offset time.                
                # if t_stim_ON_offset > t_stim_ON_offset_old + t_neutral_post_stim:
                
                
                # If the onset of the current ON period has occurred after the previous neutral time period. For example, in the description, the ON
                # state at time sample 10 occurs after the previous neutral period from time samples 4-7 (initiated from the ON period from samples 2
                # and 3). The neutral labels override the ON labels during samples 5 and 6.
                if t_state_ON_onset > t_state_ON_offset_old + t_neutral_interval:
                    
                    # Apply the neutral interval only if the difference in time between the offset of the current ON period and the onset of the next 
                    # time period is less than the neutral period. For example, in the description, the onset of the ON state at time sample 5 occurs 
                    # too soon after the offset of the previous ON state at time sample 3. The difference between these two samples is 2, which is less
                    # than the neutral period of 4 samples (only in the example in the description).
                    if t_state_ON_onset_next - t_state_ON_offset < t_neutral_interval:
                    
                        # Computing the onset and offset indices of the neutral labeled period starting immediately after the current ON state offset
                        # (specifically, the first time sample after the ON state). 
                        onset_idx_neutral  = state_on_offset_idx + 1
                        offset_idx_neutral = onset_idx_neutral + n_samples_neutral
                        
                        # Only apply the neutral period if the end of the neutral period happens before the final time sample of the current task.
                        if offset_idx_neutral < n_samples_this_task:

                            # Updating the state array for the current task with the neutral period.
                            this_task_state_array[onset_idx_neutral:offset_idx_neutral] = ['neutral'] * n_samples_neutral 

                        # Updating offset time of the previous ON state which preceeded a neutral interval.
                        t_state_ON_offset_old = t_state_ON_offset
                    
                else:
                    pass
                
            # Updating the states with the neutral periods for the current task.
            grasp_cont_dict[this_task]['stimuli'] = this_task_state_array

    else:
        pass

    return grasp_cont_dict





def power_generator(grasp_sxx_dict):
    """
    DESCRIPTION:
    Given the experimenter-specified minimum and maximum frequencies, the band power from the unstandardized and calibration-
    standardized spectrograms are computed.
    
    INPUT VARIABLES:
    grasp_sxx_dict:    [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_signals:   [xarray (channels x frequency bins x time samples) > floats (units: V^2/Hz)]; Spectral power of the
                       continuous voltage signals from the grasp-based task. Time dimension is in units of seconds.
        sxx_signals_z: [xarray (channels x frequency bins x time samples) > floats (units: V^2/Hz)]; Standardized Spectral
                       power of the continuous voltage signals from grasp-based task.
        sxx_states:    [xarray (1 x time samples) > strings ('state_ON'/'state_OFF'/'neutral')]; States array downsampled
                       to match time resolution of the signal spectral power. Time dimension is in units of seconds.
                       
    GLOBAL PARAMETERS:
    f_power_max: [list > int (units: Hz)]; For each frequency band, maximum power band frequency.
    f_power_min: [list > int (units: Hz)]; For each frequency band, minimum power band frequency.
        
    OUTPUT VARIABLES:
    grasp_bandpower_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_power:        [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each frequency band,
                          the band power is computed for each channel across every time point.
        sxx_power_z:      [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each standardized (to
                          calibration) frequency band, the band power is computed for each channel across every time point.
        sxx_states:       [xarray (1 x time samples) > strings ('state_ON'/'state_OFF'/'neutral')]; Stimulus array downsampled
                          to match time resolution of the signal spectral power. Time dimension is in units of seconds.
    """
    
    # COMPUTATION:
    
    # Initializing the dictionary of power bands for each task.
    grasp_bandpower_dict = collections.defaultdict(dict)
    
    # Iterating across all tasks. 
    for this_task in grasp_sxx_dict.keys():
        
        # Extracting the raw and calibration-standardized spectrograms.
        sxx_signals   = grasp_sxx_dict[this_task]['sxx_signals']
        sxx_signals_z = grasp_sxx_dict[this_task]['sxx_signals_z']
        
        # Extracting the channels and times.
        channels = sxx_signals.channel
        times    = sxx_signals.time
        
        # Creating the names of the powerbands.
        powerbands = ['powerband'+str(n) for n in range(len(f_power_max))]
                
        # Computing the number of channels, powerbands and time samples.
        n_bands = len(powerbands)
        n_chs   = channels.shape[0]
        n_times = times.shape[0]
                
        # Initializing the power and standardized power xarrays.
        power   = xr.DataArray(np.zeros((n_chs, n_bands, n_times)), 
                               coords={'channel': channels, 'powerband': powerbands, 'time': times}, 
                               dims=["channel", "powerband", "time"])
        
        power_z = xr.DataArray(np.zeros((n_chs, n_bands, n_times)), 
                               coords={'channel': channels, 'powerband': powerbands, 'time': times}, 
                               dims=["channel", "powerband", "time"])
        
        # Extracting the frequency array from the spectral signals dictionary.
        f_sxx = sxx_signals.frequency.values
        
        # Iterating over every pair of frequency band bounds to compute the power within them.
        for f_band_ind, (this_freqband_power_min, this_freqband_power_max) in enumerate(zip(f_power_min, f_power_max)):
                        
            # Extracting the frequency range for the experimenter-input power band.
            f_range = np.logical_and(f_sxx > this_freqband_power_min, f_sxx < this_freqband_power_max) 
                    
            # Computing the power from the unstandardized and calibration-standardized spectrograms.
            this_freqband_power   = np.sum(sxx_signals[:,f_range,:], axis=1)
            this_freqband_power_z = np.sum(sxx_signals_z[:,f_range,:], axis=1)
            
            # Naming the power band key for the current feature band.
            powerband_id = 'powerband' + str(f_band_ind)
        
            # Assigning the power and z-scored power to the appropriate dictionaries.
            power.loc[:,powerband_id,:]   = this_freqband_power.values
            power_z.loc[:,powerband_id,:] = this_freqband_power_z.values

        # Updating the dictionary of power bands.
        grasp_bandpower_dict[this_task]['sxx_power']   = power
        grasp_bandpower_dict[this_task]['sxx_power_z'] = power_z
        
        # Putting the stimuli xarray in here as well.
        grasp_bandpower_dict[this_task]['sxx_states'] = grasp_sxx_dict[this_task]['sxx_states']
        
    return grasp_bandpower_dict
        
      
        
        
        
def power_info_per_trial(grasp_bandpower_dict, t_post_on_state, t_pre_on_state):
    """
    DESCRIPTION:
    Computing the per-trial band power. 
    
    INPUT VARIABLES:
    grasp_bandpower_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_power:        [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each frequency band,
                          the band power is computed for each channel across every time point.
        sxx_power_z:      [xarray (channel x powerband x time samples] > floats (units: V^2/Hz)]; For each standardized (to
                          calibration) frequency band, the band power is computed for each channel across every time point.
        sxx_states:       [xarray (1 x time samples) > strings ('state_ON'/'state_OFF'/'neutral')]; Stimulus array downsampled
                          to match time resolution of the signal spectral power. Time dimension is in units of seconds.
    t_post_on_state:      [float (units: s)]; The amount of time after the cue for visualizing the trial-averaged information.
    t_pre_on_state:       [float (units: s)]; The amount of time before the cue for visualizing the trial-averaged information.
    
    GLOBAL PARAMETERS:
    sxx_shift: [int (units: ms)]; Length of time by which sliding window (sxx_window) shifts along the time domain.

    OUTPUT VARIABLES:
    grasp_bandpower_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_power:      Same as input.
        sxx_power_z:    Same as input.
        sxx_states:     Same as input.        
        power_trials_z: [xarray (trials x channels x frequency bins x time samples) > floats (units: V^2/Hz)]; The spectral
                        information for each trial. Time dimension is in units of seconds.
    """
    
    # COMPUTATION:

    # Extracting the channels and powerbands from task0. These are the same across all tasks.
    channels   = grasp_bandpower_dict['task0']['sxx_power'].channel
    powerbands = grasp_bandpower_dict['task0']['sxx_power'].powerband
    
    # Computing the number of channels and powerbands.
    n_chs   = channels.shape[0]
    n_bands = powerbands.shape[0]
    
    # Creating the time array and corresponding number of samples which spans one trial.
    t_stepsize          = sxx_shift/1000 # ms / (ms/s) = ms * s/ms = s
    t_sxx_per_trial     = np.arange(t_pre_on_state, t_post_on_state, t_stepsize)
    n_samples_per_trial = t_sxx_per_trial.shape[0]
    
    # Iterating across each task.
    for this_task in grasp_bandpower_dict.keys():
            
        # Extracting the standardized powerbands of the current task.
        this_task_power = grasp_bandpower_dict[this_task]['sxx_power_z']
        
        # Extracting the time array of the current task.
        t_sxx = this_task_power.time
        
        # Computing the total number of time samples in the current task.
        n_samples_this_task = t_sxx.shape[0]
        
        # Extracting the starting and ending indices of all states.
        _, _, _, this_task_start_end_inds = unique_value_index_finder(grasp_bandpower_dict[this_task]['sxx_states'])
        
        # Computing the total number of trials by finding the total number of ON states.
        n_trials_this_task = len(this_task_start_end_inds['state_ON'])
        
        # Initialize the x-array containing the per-trial band power.
        power_trials_z = xr.DataArray(np.zeros((n_trials_this_task, n_chs, n_bands, n_samples_per_trial)), 
                                      coords={'trial': np.arange(n_trials_this_task),'channel': channels, 'powerband': powerbands, 'time': t_sxx_per_trial}, 
                                      dims=["trial", "channel", "powerband", "time"])
        
        # Iterating across each state onset/offset index pair of the current task.
        for tr, (state_onset_idx, _) in enumerate(this_task_start_end_inds['state_ON']):

            # Computing the onset time of the current trial state onset.
            t_state_onset_this_trial = t_sxx[state_onset_idx]
            
            # Computing the starting and ending times of the current trial.
            t_start_this_trial = t_state_onset_this_trial + t_pre_on_state             
                
            # Computing the starting and ending time samples for the current trial.
            sample_start = int(np.asarray(np.abs(t_sxx - t_start_this_trial).argmin()))
            sample_end   = sample_start + n_samples_per_trial
            
            # If the ending sample occurs occurs after the total number of samples in current task.
            if sample_end > n_samples_this_task:

                # Computing the difference between the total number of samples in the signal, and the end sample.
                n_samples_rem = sample_end - n_samples_this_task 

                # Extract the incomplete power information of this trial and creating an array of zeros for zero-padding
                # this incomplete trial.
                this_trial_power_incomplete = this_task_power[:,:,sample_start:n_samples_this_task].values
                this_trial_zero_padding     = np.zeros((n_chs, n_bands, n_samples_rem))
                
                # Zero-padding this trial's incomplete power information with 0s to achieve a full trial length.
                this_trial_power_padded = np.concatenate((this_trial_power_incomplete, this_trial_zero_padding), axis=2)
                
            # If the ending sample falls within the total number of samples in the current task.
            else:
                
                # Extracting the power for the current trial.
                this_trial_power = this_task_power[:,:,sample_start:sample_end].values
                
            # Updating the xarray with the current trial's band power.
            power_trials_z.loc[tr] = this_trial_power
        
        # Updating the grasp dictionary with the current task's per-trial band power.
        grasp_bandpower_dict[this_task]['power_trials_z'] = power_trials_z
    
    return grasp_bandpower_dict
        


    
    
def power_trial_raster_plotting(chs_exclude, fig_height, fig_width, powerband_id, ptr_all_trials, upperlimb_or_speech, v_max, v_min):
    """
    DESCRIPTION:
    Plotting the power trial rasters across all tasks.
    
    INPUT VARIABLES:
    chs_exclude:         [list > strings]; The list of channels to be excluded in further analysis and whose spectral
                         information will not be shown.
    fig_height:          [int]; The height of the subplot figure showing the trial averaged spectrograms.
    fig_width:           [int]; The width of the subplot figure showing the trial averaged spectrograms.
    powerband_id:        [string (powerband#)]; The index of the powerband.              
    ptr_all_trials:      [xarray (trials x channels x powerbands x time samples) > floats (units: V^2/Hz)]; The spectral
                         information for each trial (across all tasks). Time dimension is in units of seconds.
    upperlimb_or_speech: [string ('upperlimb'/'speech')]; Whether to plot the spectrograms of the upper-limb or speech grid. 
    v_max:               [int]; Maximum value for colorplot.
    v_min:               [int]; Minimum value for colorplot.
    """
    
    # COMPUTATION:
    
    # Uploading the electrode grid configuration.
    grid_config = params_model_training.grid_config_dict[patient_id][upperlimb_or_speech]
    grid_config = np.array(grid_config)
    
    # Flattening the grid of channels (nested list) and eliminated excluded channels from being displayed.
    channels = list(chain.from_iterable(grid_config))    
    channels = [ch for ch in channels if ch not in chs_exclude]
    
    # Extracting the trial indices.
    tr_inds = ptr_all_trials.trial.values
    
    # Extracting the power trial rasters from the dictionary.
    ptr = ptr_all_trials.loc[:,:,powerband_id,:]
    
    # Extract the time dimensions from the power trial rasters.
    t_sxx = ptr.time.values
    
    # PLOTTING
    
    # Initiate a subplot with the same dimensionality as the grid.
    grid_rows = grid_config.shape[0]
    grid_cols = grid_config.shape[1]
    fig, axs  = plt.subplots(grid_rows, grid_cols, figsize = (fig_width, fig_height));
    
    # Iterating across all channels of the experimenter-specified type.
    for ch in channels:

        # Extracting the channel index from the grid.
        row = np.where(grid_config == ch)[0][0]
        col = np.where(grid_config == ch)[1][0]

        # If the electrode grid has only one row.
        if grid_rows == 1:
            im = axs[col].pcolormesh(t_sxx, tr_inds, ptr.loc[:,ch,:], cmap = 'bwr', vmin = v_min, vmax = v_max);
            axs[col].set_title(ch)
            axs[col].axvline(x=0, color = 'k')

        # If the electrode grid has only one column. 
        elif grid_cols == 1:
            im = axs[row].pcolormesh(t_sxx, tr_inds, ptr.loc[:,ch,:], cmap = 'bwr', vmin = v_min, vmax = v_max);
            axs[row].set_title(ch)
            axs[row].axvline(x=0, color = 'k')

        # If there are multiple rows and columns in the grid.
        else:
            im = axs[row,col].pcolormesh(t_sxx, tr_inds, ptr.loc[:,ch,:], cmap = 'bwr', vmin=v_min, vmax=v_max);
            axs[row,col].set_title(ch)
            axs[row,col].axvline(x=0, color = 'k')



            
            
def rearranging_features(data):
    """
    DESCRIPTION:
    Rearranging the data dimensions as necessary to fit the experimenter-determined model.
    
    INPUT VARIABLES:
    data: [xarray (time history x features x time samples) > floats]; Array of historical time features.
    
    GLOBAL PARAMETERS:
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
    

    
    
    
def rearranging_features_all_folds(features_dict):
    """
    DESCRIPTION:
    Depending on the experimenter-specified model type the features array will be rearranged corresponding to the
    dimensions required for model.fit() 
    
    INPUT VARIABLES:
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
        features_dict[this_fold] = rearranging_features(this_task_features)
    
    return features_dict
        
    
    
    
    
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
    
    
    
    
    
def save_model(model, directory, filename):
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


        
        

def spectrogram_generator(data_cont_dict, sxx_tag):
    """
    DESCRIPTION:
    Creating the spectrogram for each channel across all tasks. Also downsampling the states array to match the 
    spectrogram resolution.
    
    INPUT VARIABLES:
    data_cont_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        signals:    [xarray (channels x time samples (units: s) > floats (units: microvolts)]; Array of continuous 
                    voltage signals. Time dimension is in units of seconds.
        states:     [xarray (1 x time samples) > ints (0 or 1)]; Array of states at each time sample. Time dimension
                    is in units of seconds.
    sxx_tag:        [string]; Marker to show experimter which type of data is in the pipeline.
    
    GLOBAL PARAMETERS:
    sxx_shift:  [int (units: ms)]; Length of time by which sliding window (sxx_window) shifts along the time domain.
    sxx_window: [int (units: ms)]; Time length of the window that computes the frequency power.
    
    OUTPUT VARIABLES:
    sxx_data_dict:   [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_signals: [xarray (channels x frequency bins x time samples) > floats (units: V^2/Hz)]; Spectral power of
                     the continuous voltage signals.
        sxx_states:  [xarray (1 x time samples) > ints (0 or 1)]; States array downsampled to match time resolution
                     of the signal spectral power.
    """
    
    # COMPUTATION:
    
    # Printing the type of data the spectral power is being computed.
    print('\n'+sxx_tag)
    
    # Initializing the dictionary of spectral information.
    sxx_data_dict = collections.defaultdict(dict)

    # Iterating across all tasks in the data dictionary.
    for this_task in data_cont_dict.keys():
        
        # Printing for which individual task the spectrogram is being computed.
        print(sxx_tag, ' ', this_task)

        # Extracting the signals and states of the current task from the data dictionary.
        signals = data_cont_dict[this_task]['signals']
        states  = data_cont_dict[this_task]['states']

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
        # the first spectral sample being associated with 128 ms in the t_sxx array (if sxx_window = 256), it will rather be
        # 256 ms.
        t_sxx = t_sxx + (sxx_window/2)/1000

        # Converting the spectral array into an xarray.
        sxx_signals = xr.DataArray(sxx, 
                                   coords={'channel': signals.channel, 'frequency': f_sxx, 'time': t_sxx}, 
                                   dims=["channel", "frequency", "time"])
        
        # Downsampling the states such as to be at the same time resolution as the spectral signals. The resulting time 
        # coordinates will be causal.
        sxx_states = states[sxx_window_samples-1::sxx_shift_samples]
        
        # Updating the data dictionary.
        sxx_data_dict[this_task]['sxx_signals'] = sxx_signals
        sxx_data_dict[this_task]['sxx_states']  = sxx_states
    
    return sxx_data_dict





def spectrogram_info_per_trial(grasp_sxx_dict, t_post_on_state, t_pre_on_state):
    """
    DESCRIPTION:
    Computing the per-trial spectrograms aligned to the experimenter-defined state onset. Bounded by the experimenter-
    input pre- and post-state onset times.
    
    INPUT VARIABLES:
    grasp_sxx_dict:    [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_signals:   [xarray (channels x frequency bins x time samples) > floats (units: V^2/Hz)]; Spectral power of the
                       continuous voltage signals from the grasp-based task. Time dimension is in units of seconds.
        sxx_signals_z: [xarray (channels x frequency bins x time samples) > floats (units: V^2/Hz)]; Standardized Spectral
                       power of the continuous voltage signals from grasp-based task.
        sxx_states:    [xarray (1 x time samples) > strings ('state_ON'/'state_OFF'/'neutral')]; States array downsampled
                       to match time resolution of the signal spectral power. Time dimension is in units of seconds.
    t_post_on_state:   [float (units: s)]; The amount of time after the cue for visualizing the trial-averaged information.
    t_pre_on_state:    [float (units: s)]; The amount of time before the cue for visualizing the trial-averaged information.
    
    GLOBAL PARAMETERS:
    sxx_shift: [int (units: ms)]; Length of time by which sliding window (sxx_window) shifts along the time domain.

    OUTPUT VARIABLES:
    grasp_sxx_dict:    [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_signals:   Same as input.
        sxx_signals_z: Same as input.
        sxx_states:    Same as input.        
        sxx_trials_z:  [xarray (trials x channels x frequency bins x time samples) > floats (units: V^2/Hz)]; The spectral
                       information for each trial. Time dimension is in units of seconds.
    """
    
    # COMPUTATION:

    # Extracting the channels and frequency bins from task0. These are the same across all tasks.
    channels = grasp_sxx_dict['task0']['sxx_signals'].channel
    freqs    = grasp_sxx_dict['task0']['sxx_signals'].frequency
    
    # Computing the number of channels and frequency bins.
    n_chs   = channels.shape[0]
    n_freqs = freqs.shape[0]
    
    # Creating the time array and corresponding number of samples which spans one trial.
    t_stepsize          = sxx_shift/1000 # ms / (ms/s) = ms * s/ms = s
    t_sxx_per_trial     = np.arange(t_pre_on_state, t_post_on_state, t_stepsize)
    n_samples_per_trial = t_sxx_per_trial.shape[0]

    # Iterating across each task.
    for this_task in grasp_sxx_dict.keys():
            
        # Extracting the standardized spectrograms of the current task.
        this_task_sxx = grasp_sxx_dict[this_task]['sxx_signals_z']
        
        # Extracting the time array of the current task.
        t_sxx = this_task_sxx.time
        
        # Computing the total number of time samples in the current task.
        n_samples_this_task = t_sxx.shape[0]
        
        # Extracting the starting and ending indices of all states.
        _, _, _, this_task_start_end_inds = unique_value_index_finder(grasp_sxx_dict[this_task]['sxx_states'])
        
        # Computing the total number of trials by finding the total number of ON states.
        n_trials_this_task = len(this_task_start_end_inds['state_ON'])
        
        # Initialize the x-array containing the per-trial spectrograms.
        sxx_trials_z = xr.DataArray(np.zeros((n_trials_this_task, n_chs, n_freqs, n_samples_per_trial)), 
                                    coords={'trial': np.arange(n_trials_this_task),'channel': channels, 'frequency': freqs, 'time': t_sxx_per_trial}, 
                                    dims=["trial", "channel", "frequency", "time"])
        
        # Iterating across each state onset/offset index pair of the current task.
        for tr, (state_onset_idx, _) in enumerate(this_task_start_end_inds['state_ON']):

            # Computing the onset time of the current trial state onset.
            t_state_onset_this_trial = t_sxx[state_onset_idx]
            
            # Computing the starting and ending times of the current trial.
            t_start_this_trial = t_state_onset_this_trial + t_pre_on_state             
                
            # Computing the starting and ending time samples for the current trial.
            sample_start = int(np.asarray(np.abs(t_sxx - t_start_this_trial).argmin()))
            sample_end   = sample_start + n_samples_per_trial
            
            # If the ending sample occurs occurs after the total number of samples in current task.
            if sample_end > n_samples_this_task:

                # Computing the difference between the total number of samples in the signal, and the end sample.
                n_samples_rem = sample_end - n_samples_this_task 

                # Extract the incomplete spectral information of this trial and creating an array of zeros for zero-padding
                # this incomplete trial.
                this_trial_sxx_incomplete = this_task_sxx[:,:,sample_start:n_samples_this_task].values
                this_trial_zero_padding   = np.zeros((n_chs, n_freqs, n_samples_rem))
                
                # Zero-padding this trial's incomplete spectral information with 0s to achieve a full trial length
                this_trial_sxx_padded = np.concatenate((this_trial_sxx_incomplete, this_trial_zero_padding), axis=2)
                
            # If the ending sample falls within the total number of samples in the current task.
            else:
                
                # Extracting the spectrogram for the current trial.
                this_trial_sxx = this_task_sxx[:,:,sample_start:sample_end].values
                
            # Updating the xarray with the current trial's spectrogram.
            sxx_trials_z.loc[tr] = this_trial_sxx
        
        # Updating the grasp dictionary with the current task's per-trial spectrograsm.
        grasp_sxx_dict[this_task]['sxx_trials_z'] = sxx_trials_z
    
    return grasp_sxx_dict





def spectrogram_plotting(chs_exclude, f_max, f_min, fig_height, fig_width, sxx_trial_mean, upperlimb_or_speech, v_max, v_min):
    """
    DESCRIPTION:
    The experimenter may input the channel type (upperlimb or speech) such as to display the trial-averaged spectrograms
    for that set of channels. The experimenter may also wish to input the frequency bounds within which these averaged 
    spectrograms will be viewed.
    
    INPUT VARIABLES:
    chs_exclude:         [list > strings]; The list of channels to be excluded in further analysis and whose spectral
                         information will not be shown.
    f_max:               [int]; The maximum frequency to be displayed in the averaged spectral plots. Leave as [] for
                         maximum frequency.
    f_min:               [int]; The minimum frequency to be displayed in the averaged spectral plots. Leave as [] for
                         minimum frequency.
    fig_height:          [int]; The height of the subplot figure showing the trial averaged spectrograms.
    fig_width:           [int]; The width of the subplot figure showing the trial averaged spectrograms.
    sxx_trial_mean:      [xarray (channels x frequency x time samples) > floats]; Trial-averaged standardized segmented
                         spectral power.
    upperlimb_or_speech: [string ('upperlimb'/'speech')]; Whether to plot the spectrograms of the upper-limb or speech grid.    
    v_max:               [int]; Maximum value for colorplot.
    v_min:               [int]; Minimum value for colorplot.
    """
    
    # COMPUTATION:
    
    # Uploading the electrode grid configuration.
    grid_config = params_model_training.grid_config_dict[patient_id][upperlimb_or_speech]
    grid_config = np.array(grid_config)
    
    # Flattening the grid of channels (nested list) and eliminated excluded channels from being displayed.
    channels = list(chain.from_iterable(grid_config))    
    channels = [ch for ch in channels if ch not in chs_exclude]
    
    # Extract the time and frequency dimensions from the trial-averaged spectrograms as arrays.
    f_sxx = sxx_trial_mean.frequency.values
    t_sxx = sxx_trial_mean.time.values
        
    # Extract the index of the frequency bin closest to the experimenter-input minimum frequency bound.
    if f_min:
        f_min_idx = np.abs(f_sxx - f_min).argmin()
    else:
        f_min_idx = 0

    # Extract the index of the frequency bin closest to the experimenter-input maximum frequency bound.
    if f_max:
        f_max_idx = np.abs(f_sxx - f_max).argmin()
    else:
        f_max_idx = f_sxx.shape[0]
            
    # Extracting the specific frequency range within the experimenter-determined frequency bounds from the frequency
    # array and the trial averaged spectrograms.
    f_sxx_fbounds = f_sxx[f_min_idx:f_max_idx]
    sxx_fbounds   = sxx_trial_mean[:,f_min_idx:f_max_idx,:]
    
    # PLOTTING
    
    # Initiate a subplot with the same dimensionality as the grid.
    grid_rows = grid_config.shape[0]
    grid_cols = grid_config.shape[1]
    fig, axs  = plt.subplots(grid_rows, grid_cols, figsize = (fig_width, fig_height));
    
    # Iterating across all channels of the experimenter-specified type.
    for ch in channels:

        # Extracting the channel index from the grid.
        row = np.where(grid_config == ch)[0][0]
        col = np.where(grid_config == ch)[1][0]

        # If the electrode grid has only one row.
        if grid_rows == 1:
            im = axs[col].pcolormesh(t_sxx, f_sxx_fbounds, sxx_fbounds.loc[ch], cmap = 'bwr', vmin = v_min, vmax = v_max);
            axs[col].set_title(ch)
            axs[col].axvline(x=0, color = 'k')

        # If the electrode grid has only one column. 
        elif grid_cols == 1:
            im = axs[row].pcolormesh(t_sxx, f_sxx_fbounds, sxx_fbounds.loc[ch], cmap = 'bwr', vmin = v_min, vmax = v_max);
            axs[row].set_title(ch)
            axs[row].axvline(x=0, color = 'k')

        # If there are multiple rows and columns in the grid.
        else:
            im = axs[row,col].pcolormesh(t_sxx, f_sxx_fbounds, sxx_fbounds.loc[ch], cmap = 'bwr', vmin=v_min, vmax=v_max);
            axs[row,col].set_title(ch)
            axs[row,col].axvline(x=0, color = 'k')



            
            
def spectrogram_trial_averaging(grasp_sxx_dict):
    """
    DESCRIPTION:
    Computing the trial average spectrogram.
    
    INPUT VARIABLES:
    grasp_sxx_dict:    [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_signals:   [xarray (channels x frequency bins x time samples) > floats (units: V^2/Hz)]; Spectral power of the
                       continuous voltage signals from the grasp-based task. Time dimension is in units of seconds.
        sxx_signals_z: [xarray (channels x frequency bins x time samples) > floats (units: V^2/Hz)]; Standardized Spectral
                       power of the continuous voltage signals from grasp-based task.
        sxx_states:    [xarray (1 x time samples) > strings ('state_ON'/'state_OFF'/'neutral')]; States array downsampled
                       to match time resolution of the signal spectral power. Time dimension is in units of seconds.
        sxx_trials_z:  [xarray (trials x channels x frequency bins x time samples) > floats (units: V^2/Hz)]; The spectral
                       information for each trial. Time dimension is in units of seconds.
    
    NECESSARY FUNCTIONS:
    index_advancer
    
    OUTPUT VARIABLES:
    sxx_trial_mean: [xarray (channels x frequency x time samples) > floats]; Trial-averaged standardized segmented spectral 
                    power.
    """
    # COMPUTATION:
    
    # Extracting the channels and frequency bins from task0. These are the same across all tasks.
    channels        = grasp_sxx_dict['task0']['sxx_trials_z'].channel
    freqs           = grasp_sxx_dict['task0']['sxx_trials_z'].frequency
    t_sxx_per_trial = grasp_sxx_dict['task0']['sxx_trials_z'].time
    
    # Computing the number of channels and frequency bins.
    n_chs               = channels.shape[0]
    n_freqs             = freqs.shape[0]
    n_samples_per_trial = t_sxx_per_trial.shape[0]
    
    # Initializing the total number of trials across all tasks.
    n_trials_all = 0
    
    # Computing the total number of trials.
    for this_task in grasp_sxx_dict.keys():
        
        # Extractin the number of trials during the current task.
        n_trials_this_task = grasp_sxx_dict[this_task]['sxx_trials_z'].trial.shape[0]
        
        # Updating the number of total trials.
        n_trials_all += n_trials_this_task
        
    # Initializing the xarray of spectrograms per-trial.
    sxx_all_trials = xr.DataArray(np.zeros((n_trials_all, n_chs, n_freqs, n_samples_per_trial)), 
                                  coords={'trial': np.arange(n_trials_all),'channel': channels, 'frequency': freqs, 'time': t_sxx_per_trial}, 
                                  dims=["trial", "channel", "frequency", "time"])
        
    # Initializing the trial indices.    
    tr_inds = np.zeros((2,))
    
    # Iterating across all tasks.
    for this_task in grasp_sxx_dict.keys():
                
        # Extracting the standardized spectral information for each trial.
        this_task_sxx_trials = grasp_sxx_dict[this_task]['sxx_trials_z']
        
        # Extracting the number of trials in this task.
        n_trials_this_task = this_task_sxx_trials.trial.shape[0]
        
        # Updating the trial indices.
        tr_inds = index_advancer(tr_inds, n_trials_this_task)

        # Updating the array of spectrogram information for all trials.
        sxx_all_trials.loc[tr_inds[0]:tr_inds[1]-1,:,:,:] = this_task_sxx_trials.values
            
    # Taking the mean of the spectrograms of all the trials.
    sxx_trial_mean = np.mean(sxx_all_trials, axis=0)
    
    return sxx_trial_mean





def standardize_to_calibration(calib_sxx_dict, grasp_sxx_dict):
    """
    DESCRIPTION:
    For each channel at each frequency, standardizing the spectral power to the statistics of the calibration period. 
    
    INPUT VARIABLES:
    calib_sxx_dict:  [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_signals: [xarray (channels x frequency bins x time samples) > floats (units: V^2/Hz)]; Spectral power of the
                     continuous voltage signals from calibration. Time dimension is in units of seconds.
        sxx_states:  [xarray (1 x time samples) > ints (0 or 1)]; States array downsampled to match time resolution of 
                     the signal spectral power. Time dimension is in units of seconds.
    grasp_sxx_dict:  [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_signals: [xarray (channels x frequency bins x time samples) > floats (units: V^2/Hz)]; Spectral power of the
                     continuous voltage signals from the grasp-based task. Time dimension is in units of seconds.
        sxx_states:  [xarray (1 x time samples) > strings ('state_ON'/'state_OFF'/'neutral')]; States array downsampled
                     to match time resolution of the signal spectral power. Time dimension is in units of seconds.
        
    OUTPUT VARIABLES:
    grasp_sxx_dict:    [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        sxx_signals:   Same as input.
        sxx_signals_z: [xarray (channels x frequency bins x time samples) > floats (units: V^2/Hz)]; Standardized Spectral
                       power of the continuous voltage signals from grasp-based task.
        sxx_states:    Same as input.
    """

    # COMPUTATION:

    # Iterating across all tasks in the data dictionary.
    for this_task in grasp_sxx_dict.keys():
        
        # Printing out which grasp-based task is in the pipeline.
        print('STANDARDIZING GRASP TASK: ', this_task)
        
        # Extracting the spectrograms from the grasp-based task and calibration task dictionary.
        sxx_signals_calib = calib_sxx_dict[this_task]['sxx_signals']
        sxx_signals_grasp = grasp_sxx_dict[this_task]['sxx_signals']

        # Extracting the channels, frequencies, and time coordinates.
        chs   = sxx_signals_grasp.channel
        freqs = sxx_signals_grasp.frequency
        times = sxx_signals_grasp.time
        
        # Counting the number of coordinates in each dimension.
        n_chs   = chs.shape[0]
        n_freqs = freqs.shape[0]
        n_times = times.shape[0]

        # Initializing the standardized spectrograms.
        sxx_signals_z = xr.DataArray(np.zeros((n_chs, n_freqs, n_times)), 
                                     coords={'channel': chs, 'frequency': freqs, 'time': times}, 
                                     dims=["channel", "frequency", "time"])

        # Computing the mean and standard deviation of the calibration signals across time. Results in statistics for 
        # each channel and frequency pair.
        mean_calibration  = np.mean(sxx_signals_calib, axis = 2)
        stdev_calibration = np.std(sxx_signals_calib, axis = 2)
                
        # Expanding the dimensions of the calibration mean and standard deviations.
        mean_calibration  = np.expand_dims(mean_calibration, 2)
        stdev_calibration = np.expand_dims(stdev_calibration, 2)

        # Tiling the same number of samples for the calibration mean and standard deviation arrays.
        mean_calibration  = np.tile(mean_calibration, (1, n_times))
        stdev_calibration = np.tile(stdev_calibration, (1, n_times))
        
        # Computing the standardized power for each frequency.
        sxx_signals_z = np.divide(np.subtract(sxx_signals_grasp, mean_calibration),stdev_calibration)

        # Assigning the standardized spectrogram back into the spectrogram dictionary.
        grasp_sxx_dict[this_task]['sxx_signals_z'] = sxx_signals_z
                
    return grasp_sxx_dict





def string_state_maker(grasp_cont_dict, state_int2str):
    """
    DESCRIPTION:
    Converting the state values from integers to strings, as the string values are more descriptive.
    
    INPUT VARIABLES:
    grasp_cont_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        signals:     [xarray (channels x time samples (units: s) > floats (units: microvolts)]; Array of continuous
                     voltage signals from grasp-based tasks. Time dimension is in units of seconds.
        states:      [xarray (1 x time samples) > ints (0 or 1)]; Array of states at each time sample for the grasp-based
                     tasks. Time dimension is in units of seconds.
    state_int2str:   [dictionary (Key: int (state on/off value); Value: string (state_ON/state_OFF, respectively)]; Mapping
                     from the numerical values to the string values.
    
    OUTPUT VARIABLES:
    grasp_cont_dict: [dictionary (Key: string (task ID); Value: dictionary (Key/Value pairs below)];
        signals: Same as input.
        states:  [xarray (1 x time samples) > strings ('state_ON'/'state_OFF')]; Array of states at each time sample for the
                 grasp-based tasks. Time dimension is in units of seconds.
    """
    
    # COMPUTATION:
    
    # Iterating across each task in the dictionary.
    for this_task in grasp_cont_dict.keys():
    
        # Extracting the state array from the current task.
        this_task_state_array = grasp_cont_dict[this_task]['states']
                
        # Extracting the states time array and number of samples.
        t_seconds = this_task_state_array.time
        n_samples = t_seconds.shape[0]
        
        # Initializing the array of state strings.
        this_task_state_strings = xr.DataArray(np.asarray([None]*n_samples),  
                                               coords={'time': t_seconds}, 
                                               dims=["time"])
        
        # Iterating across all pairs of state values and corresponding string names.
        for state_val, state_string in state_int2str.items():
            
            # Extracting the state string of the current state value.
            state_string = state_int2str[state_val]
            
            # Extracting the indices of the current state value.
            these_inds_state_val = np.where(this_task_state_array == state_val)
        
            # Updating the array of state strings.
            this_task_state_strings[these_inds_state_val] = state_string
    
        # Updating the state arrays with the string version.
        grasp_cont_dict[this_task]['states'] = this_task_state_strings
    
    return grasp_cont_dict





# def time_history_sample_adjustment_features(features_dict, t_history):
#     """
#     DESCRIPTION:
#     Adjusting the time dimension of the features. Due to the time history, the time-shifted rows of the features arrays are
#     zero padded. As such, all columns with leading zeros should be removed. If there are N time shfited columns, this means
#     that N-1 columns should be removed. For example, consider N = 3:
    
#     Features array:
    
#     historical time shifts
#      n=2 shifts      [[0.000, 0.000, 0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101],
#      n=1 shift        [0.000, 0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521],
#      n=0 shifts       [0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521, 0.882]]
#                         t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7    t=8    t=9     (time samples)   
                        
#     Labels:             l0     l1     l2     l3     l4     l5     l6     l7     l8     l9
    
#     After curtailing, the features and labels arrays would look like:
    
#     historical time shifts
#      n=2 shifts      [[0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101],
#      n=1 shift        [0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521],
#      n=0 shifts       [0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521, 0.882]]
#                         t=2    t=3    t=4    t=5    t=6    t=7    t=8    t=9     (time samples)   
                        
#     Labels:             l2     l3     l4     l5     l6     l7     l8     l9
    
#     where time points and labels will then be re-adjusted.
    
    
#     INPUT VARIABLES:
#     features_dict: [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats (units: V^2/Hz))]
#                    Array of historical time features.
#     t_history:     [float (unit: s)]; Amount of feature time history.
                   
#     OUTPUT VARIABLES:
#     features_dict: [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats (units: V^2/Hz))]
#                    Array of historical time features. Number of time samples corresponding to time history are curtailed at the beginning
#                    of the time array.
#     """
    
#     # COMPUTATION:
    
#     # Iterating across each task.
#     for this_task in features_dict.keys():
        
#         # Extracting the features and labels for the current task.
#         this_task_features = features_dict[this_task]
        
#         # Computing the number of samples corresponding to the features.
#         n_history = int(t_history/sxx_shift)
        
#         # Extracting the features and labels corresponding to only the time samples after the first n_history samples.
#         this_task_features_curt = this_task_features[:,:,n_history:]

#         # Updating the features and labels dictionary.
#         features_dict[this_task] = this_task_features_curt
    
#     return features_dict 





def time_history_sample_adjustment(features_dict, labels_dict, t_history):
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
    features_dict: [dictionary (Key: string (task ID); Value: xarray (time history, features, time) > floats (units: V^2/Hz))]
                   Array of historical time features.
    labels_dict:   [dictionary (Key: string (task ID); Value: xarray > strings ('grasp'/'rest'))]; For each time sample in each
                   task, there exists a rest or grasp label depending on the experimenter-specified onset and offset of
                   modulation as well as the per-trial shift from the AW model.
    t_history:     [float (unit: s)]; Amount of feature time history.

    GLOBAL PARAMETERS: 
    sxx_shift: [int (units: ms)]; Length of time by which sliding window (sxx_window) shifts along the time domain.
    
    OUTPUT VARIABLES:
    features_dict: [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats (units: V^2/Hz))]
                   Array of historical time features. Number of time samples corresponding to time history are curtailed at the
                   beginning of the time array.
    labels_dict:   [dictionary (Key: string (task ID); Value: xarray > strings ('grasp'/'rest'))]; For each time sample in a task,
                   there exists a rest or grasp label depending on the experimenter-specified onset and offset of grasp modulation
                   as well as the per-trial shift from the AW model. Number of time samples corresponding to time history are
                   curtailed at the beginning of the time array.
    """
    
    # COMPUTATION:
    
    # Iterating across each task.
    for this_task in features_dict.keys():
        
        # Extracting the features and labels for the current task.
        this_task_features = features_dict[this_task]
        this_task_labels   = labels_dict[this_task]
        
        # Computing the number of samples corresponding to the features.
        n_history = int(t_history/sxx_shift)
        
        # Extracting the features and labels corresponding to only the time samples after the first n_history samples.
        this_task_features_curt = this_task_features[:,:,n_history:]
        this_task_labels_curt   = this_task_labels[n_history:]

        # Updating the features and labels dictionary.
        features_dict[this_task] = this_task_features_curt
        labels_dict[this_task]   = this_task_labels_curt
    
    return features_dict, labels_dict 





# def time_history_sample_adjustment_labels(labels_dict, t_history):
#     """
#     DESCRIPTION:
#     Adjusting the time dimension of the labels. Due to the time history, the time-shifted rows of the features arrays are
#     zero padded. As such, all columns with leading zeros should be removed. If there are N time shfited columns, this means 
#     that N-1 columns should be removed. For example, consider N = 3:
    
#     Features array:
    
#     historical time shifts
#      n=2 shifts      [[0.000, 0.000, 0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101],
#      n=1 shift        [0.000, 0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521],
#      n=0 shifts       [0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521, 0.882]]
#                         t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7    t=8    t=9     (time samples)   
                        
#     Labels:             l0     l1     l2     l3     l4     l5     l6     l7     l8     l9
    
#     After curtailing, the features and labels arrays would look like:
    
#     historical time shifts
#      n=2 shifts      [[0.234, 0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101],
#      n=1 shift        [0.523. 0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521],
#      n=0 shifts       [0.435, 0.982, 0.175, 0.759, 0.341, 0.101, 0.521, 0.882]]
#                         t=2    t=3    t=4    t=5    t=6    t=7    t=8    t=9     (time samples)   
                        
#     Labels:             l2     l3     l4     l5     l6     l7     l8     l9
    
#     where time points and labels will then be re-adjusted.
    
    
#     INPUT VARIABLES:
#     labels_dict: [dictionary (Key: string (task ID); Value: xarray > strings ('grasp'/'rest'))]; For each time sample in a task, there 
#                  exists a rest or grasp label depending on the experimenter-determined onset and offset of modulation as well as the 
#                  per-trial shift from the AW model.
#     t_history:   [float (unit: s)]; Amount of feature time history.
                   
#     OUTPUT VARIABLES:
#     labels_dict: [dictionary (Key: string (task ID); Value: xarray > strings ('grasp'/'rest'))]; For each time sample in a task, there 
#                  exists a rest or grasp label depending on the experimenter-determined onset and offset of modulation as well as the 
#                  per-trial shift from the AW model. Number of time samples corresponding to time history are curtailed at the beginning
#                  of the time array.
#     """
    
#     # COMPUTATION:
    
#     # Iterating across each task.
#     for this_task in labels_dict.keys():
        
#         # Extracting the labels for the current task.
#         this_task_labels = labels_dict[this_task]
        
#         # Computing the number of samples corresponding to the features.
#         n_history = int(t_history/sxx_shift)
        
#         # Extracting the labels corresponding to only the time samples after the first n_history samples.
#         this_task_labels_curt = this_task_labels[n_history:]

#         # Updating the labels dictionary.
#         labels_dict[this_task] = this_task_labels_curt
    
#     return labels_dict 





def training_final_model(features_dict, labels_dict):
    """
    DESCRIPTION:
    Concatenating the data and labels from all tasks and training the final model on these concatenated arrays.
    
    INPUT VARIABLES:
    features_dict: [dictionary (Key: string (task ID); Value: xarray (time history x features x time) > floats (units: V^2/Hz))]
                   Array of historical time features. Time samples reduced such that there are an equal number of features per
                   class.
    labels_dict:   [dictionary (Key: string (task ID); Value: xarray > strings ('grasp'/'rest'))]; For each time sample in each
                   task, there exists a rest or grasp label depending on the experimenter-specified onset and offset of
                   modulation as well as the per-trial shift from the AW model. Time samples reduced such that there are an equal
                   number of features per class.
                   
    NECESSARY FUNCTIONS:
    computing_eigenvectors
    concatenating_all_data_and_labels
    mean_centering
    mean_compute
    model_training_lstm
    pc_transform
    rearranging_features
    
    OUTPUT VARIABLES:
    eigenvectors_final: [array (features x pc features) > floats]; Array in which columns consist of eigenvectors which explain
                        the variance of the data in descending order. 
    final_model:        [classification model]; Model trained with data from all tasks.
    training_data_mean: [xarray (history x features) > floats (units: V^2/Hz)]; Mean power of each feature of only the 0th time 
                        shift.  This array is repeated for each historical time point.
    """
    
    # COMPUTATION:
    
    # Concatenating the data and labels from all tasks for training.
    training_data, training_labels = concatenating_all_data_and_labels(features_dict, labels_dict)
    
    # Computing the mean of the training data.
    training_data_mean = mean_compute(training_data)
    
    # Mean-centering the training data.
    training_data = mean_centering(training_data, training_data_mean)
    
    # Extracting only the training data corresponding to the 0th historical time shift.
    training_data_history0 = np.asarray(training_data.loc[0,:,:])

    # Computing the eigenvectors, using only the historical features corresponding to the 0th shift.
    eigenvectors = computing_eigenvectors(training_data_history0)
    
    # Computing the reduced-dimension PC training data.
    training_data = pc_transform(training_data, eigenvectors)

    # Rearranging features to fit the appropriate model type.
    training_data = rearranging_features(training_data)
    
    # If the model type is an LSTM
    if model_type == 'LSTM':

        # Creating a LSTM model for the current fold of training data. The third and fourth inputs in this
        # function take the spot of validation data and labels, but are meaningless here. Similarly, the 
        # validation accuracy should be ignored.
        final_model = model_training_lstm(training_data, training_labels, training_data, training_labels)
    
    return eigenvectors, final_model, training_data_mean





def training_fold_models(train_data_folds, train_labels_folds, valid_data_folds, valid_labels_folds):
    """
    DESCRIPTION:
    This is only to be used to create a confusion matrix across all validation data folds using models trained on the 
    correpsonding training data folds. Used to assess classification performance before training a final model on all
    the data.

    INPUT VARIABLES:
    train_data_folds:   [dict (key: string (fold ID); Value: xarray (dimensions vary based on model type) > floats (units: V^2/Hz))];
                        Data across all training tasks per fold. Equal number of samples per class. PC features. Rearranged 
                        according to the type of model that will be trained.
    train_labels_folds: [dict (key: string (fold ID); Value: xarray (1 x time samples) > strings ('grasp'/'rest'))]; Labels across
                        all training tasks per fold. Equal number of labels per class.
    valid_data_folds:   [dict (key: string (fold ID); Value: xarray (dimensions vary based on model type) > floats (units: V^2/Hz))];
                        Data across all validation tasks per fold. Equal number of samples per class. PC features. Rearranged 
                        according to the type of model that will be trained.
    valid_labels_folds: [dict (key: string (fold ID); Value: xarray (1 x time samples) > strings ('grasp'/'rest'))]; Labels across
                        all validation tasks per fold. Equal number of labels per class.
                        
    GLOBAL PARAMETERS:
    model_type: [string ('SVM','LSTM')]; The model type that will be used to fit the data.
                             
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





# def training_validation_split(features_dict, labels_dict):
#     """
#     DESCRIPTION:
#     Splitting the data into training and validation blocks for building the fold-models. The validation folds will be tested
#     on the models built on the corresponding training folds to confirm that the decoder is generalizable to unseen data. 
    
#     INPUT VARIABLES:
#     features_dict: [dictionary (key: string (task ID): Value: xarray (dimensions vary based on model type) > floats (units: V^2/Hz))];
#                    Rearranged features. 
#     labels_dict:   [dictionary (key: string (task ID): Value: xarray (1 x time samples) > strings ('grasp'/'rest'))]; For each sample,
#                    corresponding label.
               
#     OUTPUT VARIABLES:
#     training_data_folds:     [dict (key: string (fold ID); Value: xarray (dimensions vary based on model type) > floats (units: V^2/Hz))];
#                              For each training fold, feature xarrays are concatenated in the sample dimension.
#     training_labels_folds:   [dict (key: string (fold ID); Value: xarray (1 x time samples) > strings ('grasp'/'rest'))]; For each training
#                              fold, label xarrays are concatenated in the sample dimension.
#     validation_data_folds:   [dict (key: string (fold ID); Value: xarray (dimensions vary based on model type) > floats (units: V^2/Hz))];
#                              For each validation fold, feature xarrays are concatenated in the sample dimension.
#     validation_labels_folds: [dict (key: string (fold ID); Value: xarray (1 x time samples) > strings ('grasp'/'rest'))]; For each validation
#                              fold, label xarrays are concatenated in the sample dimension.
#     """
    
#     # COMPUTATION:

#     # Initializing dictionaries of training and validation data and labels.
#     training_data_folds     = {}
#     training_labels_folds   = {}
#     validation_data_folds   = {}
#     validation_labels_folds = {}
    
#     # Iterating across all tasks.
#     for n, this_task in enumerate(features_dict.keys()):
        
#         # Initialize the training task flag.
#         training_task0_flag = True
        
#         print('\nTESTING TASK: ', this_task)

#         # Extracting the training data and labels for the current fold.                
#         for this_task_training in features_dict.keys():
#             if this_task_training != this_task:
                
#                 print('TRAINING TASK: ', this_task_training)
                                            
#                 # Extracting the training data and labels of the current task.
#                 this_task_data   = features_dict[this_task_training]
#                 this_task_labels = labels_dict[this_task_training]
            
#                 # If the training task flag is True, intiailize the training data and labels xarrays. If not, concatenate
#                 # them with data and labels from another task.
#                 if training_task0_flag:
#                     these_training_data   = this_task_data
#                     these_training_labels = this_task_labels
                    
#                     # Setting the flag to False to never enter this IF statement again.
#                     training_task0_flag = False
                    
#                 else:                           
#                     these_training_data   = xr.concat([these_training_data, this_task_data], dim="sample")
#                     these_training_labels = xr.concat([these_training_labels, this_task_labels], dim='sample')

        
#         # Reassigning the sample coordinates to the training data and labels xarrays.
#         these_training_data   = these_training_data.assign_coords(sample=np.arange(these_training_data.sample.shape[0]))
#         these_training_labels = these_training_labels.assign_coords(sample=np.arange(these_training_labels.sample.shape[0]))
        
#         # Extracting the validation data and labels for the current fold.
#         these_validation_data   = features_dict[this_task]
#         these_validation_labels = labels_dict[this_task]
        
#         # Creating a fold ID.
#         fold_id = 'fold'+str(n)
        
#         # Updating the training and validation data and labels dictionaries with the appropriate training and validation 
#         # information.
#         training_data_folds[fold_id]     = these_training_data
#         training_labels_folds[fold_id]   = these_training_labels
#         validation_data_folds[fold_id]   = these_validation_data
#         validation_labels_folds[fold_id] = these_validation_labels
            
#     return training_data_folds, training_labels_folds, validation_data_folds, validation_labels_folds





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





# def unique_value_index_finder(my_vector):
#     """
#     Description: This function finds useful indexing information about multi-step step functions.

#     INPUT VARIABLES:
#     my_vector: [list]; List of a step signal with multiple steps of various heights
    
#     OUTPUT VARIABLES:
#     unique_vals:     [list > (strings/ints)] Individual values (at different "heights") of all the steps in the vector. For example, the unique_vals of 
#                      my_ex_vector = ['a','a','a','a','b','b','b','b','a','a','a','a','c','c','c','c','a','a','a','a'] would be [a,b,c]
#     n_steps_per_val: [dict > key: step names (ints/strings); values: number of occurrences (ints)] The number of steps for each unique value. For example, 
#                      n_steps_per_val of my_ex_vector would be {a: 3, b: 1, c: 1} because the "a" step occurs 3 times, whereas "b" and "c" steps occurs only once.
#     unique_val_inds: [dict > key: step names (ints/strings); values: list > array indices (ints)] All the indices within the vector of where a specific step occurs. 
#                      For example, the unique_val_inds of my_ex_vector would be: {a: [0,1,2,3,8,9,10,11,16,17,18,19], b: [4,5,6,7], c: [12,13,14,15]}.
#     start_end_inds:  [dict > key: step names (ints/strings); values: list > list > array indices (ints)]; The indices within the vector where the specific steps
#                      start and stop. For example the start_stop_inds of my_ex_vector would be: {a: [[0,3],[8,11],[16,19]], b: [[4,7]], c: [[12,15]]}
#     """
        
#     # COMPUTATION:
    
#     # Find the unique sorted elements of my_vector.
#     unique_vals = np.unique(my_vector)
    
#     # Initiate the list of indices.
#     unique_val_inds = {}
#     start_end_inds  = {}
#     n_steps_per_val = {}
    
#     # Iterate across each value in unique_vals.
#     for this_val in unique_vals:
                
#         this_val_inds = [i for i, x in enumerate(my_vector) if x == this_val]
#         inds          = []
#         these_inds    = []
        
#         for n in range(len(this_val_inds)):
                        
#             if n == 0:
#                 these_inds.extend([this_val_inds[n]])
#             if n > 0:
#                 if this_val_inds[n] - this_val_inds[(n - 1)] > 1:
#                     these_inds.extend([this_val_inds[(n - 1)]])
#                     inds.append(these_inds)
#                     these_inds = [this_val_inds[n]]
#             if n == len(this_val_inds) - 1:
#                 these_inds.extend([this_val_inds[n]])
#                 inds.append(these_inds)
    
#         # Updating the dictionaries.
#         n_steps_per_val[this_val] = len(inds)
#         unique_val_inds[this_val] = this_val_inds
#         start_end_inds[this_val]  = inds
        
#     # Converting the unique values to a list.
#     unique_vals = list(unique_vals)
    
#     return unique_vals, n_steps_per_val, unique_val_inds, start_end_inds





# def upload_movememt_onsets_offsets(block_id, date, dir_base, patient_id):
#     """
#     DESCRIPTION:
#     The dictionary with the onset and offset times for each movement (here, only grasp) is uploaded.
    
#     INPUT VARIABLES:
#     block_id:   [String]; Block ID of the task that was run. Should be format 'Block#'.
#     date:       [string (YYYY_MM_DD)]; Date on which the current block was run.
#     dir_base:   [string]; Base directory where all information is stored.
#     patient_id: [string]; Patient PYyyNnn ID or CCXX ID.
    
#     OUTPUT VARIABLES:
#     dict_onsets_offsets: [dict (key: movement, value: onset/offset times)]; A dictionary which contains dictionaries of movement onset/offset for multiple
#                          movements.
#     """
    
#     # COMPUTATION:
    
#     # Creating the pathway for the .txt file where the movement onsets and offsets are stored.
#     this_directory           = dir_base + patient_id + '/Speller/MovementOnsetsAndOffsets/' + date + '/' 
#     this_filename            = 'dict_OnsetOffset_' + block_id
#     path_onsets_offsets_dict = this_directory + this_filename
    
#     # Read in the dictionary from the pathway.
#     with open(path_onsets_offsets_dict, "rb") as fp:   
#         dict_onsets_offsets = pickle.load(fp)

#     # PRINTING
#     # pprint(dict_onsets_offsets)
    
#     return dict_onsets_offsets





def visualizing_affinewarp_adjustment(powerband_id, ptr_all_trials, ptr_aligned_all_trials, t_corr_end, t_corr_start, view_channel):
    """
    DESCRIPTION:
    After affine-warping the power trial rasters, the inter-trial correlations are compared to the power-trial rasters from
    before alignment.
    
    INPUT VARIABLES:
    powerband_id:           [string]; The powerband whose power trial rasters the experimenter wishes to compare.
    ptr_all_trials:         [xarray (trials x channels x powerbands x time samples) > floats (units: V^2/Hz)]; The spectral
                            information for each trial (across all tasks). Time dimension is in units of seconds.
    ptr_aligned_all_trials: [xarray (trials x channels x powerbands x time samples) > floats (units: V^2/Hz)]; The spectral
                            information for each aligned trial (across all tasks). Time dimension is in units of seconds.
    t_corr_end:             [float]; The ending time bound of the trials between which to take the pair-wise correlations.
    t_corr_start:           [float]; The starting time bound of the trials between which to take the pair-wise correlations.
    view_channel:           [string]; The experimenter-specified channel whose aligned and unaligned trials will be 
                            vizualized.
    """
    
    # COMPUTATION:
    
    # Extracting the unaligned and aligned power trial rasters for the experimenter-specified power band.
    ptr_unaligned = ptr_all_trials.loc[:,:,powerband_id,:]
    ptr_aligned   = ptr_aligned_all_trials.loc[:,:,powerband_id,:]

    # Extracting the time array for the trial segment. The time array will be the same for both aligned and unaligned power
    # trial rasters.
    t_segment_ptr = ptr_unaligned.time

    # Extracting the power trial rasters for the experimenter-specified visualization channel.
    this_ch_ptr_unaligned = ptr_unaligned.loc[:,view_channel,:]
    this_ch_ptr_aligned   = ptr_aligned.loc[:,view_channel,:]

    # Computing the total number of trials in this class.
    n_trials = this_ch_ptr_unaligned.trial.shape[0]

    # Computing the time indices between which the signal correlations will happen.
    t_corr_inds_bool = np.logical_and(t_segment_ptr > t_corr_start, t_segment_ptr < t_corr_end)

    # Computing number of total pair-wise correlations (excluding auto-correlations of 1)
    n_total_corr = n_trials**2
    n_diag_corr  = n_trials
    n_pw_corr    = (n_total_corr - n_diag_corr)/2

    # Computing the correlation matrix (and mean) of all the trials pre-alignment by using a Pandas dataframe.
    df_ptr_unaligned         = pd.DataFrame(this_ch_ptr_unaligned[:,t_corr_inds_bool].transpose())
    corr_ptr_unaligned       = df_ptr_unaligned.corr()
    corr_ptr_unaligned_lower = np.tril(corr_ptr_unaligned)
    sum_total_corr_unaligned = corr_ptr_unaligned_lower.sum()
    sum_diag_corr_unaligned  = np.sum(np.diagonal(corr_ptr_unaligned))
    mean_corr_ptr_unaligned  = round((sum_total_corr_unaligned - sum_diag_corr_unaligned)/n_pw_corr, 3)

    # Computing the correlation matrix (and mean) of all the trials post-alignment by using a Pandas dataframe.
    df_ptr_aligned         = pd.DataFrame(this_ch_ptr_aligned[:,t_corr_inds_bool].transpose())
    corr_ptr_aligned       = df_ptr_aligned.corr()
    corr_ptr_aligned_lower = np.tril(corr_ptr_aligned)
    sum_total_corr_aligned = corr_ptr_aligned_lower.sum()
    sum_diag_corr_aligned  = np.sum(np.diagonal(corr_ptr_aligned))
    mean_corr_ptr_aligned  = round((sum_total_corr_aligned - sum_diag_corr_aligned)/n_pw_corr, 3)

    # Computing the difference matrix (and mean) from each pairwise correlation.
    corr_diff       = corr_ptr_aligned - corr_ptr_unaligned
    corr_diff_lower = np.tril(corr_diff)
    mean_corr_diff  = round(mean_corr_ptr_aligned - mean_corr_ptr_unaligned, 3)

    # Computing the average difference in pair-wise correlations for each trial.
    corr_diff_trial      = np.sum(corr_diff, axis = 0)
    corr_diff_mean_trial = corr_diff_trial/(n_trials - 1)

    # PLOTTING
    fig, ax = plt.subplots(1,2, figsize = (20, 5), gridspec_kw={'width_ratios': [2.5, 1]})
    ax[0].plot(t_segment_ptr, this_ch_ptr_unaligned.transpose())
    ax[0].set_title('Channel ' + view_channel + ' Cue-Aligned')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('HG Power')
    ax[0].grid()
    ax[1] = sns.heatmap(corr_ptr_unaligned_lower, vmin=-1, vmax=1, cmap = 'coolwarm')
    ax[1].set_title('Channel ' + view_channel + ' Mean Correlation: ' + str(mean_corr_ptr_unaligned))
    ax[1].set_xlabel('Trials');
    ax[1].set_ylabel('Trials');

    fig, ax = plt.subplots(1,2, figsize = (20, 5), gridspec_kw={'width_ratios': [2.5, 1]})
    ax[0].plot(t_segment_ptr,  this_ch_ptr_aligned.transpose())
    ax[0].set_title('Channel ' + view_channel + ' Re-Aligned')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('HG Power')
    ax[0].grid()
    ax[1] = sns.heatmap(corr_ptr_aligned_lower, vmin=-1, vmax=1, cmap = 'coolwarm')
    ax[1].set_title('Channel ' + view_channel + ' Mean Correlation: ' + str(mean_corr_ptr_aligned))
    ax[1].set_xlabel('Trials');
    ax[1].set_ylabel('Trials');    

    fig, ax = plt.subplots(1,2, figsize = (20, 5), gridspec_kw={'width_ratios': [1, 2.25]})
    ax[0] = sns.heatmap(corr_diff_lower, vmin=-1, vmax=1, cmap = 'vlag', ax = ax[0])
    ax[0].set_title('Channel ' + view_channel + ' Mean Correlation Difference: ' + str(mean_corr_diff))
    ax[0].set_xlabel('Trials');
    ax[0].set_ylabel('Trials');
    ax[1].plot(corr_diff_mean_trial);
    ax[1].axhline(y = 0, xmin = 0, xmax = n_trials, color = 'k')
    ax[1].set_title('Channel ' + view_channel + ' Mean Correlation Pairwise Difference Per Trial')
    ax[1].set_xlabel('Trials')
    ax[1].set_ylabel('Mean PW-Correlation Difference')
    ax[1].set_ylim([-1,1])
    ax[1].set_yticks(np.round(np.arange(-1,1.2,0.2),2))
    ax[1].grid()
     
        
        
           
        
def visualizing_warping_channels_traces(chs_alignment, powerband_id, ptr_aligned_all_trials):
    """
    DESCRIPTION:
    The average aligned high gamma activity (across trials) is plotted for each of the channels used for affine warping. 

    INPUT VARIABLES:
    chs_alignment:          [list > strings]; The list of channels which will be used for affine warp. Leave as [] if
                            all channels will be used.
    powerband_id:           [string]; The powerband whose average trials traces will be plotted.
    ptr_aligned_all_trials: [xarray (trials x channels x powerbands x time samples) > floats (units: V^2/Hz)]; The spectral
                            information for each aligned trial (across all tasks). Time dimension is in units of seconds.
    """
    
    # COMPUTATION:
    
    # Extracting the power of only the channels and powerband used for alignment..
    ptr_channels_powerband = ptr_aligned_all_trials.loc[:, chs_alignment, powerband_id,:]

    # Extracting the per-trial time array.
    t_segment_ptr = ptr_channels_powerband.time

    # Taking the mean and standard deviation across trials.
    this_stimulus_trace_means  = ptr_channels_powerband.mean(dim='trial')
    this_stimulus_trace_stdevs = ptr_channels_powerband.std(dim='trial')

    # PLOTTING
    fig, ax = plt.subplots(2,1, figsize = (20, 10))
    ax[0].plot(t_segment_ptr, this_stimulus_trace_means.transpose());
    ax[0].set_title('Average power traces for AW channels')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Mean Bandpower (V^2/Hz)')
    ax[0].legend(chs_alignment)
    ax[0].grid()
    ax[1].plot(t_segment_ptr, this_stimulus_trace_stdevs.transpose());
    ax[1].set_title('Standard deviation of power traces for AW channels')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Stdev Bandpowr (V^2/Hz)')
    ax[1].legend(chs_alignment)
    ax[1].grid()
    
    # fig.savefig('TrialAveraged_Traces', bbox_inches='tight')
    # fig.savefig('TrialAveraged_Traces.svg', format = 'svg', bbox_inches='tight', dpi = 2000)
