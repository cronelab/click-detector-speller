
# IMPORTING LIBRARIES
import numpy as np
import pickle

import matplotlib.pyplot as plt
from scipy.stats import bootstrap



def extracting_latencies(date_block_dict, dir_base, folder, movement):
    """
    DESCRIPTION:
    Extracting all the latencies from the experimenter-input blocks into one array.
    
    INPUT VARIABLES:
    date_block_dict:     [dictionary (keys: strings (dates); values: list > ints (blocks)]; Dictionary of dates and 
                         blocks corresponding to each date. 
    dir_click_latencies: [string]; Base directory where the click latencies are stored.
    folder_bci2k_or_ui:  [string (BCI2000/UI)]; The specific folder from where to extract BCI2000 or UI latencies.
    movement:            [string]; The movement from which the onsets and offsets will be extracted.

    OUTPUT VARIABLES:
    latencies_arr: [array > floats (units: s)]; Array of latencies from all experimenter-input blocks.
    """

    # COMPUTATION:
    
    # Initializing the list of latencies.
    list_of_latencies = []

    # Iterating across all dates. 
    for date in date_block_dict.keys():

        # Extracting the block list from dictionary.
        block_list = date_block_dict[date]

        # Iterating across all blocks from the current block list.
        for block in block_list:

            # Creating the pathway for the .txt file where the starting and ending times of the current block are
            # stored.
            this_directory       = dir_base + date + '/' + folder + '/'
            this_filename        = date + '_' + 'click_latencies'
            path_click_latencies = this_directory + this_filename

            # Read in the dictionary from the pathway.
            with open(path_click_latencies, "rb") as fp:   
                dict_click_latencies = pickle.load(fp)

            # Extracting the click latencies from the dictionary.
            t_click_latencies = dict_click_latencies[movement]['block'+str(block)]

            # Adding the array of click latencies to the list.
            list_of_latencies.append(t_click_latencies)

    # Concatenating the arrays from the list into one array.
    latencies_arr = np.hstack(list_of_latencies)
    
    return latencies_arr





def latency_stats(latencies_arr):
    """
    DESCRIPTION:
    Computing the mean and standard deviation of the latencies.

    INPUT VARIABLES:
    latencies_arr: [array > floats (units: s)]; Array of latencies from all experimenter-input blocks.

    OUTPUT VARIABLES:
    latencies_mean:  [float (units: s)]; Mean of latencies.
    latencies_stdev: [float (units: s)]; Standard deviation of latencies.
    """

    # COMPUTATION:

    # Computing the mean, median, and standard deviation of the latencies.
    latencies_mean   = round(np.mean(latencies_arr), 3)
    latencies_median = round(np.median(latencies_arr), 3)
    latencies_stdev  = round(np.std(latencies_arr), 3)
    
    # Computing the confidence interval.
    ci_95 = bootstrap((latencies_arr,), np.mean, confidence_level=0.95, n_resamples=10000, method='BCa').confidence_interval

    # Printing the mean and standard deviation.
    print('Latency Mean (s): \t', latencies_mean)
    print('Latency Median (s): \t', latencies_median)
    print('Latency Stdev (s): \t', latencies_stdev)
    print('95% CI: ', ci_95)

    
    plt.hist(latencies_arr, bins=50)