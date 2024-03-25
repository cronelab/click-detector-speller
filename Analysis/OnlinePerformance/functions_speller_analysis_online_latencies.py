
# IMPORTING LIBRARIES
import numpy as np
import pickle
import shutil

def save_script_backup():
    
    """
    Automatically saving this entire script immediately when it's called.
    """
    
    # Creating the original and target file directories.
    original = r'/home/dan/Projects/PseudoOnlineTests_for_RTCoG/Scripts/Recent/SpellerAnalysis/functions_speller_analysis_online_latencies.py'
    target   = r'/mnt/shared/danprocessing/BACKUP/Projects/PseudoOnlineTests_for_RTCoG/Scripts/SpellerAnalysis/functions_speller_analysis_online_latencies.py'

    # Saving.
    shutil.copyfile(original, target)
    
# Immediately saving script.   
save_script_backup()





def extracting_latencies(date_block_dict, dir_base, movement):
    """
    DESCRIPTION:
    Extracting all the latencies from the experimenter-input blocks into one array.
    
    INPUT VARIABLES:
    date_block_dict: [dictionary (keys: strings (dates); values: list > ints (blocks)]; Dictionary of dates and blocks corresponding to each date. 
    dir_base:        [string]; Base directory where all information for this subject and task (Speller) is found.
    movement:        [string]; The movement from which the onsets and offsets will be extracted.

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

            # Creating the pathway for the .txt file where the starting and ending times of the current block are stored.
            this_directory       = dir_base + 'ClickLatencies/' + date + '/'
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

    # Computing the mean and standard deviation of the latencies.
    latencies_mean  = round(np.mean(latencies_arr), 3)
    latencies_stdev = round(np.std(latencies_arr), 3)

    # Printing the mean and standard deviation.
    print('Latency Mean (s): \t', latencies_mean)
    print('Latency Stdev (s): \t', latencies_stdev)
