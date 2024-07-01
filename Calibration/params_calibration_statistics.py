





######################## PATIENT AND DATA INFORMATION #########################
calib_state_val = 2
car             = True
date            = '2023_12_08'
dir_saving      = 'path/to/saving/calibration/statistics'
exper_name      = 'Go_Grasp_RH_Long_Block1'
file_extension  = 'hdf5'
patient_id      = 'CC01'
sampling_rate   = 1000 # 1000 or 2000
state_name      = 'StimulusCode'
sxx_shift       = 100
sxx_window      = 256

"""
PARAMETERS:
calib_state_val: [int]; The state value from where to extract the appropriate calibration data.
car:             [bool (True/False)] Whether or not CAR filtering will be performed.
date:            [string]; The date from which the calibration statistics will be computed.
dir_saving:      [string]; Directory where calibration statistics are saved.
exper_name:      [string]; Name of the experiment from which the calibration statistics will be computed.
file_extension:  [string (hdf5/mat)]; The data file extension of the data.
patient_id:      [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
sampling_rate:   [int (samples/s)]; Sampling rate at which the data was recorded.
state_name:      [string]; BCI2000 state name that contains relevant state information.
sxx_shift:       [int (units: ms)]; Length of time by which sliding window (sxx_window) shifts along the time domain.
sxx_window:      [int (units: ms)]; Time length of the window that computes the frequency power.
"""










# Patient ID:
# patient_id = 'CC01'

# File Extension:
# file_extension: [string (hdf5/mat)]; The data file extension to be used.
# file_extension = 'hdf5'

# Date:
# date = '2023_12_08'

# # Task
# task = 'Go_Grasp_RH_Long_Block1' 


# # Calibration Stimulus
# calibration_stimulus = 2

# SPECTROGRAM PARAMETERS
# shift         = 100
# window_length = 256
# shift:         [int (units: ms)]; Length of time by which sliding window shifts along the time domain.
# window_length: [int (units: ms)]; Time length of the window that computes the frequency power.



###################### INCLUDED CHANNELS ####################

# All channels
chs_include = ['chan'+str(n+1) for n in range(128)]

# # Only top grid.
# chs_include = ['chan'+str(n+1) for n in range(65,128)]

# Only top grid, and excluding channels 114, 121, 122.
# chs_include = ['chan'+str(n+1) for n in range(65,128)]; chs_include.remove('chan114'); chs_include.remove('chan121'); chs_include.remove('chan122'); 

############################# EXCLUDED CHANNELS #############################

# Initializing the dictionary for channels to be eliminated from future analysis. These are in addition to previously
# recorded bad channels. 
elim_channels = {}
elim_channels['CC01'] = ['ainp1','ainp2','ainp3']
# elim_channels['CC01'] = ['ainp1','ainp2','ainp3'] + ['chan' + str(n) for n in range(1,65)]

"""
PARAMETERS:
elim_channels: [dictionary (key: string (patient ID); Value: list > strings (bad channels))]; The list of bad or non-
               neural channels to be exlucded from further analysis.
"""




############################# CAR CHANNELS #############################

# Initializing the dictionary, which holds the sets of channels to be independently CAR-ed.
car_channels = {}

# CC01
car_channels['CC01'] = [['chan66', 'chan67', 'chan68', 'chan69', 'chan70', 'chan74', 'chan75', 'chan76', 'chan77',\
                         'chan78', 'chan84', 'chan85', 'chan86', 'chan91', 'chan92', 'chan93', 'chan94', 'chan99',\
                         'chan100', 'chan101', 'chan102', 'chan108', 'chan109', 'chan110', 'chan117', 'chan118',\
                         'chan125', 'chan126', 'chan71', 'chan72', 'chan79', 'chan80', 'chan87', 'chan88', 'chan95',\
                         'chan96', 'chan103', 'chan104', 'chan112', 'chan120', 'chan128', 'chan89', 'chan73', 'chan65',\
                         'chan90', 'chan83', 'chan82', 'chan81', 'chan121', 'chan122', 'chan114', 'chan119', 'chan113',\
                         'chan111', 'chan116', 'chan105', 'chan107', 'chan115', 'chan98', 'chan97', 'chan127',\
                         'chan106', 'chan124', 'chan123'],\
                        ['chan52', 'chan49', 'chan50', 'chan61', 'chan51', 'chan57', 'chan59', 'chan56', 'chan62',\
                         'chan35', 'chan60', 'chan44', 'chan54', 'chan39', 'chan55', 'chan48', 'chan64', 'chan33',\
                         'chan58', 'chan47', 'chan63', 'chan34', 'chan42', 'chan46', 'chan53', 'chan41', 'chan37',\
                         'chan40', 'chan36', 'chan45', 'chan43', 'chan38', 'chan19', 'chan18', 'chan17', 'chan24',\
                         'chan21', 'chan23', 'chan22', 'chan30', 'chan20', 'chan25', 'chan28', 'chan27', 'chan29',\
                         'chan31', 'chan32', 'chan6', 'chan26', 'chan14', 'chan13', 'chan15', 'chan12', 'chan16',\
                         'chan7', 'chan4', 'chan9', 'chan3', 'chan5', 'chan2', 'chan8', 'chan1', 'chan10', 'chan11']]

"""
PARAMETERS:
car_channels: [dictionary (key string (patient ID); Value: list > list > strings (channels))]; The sublists of channels
              that are CAR filtered together.
"""
