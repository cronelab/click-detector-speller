# IMPORTING LIBRARIES:
import collections
import shutil


################################### SAVE SELF ###################################
def save_script_backup():
    """
    Automatically saving this entire script immediately when it's called.
    """
    
    # Creating the original and target file directories.
    original = r'/home/dan/Projects/PseudoOnlineTests_for_RTCoG/Scripts/Recent/OfflineTraining_BrainClick/params_model_training_visual_labeling.py'
    target   = r'/mnt/shared/danprocessing/BACKUP/Projects/PseudoOnlineTests_for_RTCoG/Scripts/OfflineTraining_BrainClick/params_model_training_visual_labeling.py'

    # Saving.
    shutil.copyfile(original, target)
    
# Immediately saving script.   
save_script_backup()


######################## PATIENT AND DATA INFORMATION #########################
aw_model_type   = 'shift'
car             = True
calib_state_val = 1# 1
f_power_max     = [170] # [45, 115, 170]
f_power_min     = [110] # [15, 70, 125]
file_extension  = 'mat'
model_classes   = ['rest', 'grasp']
model_name      = 'TestModel'
model_type      = 'LSTM'
n_pc_thr        = 'None'
patient_id      = 'CC01'
percent_var_thr = 'None'
sampling_rate   = 1000 # 1000
sxx_shift       = 100
sxx_window      = 256

"""
PARAMETERS:
aw_model_type:   [string ('shift'/'piecewise')]; The type of affine warp transformation the experimenter wishes to 
                 perform. Leave empty [] if no AW-alignment will occur.
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
percent_var_thr: [float (between 0 and 1)]; The percent variance which the user wishes to capture with the principal components. Will compute the number
                 of principal components which capture this explained variance as close as possible, but will not surpass it. Set to 'None'
                 if n_pc_thr is not 'None', or set to 'None' along with n_pc_thr if all of the variance will be used (no PC transform).
sampling_rate:   [int (samples/s)]; Sampling rate at which the data was recorded.
sxx_shift:       [int (units: ms)]; Length of time by which sliding window (sxx_window) shifts along the time domain.
sxx_window:      [int (units: ms)]; Time length of the window that computes the frequency power.
"""



############################ LSTM PARAMETERS ###################################
alpha         = 0.001
batch_size    = 45
dropout       = 0.3
epochs        = 30 # 10
n_hidden_lstm = 25

"""
PARAMETERS:
alpha:         [float]; Learning rate for gradient descent
batch_size:    [int]; Number of samples on which the LSTM is trained with each step.
dropout:       [float (between 0 and 1)]; Percentage of all hidden layers to be excluded from update during one backprop step.
epochs:        [int]; Number of steps that are taken to train the LSTM.
n_hidden_lstm: [int]; Number of hidden units in the LSTM.
"""


########################## CHANNEL GRIDS ##########################

# Creating the dictionary for the grid configuration.
grid_config = collections.defaultdict(dict)

grid_config['CC01']\
           ['upperlimb'] = [["chan121", "chan122", "chan123", "chan124", "chan125", "chan126", "chan127", "chan128"],
                            ["chan113", "chan114", "chan115", "chan116", "chan117", "chan118", "chan119", "chan120"],
                            ["chan105", "chan106", "chan107", "chan108", "chan109", "chan110", "chan111", "chan112"],
                            ["chan97", "chan98", "chan99", "chan100", "chan101", "chan102", "chan103", "chan104"],
                            ["chan89", "chan90", "chan91", "chan92", "chan93", "chan94", "chan95", "chan96"],
                            ["chan81", "chan82", "chan83", "chan84", "chan85", "chan86", "chan87", "chan88"],
                            ["chan73", "chan74", "chan75", "chan76", "chan77", "chan78", "chan79", "chan80"],
                            ["chan65", "chan66", "chan67", "chan68", "chan69", "chan70", "chan71", "chan72"]]

grid_config['CC01']\
           ['speech'] = [["chan57", "chan58", "chan59", "chan60", "chan61", "chan62", "chan63", "chan64"],
                         ["chan49", "chan50", "chan51", "chan52", "chan53", "chan54", "chan55", "chan56"],
                         ["chan41", "chan42", "chan43", "chan44", "chan45", "chan46", "chan47", "chan48"],
                         ["chan33", "chan34", "chan35", "chan36", "chan37", "chan38", "chan39", "chan40"],
                         ["chan25", "chan26", "chan27", "chan28", "chan29", "chan30", "chan31", "chan32"],
                         ["chan17", "chan18", "chan19", "chan20", "chan21", "chan22", "chan23", "chan24"],
                         ["chan9", "chan10", "chan11", "chan12", "chan13", "chan14", "chan15", "chan16"],
                         ["chan1", "chan2", "chan3", "chan4", "chan5", "chan6", "chan7", "chan8"]]





############################# EXCLUDED CHANNELS #############################

# Initializing the dictionary for channels to be eliminated from future analysis. These are in addition to previously recorded bad channels. 
elim_channels = {}
elim_channels['CC01'] = ['ainp1','ainp2','ainp3']
# elim_channels['CC01'] = ['chan19','chan38','chan48','chan52','ainp1','ainp2','ainp3']

# elim_channels['CC01'] = ['chan'+str(n) for n in range(129)] + ['ainp1','ainp2','ainp3']
# elim_channels['CC01'].remove('chan108')
# elim_channels['CC01'].remove('chan118')
# elim_channels['CC01'].remove('chan110')
# elim_channels['CC01'].remove('chan94')
# elim_channels['CC01'].remove('chan109')
# elim_channels['CC01'].remove('chan100')
# elim_channels['CC01'].remove('chan102')
# elim_channels['CC01'].remove('chan92')
# elim_channels['CC01'].remove('chan93')
# elim_channels['CC01'].remove('chan117')
# elim_channels['CC01'].remove('chan101')
# elim_channels['CC01'].remove('chan116')

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





