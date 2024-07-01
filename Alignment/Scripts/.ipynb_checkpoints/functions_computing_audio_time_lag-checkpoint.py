
# IMPORTING LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal as signal
import xarray as xr

from h5eeg import H5EEGFile

from scipy.io import loadmat
from scipy.io import wavfile





def aligning_bci2000_audio_and_click(audio_from_bci2k, t_lag):
    """
    DESCRIPTION:
    Aligning the BCI2000 audio array with the video audio array according to the time lag of the video audio relative to
    the BCI2000 audio. 

    INPUT VARIABLES:
    audio_from_bci2k: [xarray (timesamples,) > floats]; Audio signal recorded from BCI2000. Time dimension is in units
                      of s.
    t_lag:            [float (units: ms)]; The time lag between the audio from BCI2000 and the video audio. In other 
                      words, t_lag is the amount of time that the BCI2000 audio signal is leading the video audio 
                      signal. If t_lag = 150 ms, this means that BCI2000 audio is ahead of the video audio by 150 ms.
                      For example, an audio event registered by the video to be at 3.0 s would actually be registered at
                      3.15 s by BCI2000. 

    OUTPUT VARIABLES:
    audio_from_bci2k_aligned: [xarray (timesamples,)> floats]; Aligned audio from BCI2000 accounting for the relative
                              time lag of the video audio. Time array is in units of s at BCI2000 sampling rate.
    """
    
    # COMPUTATION:
    
    # Converting the lag into units of seconds.
    t_lag_seconds = t_lag/1000 # ms * (1 s/1000 ms) = s

    # Extracting the time array from the BCI2000 audio array.
    bci2k_times = audio_from_bci2k.time_seconds
        
    # Extracting only audio time after lag.
    bci2k_lag        = bci2k_times - t_lag_seconds # units of seconds. 
    bci2k_bool       = bci2k_lag > 0
    bci2k_times_bool = bci2k_lag[bci2k_bool]
    
    # Zero-ing the time from BCI2000. 
    bci2k_times_zeroed = bci2k_times_bool - bci2k_times_bool[0] + bci2k_times[0]

    # Creating an xarray of the aligned BCI2000 audio.
    audio_from_bci2k_aligned= xr.DataArray(audio_from_bci2k[bci2k_bool],
                                           coords={'time_seconds': bci2k_times_zeroed},
                                           dims=["time_seconds"])

    return audio_from_bci2k_aligned





def cross_corr_audio(audio_from_bci2k, audio_from_video, sampling_rate_audio_from_bci2k, t_end_xcorr, t_start_xcorr):
    """
    DESCRIPTION:
    Cross correlating the audio signals from BCI2000 and the OBS video feed to find how much the Video's audio lags
    behind the audio from BCI2000.
    
    INPUT VARIABLES:
    audio_from_bci2k:               [xarray (timesamples,) > floats]; Audio signal recorded from BCI2000. Time dimension
                                    is in units of s.
    audio_from_video:               [xarray (timesamples,) > int]; Downsampled video audio signal array that matches the 
                                    sampling rate of the BCI2000 analog input. Time dimension is in units of s.
    sampling_rate_audio_from_bci2k: [int (units: Hz)]; The sampling rate of the audio signal from BCI2000.
    t_xcorr_end:                    [int (units: s)]; Describes the ending time point of the audio signal segment that 
                                    will be cross correlated.
    t_xcorr_start:                  [int (units: s)]; Describes the starting time point of the audio signal segment that
                                    will be cross correlated.

    OUTPUT VARIABLES:
    t_lag: [float (units: ms)]; The time lag between the audio from BCI2000 and the video audio. In other words, t_lag
           is the amount of time that the BCI2000 audio signal is leading the video audio signal. If t_lag = 150 ms, 
           this means that BCI2000 audio is ahead of the video audio by 150 ms. For example, an audio event registered
           by the video to be at 3.0 s would actually be registered at 3.15 s by BCI2000. 
    """
    # COMPUTATION:
    
    # Computing starting and ending samples describing time segment for cross correlation.
    samples_start_xcorr = int(t_start_xcorr*sampling_rate_audio_from_bci2k) # s * sa/s = sa
    samples_end_xcorr   = int(t_end_xcorr*sampling_rate_audio_from_bci2k)   # s * sa/s = sa
    
    # Extracting the audio segments from BCI2000 and from the Video feed that will be used for cross correlation.
    audio_from_bci2k_xcorr = audio_from_bci2k[samples_start_xcorr:samples_end_xcorr]
    audio_from_video_xcorr = audio_from_video[samples_start_xcorr:samples_end_xcorr]
    
    # Computing the number of samples from the audio from BCI2000 and the audio from the video file.
    n_samples_audio_from_bci2k = audio_from_bci2k_xcorr.shape[0]
    n_samples_audio_from_video = audio_from_video_xcorr.shape[0]

    # Computing the cross correlation between the BCI2000 audio signal and the video audio signal. 
    corr      = signal.correlate(audio_from_bci2k_xcorr, audio_from_video_xcorr)
    corr_lags = signal.correlation_lags(n_samples_audio_from_bci2k, n_samples_audio_from_video)

    # Using the index location of maximum correlation as the lag.
    idx_max_corr = np.argmax(corr) 
    idx_lag      = corr_lags[idx_max_corr]
        
    # By using the maximum correlation lag index, computing the time lag (units: ms) that BCI2000 audio is ahead of the
    # audio from the video file.
    factor = sampling_rate_audio_from_bci2k/1000 # sa/ms
    t_lag  = int(idx_lag/factor)                 # sa x ms/sa = ms
    
    # Printing
    print('Time lag (ms): ', t_lag)

    return t_lag





def downsample_video_audio_to_bci2k(audio_from_video, sampling_rate_audio_from_bci2k, sampling_rate_audio_from_video):
    """
    DESCRIPTION:
    Downsampling the audio signal array from the video feed to match the time resolution of the audio signal array
    from BCI2000.

    INPUT VARIABLES:
    audio_from_video:               [xarray (timesamples,) > floats]; Audio signal recorded from the video feed. Time 
                                    dimension is in units of s.
    sampling_rate_audio_from_bci2k: [int (units: Hz)]; The sampling rate of the audio signal from BCI2000.
    sampling_rate_audio_from_video: [int (units: Hz)]; The sampling rate of the audio signal from the video file.

    OUTPUT VARIABLES:
    audio_from_video_downsampled: [xarray (timesamples,) > int]; Downsampled video audio signal array that matches the
                                  sampling rate of the BCI2000 analog input. Time dimension is in units of s.
    """
    
    # COMPUTATION:
    
    # Computing the ratio of the audio sampling rate from the video file as compared to the audio sampling rate from 
    # BCI2000. 
    audio_ratio = int(sampling_rate_audio_from_video/sampling_rate_audio_from_bci2k)
        
    # Downsampling the audio signal from the video file using the audio ratio.    
    audio_from_video_downsampled = audio_from_video[audio_ratio-1::audio_ratio]

    # PRINTING:
    print('\nAUDIO SIGNAL:')
    print(audio_from_video_downsampled)
    
    return audio_from_video_downsampled 





def extract_audio_from_bci2k(ainp, block_id, date, file_extension, patient_id, task):
    """
    DESCRIPTION:
    Extracting the audio signal recorded by BCI2000 as an array using the experimenter-specified analog port.
    
    INPUT VARIABLES:
    ainp:           [string]; BNC ports recorded by BCI2000. Choose the port in which the audio signal was recorded.
                    Options are: 'ainp1', 'ainp2', 'ainp3'.
    block_id:       [String (BlockX, where X is an int)]; Block ID of the task that was run.
    date:           [string (YYYY_MM_DD)]; Date on which the block was run.
    file_extension: [string (hdf5/mat)]; The file type of BCI2000 data from which the audio signal will be extracted.
    patient_id:     [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:           [string]; Type of task that was run.                    
                            
    OUTPUT VARIABLES:
    audio_signal:  [xarray > floats]; Audio signal recorded from BCI2000. Time dimension is in units of s.
    sampling_rate: [int (units: Hz)]; The sampling rate of the audio signal from BCI2000.
    """
    
    # COMPUTATION:
    
    # Creating the path for the current date/task pair.
    audio_signal_path = '/mnt/shared/ecog/' + patient_id + '/' + file_extension + '/' + date + '/' + task + '_' + \
                        block_id + '.' + file_extension
    
    # Extracting all eeg signals and sampling rate from the audio pathway based on if the file extension is .hdf5 or
    # .mat.
    if file_extension == 'hdf5':
        
        # Extracting the current date's .hdf5 data.
        h5file        = H5EEGFile(audio_signal_path)
        eeg           = h5file.group.eeg(); 
        eeglabels     = eeg.get_labels(); 
        eeg_signals   = eeg.dataset[:]
        sampling_rate = int(eeg.get_rate())
        
    if file_extension == 'mat':

        # Extracting the current date's .mat data.
        matlab_data   = loadmat(audio_signal_path, simplify_cells=True)
        eeglabels     = matlab_data['parameters']['ChannelNames']['Value']  
        eeg_signals   = matlab_data['signal']
        sampling_rate = int(matlab_data['parameters']['SamplingRate']['NumericValue'])        
    
    # Creating the time array.
    time_seconds = np.arange(eeg_signals.shape[0])/sampling_rate
    
    # Extracting the appropriate channel index from the original eeglabels list.
    eeg_ind = np.argwhere(eeglabels == ainp)[0][0]    
    
    # Extracting the audio signal from the array of eeg signals.
    audio_signal = eeg_signals[:,eeg_ind].astype('float64')
    
    # Converting the audio signal into an xarray.
    audio_signal = xr.DataArray(audio_signal,
                                coords={'time_seconds': time_seconds},
                                dims=["time_seconds"])
    
    # PRINTING:
    print('\nAUDIO SAMPLING RATE (sa/s):')
    print(sampling_rate)
    print('\nAUDIO SIGNAL:')
    print(audio_signal)
    
    return audio_signal, sampling_rate





def extract_audio_from_video(block_id, date, dir_audio, patient_id, task):
    """
    DESCRIPTION:
    Extracting as an array the audio signal recorded by the video using the .wav file converted from Microsoft OBS.
    
    INPUT VARIABLES:
    block_id:   [String (BlockX, where X is an int)]; Block ID of the task that was run.
    date:       [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_audio:  [string]; Directory where the audio information is stored.
    patient_id: [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    task:       [string]; Type of task that was run.
    
    OUTPUT VARIABLES:
    audio_signal:  [xarray (timesamples, )> floats]; Audio signal recorded from the video feed. Time dimension is in 
                   units of s.
    sampling_rate: [int (units: Hz)]; The sampling rate of the audio signal from the video file.
    """
    
    # COMPUTATION:
    
    # Creating the filename for the audio file corresponding to date_block pair.
    filename_wav = date + '_' + block_id + '_audio.wav'

    # Creating the audio signal pathway.
    audio_signal_path = dir_audio + patient_id + '/' + task + '/' + 'Audio/' + date + '/' + filename_wav

    try:
        # Extracting the audio signal and sampling rate from the audio signal path.
        sampling_rate, audio_signal = wavfile.read(audio_signal_path)
    except:
        print('No Audio File. No alignment of click highlights will occur')
        audio_sampling_rate = None
        audio_signal        = None
        
    # Converting the audio signal from int16 (by default) to float64.
    audio_signal = audio_signal[:,0].astype('float64')
    
    # Computing the total number of samples from the video audio to create a time array at the video audio resolution.
    n_samples_audio = audio_signal.shape[0]
    t_seconds_audio = np.arange(n_samples_audio)/sampling_rate
    
    # Converting the audio signal into an xarray.
    audio_signal = xr.DataArray(audio_signal,
                                coords={'time_seconds': t_seconds_audio},
                                dims=["time_seconds"])

    # PRINTING:
    print('\nAUDIO SAMPLING RATE (sa/s):')
    print(sampling_rate)
    print('\nAUDIO SIGNAL:')
    print(audio_signal)
    
    return audio_signal, sampling_rate





def plotting_audio_signals(audio_from_bci2k, audio_from_video, t_view_end, t_view_start):
    
    """
    DESCRIPTION:
    Plotting the audio signals recorded from the video camera and by BCI2000 prior to alignment.
    
    INPUT VARIABLES:
    audio_from_bci2k: [xarray (timesamples,) > floats]; Audio signal recorded from BCI2000. Time dimension is in units
                      of s.
    audio_from_video: [xarray (timesamples,) > int]; Downsampled video audio signal array that matches the sampling rate
                      of the BCI2000 analog input. Time dimension is in units of s.
    t_view_end:       [float (units: s)]; End of viewing window.
    t_view_start:     [float (units: s)]; Start of viewing window.
    """
    
    # COMPUTATION:
    
    # Extracting the time arrays from the audio signals from BCI2000 and the Video feed.
    t_seconds_audio_from_bci2k = audio_from_bci2k.time_seconds
    t_seconds_audio_from_video = audio_from_video.time_seconds

    # Creating boolean arrays that will be used to show the audio clips only within the user specified start and end 
    # viewing times.
    zoom_bool_audio_from_bci2k = np.logical_and(t_seconds_audio_from_bci2k > t_view_start,\
                                                t_seconds_audio_from_bci2k < t_view_end)
    zoom_bool_audio_from_video = np.logical_and(t_seconds_audio_from_video > t_view_start,\
                                                t_seconds_audio_from_video < t_view_end)

    # Extracting the zoomed-in versions of the audio signal and corresponding time signal.
    audio_from_bci2k_zoom           = audio_from_bci2k[zoom_bool_audio_from_bci2k]
    audio_from_video_zoom           = audio_from_video[zoom_bool_audio_from_video]
    t_seconds_audio_from_bci2k_zoom = t_seconds_audio_from_bci2k[zoom_bool_audio_from_bci2k]
    t_seconds_audio_from_video_zoom = t_seconds_audio_from_video[zoom_bool_audio_from_video]

    # PLOTTING
    fig = plt.figure(figsize=(15,10))
    plt.plot(t_seconds_audio_from_video_zoom, audio_from_video_zoom, label = 'Video Audio')
    plt.plot(t_seconds_audio_from_bci2k_zoom, audio_from_bci2k_zoom, label = 'BCI2000 Audio')
    plt.title('Audio Signals from OBS Video and BCI2000 Analog Input')
    plt.xlabel('Time (s)')
    plt.legend(loc="upper left")
    plt.grid()
    
    
    
    
    
def save_time_lag(block_id, date, dir_time_lag, t_lag, task):
    """
    DESCRIPTION:
    Saving the time lag to the appropriate folder.
    
    INPUT VARIABLES:
    block_id:     [String (BlockX, where X is an int)]; Block ID of the task that was run.
    date:         [string (YYYY_MM_DD)]; Date on which the block was run.
    dir_time_lag: [string]; Directory where the time lags are stored.
    t_lag:        [float (units: ms)]; The time lag between the audio from BCI2000 and the video audio. In other words,
                  t_lag is the amount of time that the BCI2000 audio signal is leading the video audio signal. If 
                  t_lag = 150 ms, this means that BCI2000 audio is ahead of the video audio by 150 ms. For example, an
                  audio event registered by the video to be at 3.0 s would actually be registered at 3.15 s by BCI2000. 
    task:         string]; Type of task that was run. 
    """
    
    # COMPUTATION:
    
    # Creating the base path and filename for the time lags.
    dir_lag      = dir_time_lag + date + '/' + block_id + '/'
    filename_lag = date + '_' + block_id + '.txt'

    # Check to see if the directory for the lag exists.
    dir_lag_exist = os.path.exists(dir_lag)

    # If the directory doesn't exist, create it.
    if not dir_lag_exist:
        os.makedirs(dir_lag)

    # Creating the pathway for the lag for the current date+block pair.
    path_lag = dir_lag + filename_lag

    # Saving the time lag.
    f = open(path_lag, "w")
    f.write(str(t_lag))
    f.close()