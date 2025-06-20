# Click Detector Codebase

This repository containsthe code for the following paper:
**[A click-based electrocorticographic brain-computer interface enables long-term high-performance switch scan spelling](https://www.nature.com/articles/s43856-024-00635-3)** by Candrea et al. 



## Section 0: Conversion from .dat to .mat
Use the convert-to-mat  file to convert the .dat file to a .mat file. The resulting .mat file is imported as a dictionary which has three keys: signal, states, and parameters. The signal key contains the time domain of all neural and analog channels, the states key contains the time domain of all state changes (cue presentations, clicks, etc.), and the parameters key contain all parameters from a block of data (sampling rate, number of channels, etc.).

## Section 1: Model Training
Run **model_training.ipynb**. Training data was collected using a “Go” task, where the participant was instructed to perform a brief grasping movement every time “Go” appeared on a monitor. High gamma (HG, 70-110 Hz) spectral power was computed and used to train a rest vs. grasp classification model (LSTM). As our training pipeline assumes that the participant is in a completely locked-in state, the movement onset and offset were not used to inform training labels – rather the onsets and offsets of HG modulation were used instead. Affine warping was used to re-align power-trial rasters for more accurate labeling of trial-averaged HG onset and offset (doi: 10.1016/j.neuron.2019.10.020, https://github.com/ahwillia/affinewarp.git ). A confusion matrix showing the sample-by-sample confusion score from cross-validation is computed to estimate the classification model’ accuracy on unseen data.

## Section 2: Pose Estimation
### Computation
When testing the switch-scanning speller, a video camera was used to capture the participant's movements of his right hand. This is because he would generate clicks by attempting single grasps of his hand and these residual movements were still observable. Thus, these were used as ground truth for movement onset. Simultaneously the monitor displaying the spelling application to the participant was also recorded. 

The participant’s hand movements can be tracked throughout the duration of a spelling block using **hand_trajectory_extraction.py**. In this script, a pose-estimation algorithm (doi:10.48550/ARXIV.2006.10214) computes the hand landmarks across time from a video file and saves them as an xarray. Secondly, the button clicks from the spelling application can be tracked using **click_detection.py**. Specifically, this script looks for changes in color in specific areas of the video that correspond to button clicks. Note that the stimulus column (pre-selection column for a preparation period) may also be highlighted due to multiple cycles through the row and not button clicks – these must be accounted for in analysis. Click on the **run_hand_and_clicks.bat** file to run both scripts sequentially.

### Segmentation
Onset and offset of attempted hand movements were not used to inform training labeling. However, they can still be used to compute ground-truth performance metrics. Specifically, the onset of attempted grasps can be used to inform true and false positive click detections as well as click-latencies. The **pose_estimation_segmentation.ipynb** script is used to manually label the onset and offset of these attempted hand movements using the hand landmarks recorded across the duration of a spelling block (see above). After manually segmenting onsets and offsets from a few spelling blocks, the **pose_estimation_model_training.ipynb** can be used to train an automated onset and offset detector. Finally, this automated detector is used in **pose_estimation_segmentation_inference.ipynb** to automatically detect these attempted movement onsets and offsets. It is possible to then use the pose_estimation_segmentation.ipynb to correct any mistakes that the model makes.

## Section 3: Aligning Neural and Video Data
Since video of the hand and monitor were recorded continuously across multiple spelling blocks, a recording for each block had to be cropped to approximately the starting and ending time in a video-editing tool such as Adobe Premiere Pro. The first video frame in which the BCI2000 Operator status was “Running” marked the start of the block, which was chosen to approximate the start of data recording by BCI2000. For some blocks, the participant was instructed to wait several more seconds before actually beginning to use the speller (misc. preparation on the experimenter’s side). As such, after the pose-estimation scripts (in Computation section) are run, we can determine a fine-tuned starting and ending time for each block, convert these from hh:mm:ss:ff (default in Premiere Pro) to units of seconds in **computing_block_start_stop_times.ipynb** and crop the hand and click arrays accordingly in **curtailing_video_data.ipynb**. 

To align the neural data recorded by BCI2000 with the video of the hand and monitor, an audio synchronization cue was played at the beginning of each spelling block. The time lag between BCI2000 recording and video capture can then be computed using **computing_audio_time_lag.ipynb** and is used in **curtailing_bci2000_data.ipynb** to appropriately shift the BCI2000 neural signals.

After alignment, it is now possible to append in the .mat file the movement onset and offset times to the states information recorded by BCI2000. This is done in **adding_movement_states.ipynb**. This information could be used to inform labeling if labeling by visually determining onset and offset of HG modulation is not sufficient.

## Section 4: Simulated Clicks
Once a classification model is created (see Section 1), it is possible to test this model on any previously recorded data. In particular, **simulating_clicks.ipynb** is used to play back data from spelling blocks and preprocesses it for classification as well as applying post-processing (voting) in the same way as in online use. As such, this script generates simulated clicks as if they were computed online. Note that no spelling rates can be directly computed because the simulated clicks give no information on which keys could have actually been pressed. However, this script can also be used to play back data from other tasks (even non-upper-limb related) if that is helpful.

## Section 5: Analysis
### Online Performance
After using a click detector in real time in a spelling block, **speller_analysis_online_click_detections.ipynb** can be used to compute the true and false positives, as well as the mean and standard deviation of the latency to algorithm detection as well as on-screen click  for that block. Additionally, **speller_analysis_online_click_latencies.ipynb** can be used to aggregate click latencies over multiple blocks and compute their combined mean and standard deviation.

### Simulated Performance
Similarly, the **speller_analysis_simulated_click_detections.ipynb** can be used to compute the true and false positives, but can only compute the latency information about the algorithmic detection, since in simulation, there is no on-screen click. 
