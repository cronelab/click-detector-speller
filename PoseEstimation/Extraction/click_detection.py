
# IMPORTING LIBRARIES
import collections # DELETE IF NECESSARY
import copy
import cv2
import itertools
import mediapipe as mp
import numpy as np
import pickle # DELETE IF NECESSARY
import os
import time
import xarray as xr # DELETE IF NECESSARY

from tqdm import trange

# README:
# This must be run from Roach (in VidRecog2 environment). Uploaded to Zappa only as a backup.

def detect_backspace(fps, mask_lower_red, mask_upper_red, mask_lower_yellow, mask_upper_yellow, n_frames, path_vid_backspace, t_backspace_dwell):
    """
    DESCRIPTION:
    For every frame of the keyboard video, a backspace click is represented by the backspace button and right stimulus row buttons switching
    from yellow and red respectively and turning to gray and white respectively. As such, after a backspace click, there should be no yellow
    or red anywhere in the region of the backspace key. Afterward, each detection is stored in a click highlight array. 

    Note: There is no dwell time on the backspace button when it is clicked. As such, the click is not visible as a button color change for 
          any duration of time. In order to properly record this click, an artificial time of t_backspace_dwell is used to lenghten this click
          "duration." This backspace dwell time is translated to frames using a frame counter threshold (see code). This is useful for down-
          sampling due to spectral analysis where it is common to use a 100 ms shift, which could completely skip the one or two frames where
          the backspace button is clicked. 

    INPUT VARIABLES:
    fps:                [int]; Frames per second (fps) of the recorded video.
    mask_lower_red:     [array > floats]; HSV lower limit for red.
    mask_upper_red:     [array > floats]; HSV upper limit for red.
    mask_lower_yellow:  [array > floats]; HSV lower limit for yellow.
    mask_upper_yellow:  [array > floats]; HSV upper limit for yellow.
    n_frames:           [int]; Total number of frames in the video.
    path_vid_backspace: [string]; Pathway to the video.
    t_backspace_dwell:  [float (units: ms)]; The artificial dwell time to remain on the backspace button. See Note.

    OUTPUT VARIABLES:
    click_highlights: [list > strings]; For every frame, a 'no_click' or 'backspace_click' is assigned to describe the activity of the speller.
    """

    # COMPUTATION:

    # Uploading the file for CV video capture.
    cap = cv2.VideoCapture(path_vid_backspace)

    # Initializing the list which will hold all the frames where a click highlight is made.
    click_highlights = ['no_click'] * n_frames

    # Initializing the boolean for whether red AND yellow appeared on the current and previous frame of an image. 
    contains_both      = False
    contains_both_prev = False

    # Computing the frame counter threshold. 
    frame_counter_thr = int((t_backspace_dwell/1000) * fps) # Units: (ms * s/ms) * frames/s = s * frames/s = frames

    # Initializing the frame counter.
    frame_counter = 0

    # Iterating across all channels.
    for frame_idx in trange(n_frames, position=0, desc='Iterating across all video frames.'):

        # Reading each frame from the Camera capture 
        _, image = cap.read()

        # Convert the BGR image to HSV.
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Look for the red color anywhere in the backspace region.
        mask_red     = cv2.inRange(image_hsv, mask_lower_red, mask_upper_red)
        mask_red     = mask_red.astype(bool)
        contains_red = (True in mask_red)
        
        # Look for the red color anywhere in the backspace region.
        mask_yellow     = cv2.inRange(image_hsv, mask_lower_yellow, mask_upper_yellow)
        mask_yellow     = mask_yellow.astype(bool)
        contains_yellow = (True in mask_yellow)
        
        # If both red and yellow are detected in the video frame.
        if contains_red and contains_yellow:
            contains_both = True
        else: 
            contains_both = False

        # If neither red nor yellow are detected in the current video frame. 
        if (not contains_red) and (not contains_yellow):
            contains_neither = True
        else: 
            contains_neither = False

        # If neither red nor yellow are  detected in the current frame, but they were in the previous frame, that means a click occurred in 
        # the current frame. Otherwise, set the click boolean to False.
        if contains_neither and contains_both_prev:
            click_detection = True
        else:
            click_detection = False

        # Setting the current boolean detector for both red and yellow to be the previous boolean decision for the next frame.
        contains_both_prev = contains_both

        # If there is a click detection, start the frame counter.
        if click_detection:
            frame_counter += 1

        # If the frame counter was started but is below the frame counter threshold, keep incrementing the frame counter and change the 
        # click highlight from 0 to 1 for the current frame.
        if (frame_counter > 0) and (frame_counter <= frame_counter_thr):
            frame_counter += 1
            click_highlights[frame_idx] = 'backspace_click'   

        # If the frame counter exceeds the frame coutner threshold, set it back to 0. 
        if frame_counter > frame_counter_thr:
            frame_counter = 0

        # Showing the hand with the pose-estimated coordinates.
        cv2.imshow('Speller Backspace Key', image)

        if cv2.waitKey(10) == ord('q'):
            break

    # Release the camera and close the camera window.
    cap.release()
    cv2.destroyAllWindows()

    return click_highlights


def detect_keyboard(mask_lower, mask_upper, n_frames, path_vid_keyboard, percent_pixels_thr):
    """
    DESCRIPTION:
    For every frame of the keyboard video, a word or character click is represented by the color teal being detected in the video
    frame. Each detection is stored in an array.

    INPUT VARIABLES:
    mask_lower:         [array > floats]; HSV lower limit for teal.
    mask_upper:         [array > floats]; HSV upper limit for teal.
    n_frames:           [int]; Total number of frames in the video.
    path_vid_keyboard:  [string]; Pathway to the video.
    percent_pixels_thr: [float (unit: %)]; Percentage of pixels which must be highlighted to suggest that a true click occurred. Otherwise, a click 
                        might be falsely detected due to one or two pixels being highlighted. This has happened before.

    OUTPUT VARIABLES:
    click_highlights: [list > strings]; For every frame, a 'no_click' or 'keyboard_click' is assigned to describe the activity of the speller.
    """

    # COMPUTATION:

    # Uploading the file for CV video capture.
    cap = cv2.VideoCapture(path_vid_keyboard)

    # Initializing the list which will hold all the frames where a click highlight is made.
    click_highlights = ['no_click'] * n_frames

    # Iterating across all channels.
    for frame_idx in trange(n_frames, position=0, desc='Iterating across all video frames.'):

        # Reading each frame from the Camera capture 
        _, image = cap.read()

        # Convert the BGR image to HSV.
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # If the teal color is anywhere in the current image, this means a keyboard selection was made. 
        mask = cv2.inRange(image_hsv, mask_lower, mask_upper)
        mask = mask.astype(bool)

        # Computing the total number of pixels and highlighted pixels for the current frame.
        n_pixels_total     = mask.shape[0] * mask.shape[1]
        n_pixels_highlight = sum(sum(mask))

        # Computing the percentage of pixels highlighted:
        percent_pixels_highlighted = n_pixels_highlight/n_pixels_total

        # If the number of pixels highlighted surpass the threshold, then a click was detected.
        if percent_pixels_highlighted > percent_pixels_thr:
            click_highlights[frame_idx] = 'keyboard_click'

        # Showing the hand with the pose-estimated coordinates.
        cv2.imshow('Speller Keyboard', image)
        
        if cv2.waitKey(10) == ord('q'):
            break

    # Release the camera and close the camera window.
    cap.release()
    cv2.destroyAllWindows()

    return click_highlights


def detect_stimcolumn(mask_lower, mask_upper, n_frames, path_vid_stimcolumn):
    """
    DESCRIPTION:
    For every frame of the keyboard video, a row selection click is represented by the color yellow being detected in the video
    frame (in the stimulus column). Each detection is stored in an array.

    INPUT VARIABLES:
    mask_lower:          [array > floats]; HSV lower limit for yellow.
    mask_upper:          [array > floats]; HSV upper limit for yellow.
    n_frames:            [int]; Total number of frames in the video.
    path_vid_stimcolumn: [string]; Pathway to the video.

    OUTPUT VARIABLES:
    click_highlights: [list > strings]; For every frame, a 'no_click' or 'stimcolumn_click' is assigned to describe the activity of the speller.
    """

    # COMPUTATION:

    # Uploading the file for CV video capture.
    cap = cv2.VideoCapture(path_vid_stimcolumn)

    # Initializing the list which will hold all the frames where a click highlight is made.
    click_highlights = ['no_click'] * n_frames

    # Iterating across all channels.
    for frame_idx in trange(n_frames, position=0, desc='Iterating across all video frames.'):

        # Reading each frame from the Camera capture 
        _, image = cap.read()

        # Convert the BGR image to HSV.
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # If the yellow color is anywhere in the current image, this means a keyboard selection was made. 
        mask = cv2.inRange(image_hsv, mask_lower, mask_upper)
        mask = mask.astype(bool)
        contains_yellow = (True in mask)
        if contains_yellow:
            click_highlights[frame_idx] = 'stimcolumn_click'            

        # Showing the hand with the pose-estimated coordinates.
        cv2.imshow('Speller Stimulation Column', image)
        
        if cv2.waitKey(10) == ord('q'):
            break

    # Release the camera and close the camera window.
    cap.release()
    cv2.destroyAllWindows()

    return click_highlights


def count_frames(path_vid):
    """
    DESCRIPTION:
    Counting the total number of frames in the video from the file pathway.

    INPUT VARIABLES:
    path_vid: [string]; Path to the video file to be extracted.

    OUTPUT VARIABLES:
    n_frames: [int]; Total number of frames in the video. 
    """

    # COMPUTATION: 

    # Grab a pointer to the video file.
    video = cv2.VideoCapture(path_vid)

    # Initialize the total number of frames read.
    total = 0

    start_time = time.time()
    # Loop over the frames of the video.
    while True:

        # Grab the current frame.
        grabbed, frame = video.read()

        # Check to see if we have reached the end of the video.
        if not grabbed:
            break

        # Increment the total number of frames read.
        total += 1

    print('Frame detection time: ', time.time() - start_time)
    print('Total number of frames: ', total)

    # Return the total number of frames in the video file
    return total


if __name__ == '__main__':

    """
    DESCRIPTION:
    Using the speller application, it is possible to click within the stimulus column (for selecting a row), within the keyboard (for selecting
    a letter, word, or functional key), or on the backspace key (far right). In the current version of the speller (Nov 8 to current), each of 
    these clicks are displayed by different colors on the buttons. A click on the stimulus column appears as yellow, a click in the keyboard 
    appears as teal, and a click on the backspace does not manifest (must use a combination of other colors on other buttons; see detect_backspace).

    In the following script, we extract the video frames during each of these clicks and save them as arrays to .npy files. The videos must be 
    post-processed from the keyboard video (itself cropped from a larger video which also contains the participant's hand, face and the main screen
    of the testing machine). Consequently, there must be three videos (one for each type of click).
    
    Each of the videos must satisfy the following format (see code):
    1) filename_vid_backspace  = date + '_Speller_block' + str(block) + '_backspace.mp4'
    2) filename_vid_keyboard   = date + '_Speller_block' + str(block) + '_keyboard.mp4'
    3) filename_vid_stimcolumn = date + '_Speller_block' + str(block) + '_stimcolumn.mp4'

    Finally, we must also provide the directory in which these videos are stored (dir_base).
    """

    # EXPERIMENTER INPUTS
    block_start        = 1
    block_end          = 3
    date               = '2022_11_18'
    dir_base           = 'D:/Hopkins/CroneLab2/Projects/CortiComm/Code/VideoInformation/Speller/'
    fps                = 30
    percent_pixels_thr = 0.0005 
    t_backspace_dwell  = 500 
    
    """
    INPUT VARIABLES:
    block_end:          [int]; The number of the last block from the date.
    block_start:        [int]; The number of the first block from the date.
    date:               [string]; The date ('yyyy_mm_dd') of the video from which clicks will be extracted.
    dir_base:           [string]; The directory which holds the videos for click detection.
    fps:                [int]; Frames per second (fps) of the recorded video.
    percent_pixels_thr: [float (unit: %)]; Percentage of pixels which must be highlighted to suggest that a true click occurred. Otherwise, a click 
                        might be falsely detected due to one or two pixels being highlighted (see Note in detect_keyboard function).
    t_backspace_dwell:  [int (units: ms)]; The artificial dwell time to remain on the backspace button (see Note in detect_backspace function).
    """

    # COMPUTATION:

    # Iterating across each experimenter-input block to detect clicks.
    for block in range(block_start, block_end+1):

        print('Block: ', block)

        # Defining the HSV mask limits for teal, red and yellow colors necessary for highlight detection.
        mask_lower_red    = np.array([170, 160, 70])
        mask_upper_red    = np.array([180, 250, 250])
        mask_lower_teal   = np.array([40, 20, 250])
        mask_upper_teal   = np.array([50, 255, 255])
        mask_lower_yellow = np.array([15, 230, 240])
        mask_upper_yellow = np.array([30, 255, 255])

        # Defining the directory for the videos corresponding to the current date and block.
        dir_vid = dir_base + 'videos/' + date + '/Block' + str(block) + '/'

        # Defining the video file names for backspace, keyboard and stimulus columns corresponding to the current date and block.
        filename_vid_backspace  = date + '_Speller_block' + str(block) + '_backspace.mp4'
        filename_vid_keyboard   = date + '_Speller_block' + str(block) + '_keyboard.mp4'
        filename_vid_stimcolumn = date + '_Speller_block' + str(block) + '_stimcolumn.mp4'

        # Defining the file paths for the backspace, keyboard and stimulus column videos.
        path_vid_backspace  = dir_vid + filename_vid_backspace
        path_vid_keyboard   = dir_vid + filename_vid_keyboard
        path_vid_stimcolumn = dir_vid + filename_vid_stimcolumn
        
        # Extracting the total number of frames in the video. Since all three videos should be the exact same length, there is no 
        # need to count the number of frames for more than one video. 
        n_frames = count_frames(path_vid_backspace)
        # n_frames = 500 # DELETE IF NECESSARY

        # Creating the click detections from the keyboard, backspace key, and stimulus columns. 
        click_highlights_backspace  = detect_backspace(fps, mask_lower_red, mask_upper_red, mask_lower_yellow, mask_upper_yellow, n_frames, path_vid_backspace, t_backspace_dwell)
        click_highlights_keyboard   = detect_keyboard(mask_lower_teal, mask_upper_teal, n_frames, path_vid_keyboard, percent_pixels_thr)
        click_highlights_stimcolumn = detect_stimcolumn(mask_lower_yellow, mask_upper_yellow, n_frames, path_vid_stimcolumn)

        ############################################################################################
        # DELETE IF NECESSARY
        
        # Computing the array of seconds.
        time_seconds = np.round(np.arange(n_frames)/fps, 3)   # frames x sec/frame = sec

        # Converting the above three highlights arrays into xarrays.
        click_highlights_backspace  = xr.DataArray(click_highlights_backspace,
                                                   coords={'time_seconds': time_seconds},
                                                   dims=["time_seconds"])
        click_highlights_keyboard   = xr.DataArray(click_highlights_keyboard,
                                                   coords={'time_seconds': time_seconds},
                                                   dims=["time_seconds"])
        click_highlights_stimcolumn = xr.DataArray(click_highlights_stimcolumn,
                                                   coords={'time_seconds': time_seconds},
                                                   dims=["time_seconds"])

        # Initializing the click highlights dictionary.
        click_highlights = collections.defaultdict(dict)
        click_highlights['backspace']['data']  = click_highlights_backspace
        click_highlights['keyboard']['data']   = click_highlights_keyboard
        click_highlights['stimcolumn']['data'] = click_highlights_stimcolumn
        
        click_highlights['backspace']['plotcolor']  = 'gold'
        click_highlights['keyboard']['plotcolor']   = 'green'
        click_highlights['stimcolumn']['plotcolor'] = 'red'

        ############################################################################################








        # Saving the click highlights for each detection.
        # dir_click_detections = dir_base + 'click_detections/' + date +'/Block' + str(block) + '/' 
        dir_click_detections = dir_base + 'click_detections/' + date +'/Xarrays/' # DELETE IF NECESSARY

        # Checking whether or not the directory for click detections exists. If it does not exist,  it is created here.
        dir_click_detections_exists = os.path.exists(dir_click_detections)
        if not dir_click_detections_exists:
            os.makedirs(dir_click_detections)

        # Defining the files which hold the click highlights for the word/characters, backspace key and stimulus columns. 
        file_click_highlights_backspace  = date + '_Block' + str(block) + '_click_highlights_backspace'
        file_click_highlights_keyboard   = date + '_Block' + str(block) + '_click_highlights_keyboard'
        file_click_highlights_stimcolumn = date + '_Block' + str(block) + '_click_highlights_stimcolumn'

        # Defining the file paths for saving the click highlights.
        path_click_highlights_backspace  = dir_click_detections + file_click_highlights_backspace
        path_click_highlights_keyboard   = dir_click_detections + file_click_highlights_keyboard
        path_click_highlights_stimcolumn = dir_click_detections + file_click_highlights_stimcolumn

        # Saving the click detection arrays.
        # np.save(path_click_highlights_backspace, click_highlights_backspace, allow_pickle = True)
        # np.save(path_click_highlights_keyboard, click_highlights_keyboard, allow_pickle = True)
        # np.save(path_click_highlights_stimcolumn, click_highlights_stimcolumn, allow_pickle = True)



        # DELETE IF NECESSARY
        path_click_highlights = dir_click_detections + date + '_Block' + str(block) + '_click_highlights'
        with open(path_click_highlights, "wb") as fp: pickle.dump(click_highlights, fp)