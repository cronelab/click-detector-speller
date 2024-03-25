
# IMPORTING LIBRARIES
import copy
import cv2
import itertools
import mediapipe as mp
import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)

import skimage.measure

import os
import time

from tqdm import trange

# README:
# This must be run from Roach (in VidRecog2 environment). Uploaded to Zappa only as a backup.



def detect_keyboard_highlights(n_frames, path_vid_keyboard_no_stim_rows, percent_pixels_thr):
    """
    DESCRIPTION:
    Every frame of the keyboard video is compared to the previous frame to see whether a change in the highlighted row or column has occurred.

    INPUT VARIABLES:
    n_frames:                       [int]; Total number of frames in the video.
    path_vid_keyboard_no_stim_rows: [string]; Pathway to the video.
    percent_pixels_thr:             [float (unit: %)]; Percentage of pixels which must be changed from the previous frame for detecting a highlight.
    
    OUTPUT VARIABLES:
    keyboard_highlights: [list > strings]; For every frame, a 'pause' (no highlight change) or 'switch' (highlight change) is assigned to describe
                          whether the highlight changed from one row to another.
    """

    # COMPUTATION:

    # Uploading the file for CV video capture.
    cap = cv2.VideoCapture(path_vid_keyboard_no_stim_rows)

    # Initializing the list which will hold all the frames where a highlight detection (switch) is made.
    keyboard_highlights = ['pause'] * n_frames



    # Iterating across all channels.
    for frame_idx in trange(n_frames, position=0, desc='Iterating across all video frames.'):

        # Reading each frame from the Camera capture 
        _, image = cap.read()

        # Convert the BGR image to HSV.
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Extracting only the hue.
        # image_h = image_hsv[:,:,0]

        # Downsample the image by pooling.
        # image_h = skimage.measure.block_reduce(image_h, (2,2), np.max)

        if frame_idx == 0:
            image_buffer = np.zeros((image_hsv.shape[0], image_hsv.shape[1], 2))

        # Computing the total number of pixels. 
        # n_pixels_total = image_hsv.shape[0] * image_hsv.shape[1]
        # n_pixel_values_total = sum(sum(image_h))

        # After the 0th frame.
        if frame_idx > 1:
            
            h, w, z = image_hsv.shape;
            
            # print(image_h[550:600,550:600])
            # print(image_h[550:600,550:600] == image_buffer[550:600,550:600,1])

            # diff = cv2.subtract(image_hsv, image_hsv_prev)
            # diff = np.abs(image_h[200:250,0:50] - image_buffer[200:250,0:50,1]);
            # diff = np.abs(image_h[200:250,0:50] - image_prev[200:250,0:50]);
            diff = np.abs(image_hsv - image_prev);

            # print(diff)
            # inds = np.where(diff < 250)
            # print('INDS: ', inds)

            # diff[inds] = 0
            # diff = image_h[550:600,550:600] - image_buffer[550:600,550:600,1];
            err = np.sum(diff**2);
            mse = err/(float(h*w));
            msre = round(np.sqrt(mse), 3);

            print('\nMSRE: ',msre)
            # # Computing the change in image
            # # image_del = image_hsv == image_hsv_prev
            # image_del = image_hsv - image_hsv

            # # Computing the number of changed pixels.
            # n_changed_pixels = sum(sum(image_del))

            # # Computing the percentage of changed pixels.
            # print(n_changed_pixels, n_pixel_values_total)
            # # percent_changed_pixels = n_changed_pixels/n_pixels_total
            # percent_changed_pixels = n_changed_pixels/n_pixel_values_total

            # print(percent_changed_pixels)

            # # If more than the experimenter-set percent of pixels are different, detect a switched highlight.
            # if percent_changed_pixels > percent_pixels_thr:
            #     keyboard_highlights[frame_idx] = 'switch'

            #     print('HIGHLIGHT SWITCH')

        # Assigning the current image to next iteration's previous image
        # image_hsv_prev = image_hsv

        # image_buffer[:,:,1] = image_buffer[:,:,0]
        # image_buffer[:,:,0] = image_h

        image_prev = image_hsv
    

        # Showing the keyboard without the stimulus rows.
        # cv2.imshow('Speller Keyboard', image[200:250,0:50])
        cv2.imshow('Speller Keyboard', image)
        
        if cv2.waitKey(10) == ord('q'):
            break

    # Release the camera and close the camera window.
    cap.release()
    cv2.destroyAllWindows()

    return keyboard_highlights


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
    block_start        = 4
    block_end          = 4
    date               = '2023_06_13'
    dir_base           = 'D:/Hopkins/CroneLab2/Projects/CortiComm/Code/VideoInformation/'
    fps                = 60
    percent_pixels_thr = 0.005 
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

        # Defining the directory for the videos corresponding to the current date and block.
        dir_vid = dir_base + 'videos/' + date + '/Block' + str(block) + '/'

        # Defining the video file names for keyboard without the first three stimulus rows.
        filename_vid_keyboard_no_stim_rows = date + '_Speller_block' + str(block) + '_keyboard_no_stim_rows.mp4'

        # Defining the file path.
        path_vid_keyboard_no_stim_rows = dir_vid + filename_vid_keyboard_no_stim_rows
        
        # Extracting the total number of frames in the video. 
        # n_frames = count_frames(path_vid_keyboard_no_stim_rows)
        n_frames = 10000

        # Creating the array of timestamps with highlight detections from the keyboard without the stimulus rows. 
        all_keyboard_highlights = detect_keyboard_highlights(n_frames, path_vid_keyboard_no_stim_rows, percent_pixels_thr)

        # Saving the hand trajectories.
        dir_keyboard_highlights = dir_base + 'highlight_detections/' + date + '/'

        # Checking whether or not the directory for highlight detections exists. If it does not exist, it is created here.
        dir_keyboard_highlights_exists = os.path.exists(dir_keyboard_highlights)
        if not dir_keyboard_highlights_exists:
             os.makedirs(dir_keyboard_highlights)

        # Defining the file which hold the highlight detections. 
        file_highlight_detections = date + '_Block' + str(block) + '_highlight_detections'

        # Defining the file path for saving the highlight detections.
        path_keyboard_highlights = dir_keyboard_highlights + file_highlight_detections

        # Saving the array of highlight detections.
        np.save(path_keyboard_highlights, all_keyboard_highlights, allow_pickle=True)
