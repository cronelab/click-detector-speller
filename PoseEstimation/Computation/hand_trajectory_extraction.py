



# IMPORTING LIBRARIES
import copy
import cv2
import itertools
import mediapipe as mp
import numpy as np
import os
import time
import xarray as xr # DELETE IF NECESSARY

from tqdm import trange

# README:
# This must be run from Roach (in VidRecog2 environment). Uploaded to Zappa only as a backup.

def extract_hand_trajectories(fps, n_frames, n_landmarks, path_vid):
    """
    DESCRIPTION:
    The landmarks of each frame are recorded and saved in the hand_trajectories array. 

    INPUT VARIABLES:
    fps:         [int]; Frames per second of the video camera. # DELETE IF NECESSARY
    n_frames:    [int]; Total number of frames in the video. 
    n_landmarks: [int]; Total number of hand landmarks (x- and y- coordinates).
    path_vid:    [string]; Pathway to the video.

    OUTPUT VARIABLES:
    hand_trajectories: [array (landmarks x frames) > floats]; The landmarks (x- and y- coordinates) at each frame. 
    """

    # COMPUTATION: 

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

    # Uploading the file for CV video capture.
    cap = cv2.VideoCapture(path_vid)

    # Initializing the array which will hold the trajectories of the landmarks across all the frames of the video.
    hand_trajectories = np.zeros((n_landmarks, n_frames))

    # Iterating across all channels.
    for frame_idx in trange(n_frames, position=0, desc='Iterating across all video frames.'):

        # Reading each frame from the Camera capture 
        ret, image = cap.read()

        # Copying the current frame over for pasting the hand coordinates.
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Determining if the CV algorithm can detect hand landmarks on the screen.
        results                  = hands.process(image)
        hand_landmarks_on_screen = results.multi_hand_landmarks

        # Extracting the new hand landmarks for the current frame. Only if the hand(s) is on the screen. 
        if hand_landmarks_on_screen is not None:
            for hand_landmarks in hand_landmarks_on_screen:

                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Drawing the bounding rectangle and the landmarks on the current frame.
                debug_image = draw_landmarks(debug_image, landmark_list)

        # If the hand landmarks are not detected on screen, use the previous landmarks as they should be good estimations.
        else: 
             # Landmark calculation
             landmark_list = calc_landmark_list(debug_image, hand_landmarks)

             # Conversion to relative coordinates / normalized coordinates
             pre_processed_landmark_list = pre_process_landmark(landmark_list)

        # Converting landmark list to array.
        landmarks_new = np.array(pre_processed_landmark_list)

        # Assigning the new landmarks to the array of hand trajectories.
        hand_trajectories[:,frame_idx] = landmarks_new

        # Updating the frame index.
        frame_idx += 1

        # Showing the hand with the pose-estimated coordinates.
        cv2.imshow('Hand Gesture Recognition', debug_image)

        if cv2.waitKey(10) == ord('q'):
            break

    ####################################################################
    # DELETE IF NECESSARY

    # Computing the array of seconds.
    time_seconds = np.round(np.arange(n_frames)/fps, 3)   # frames x sec/frame = sec
    
    mp_hand_landmarks = ['WRIST_x', 'WRIST_y', 'THUMB_CMC_x', 'THUMB_CMC_y', 'THUMB_MCP_x', 'THUMB_MCP_y', 'THUMB_IP_x',\
                         'THUMB_IP_y', 'THUMB_TIP_x', 'THUMB_TIP_y', 'INDEX_FINGER_MCP_x', 'INDEX_FINGER_MCP_y', 'INDEX_FINGER_PIP_x',\
                         'INDEX_FINGER_PIP_y', 'INDEX_FINGER_DIP_x', 'INDEX_FINGER_DIP_y', 'INDEX_FINGER_TIP_x', 'INDEX_FINGER_TIP_y',\
                         'MIDDLE_FINGER_MCP_x', 'MIDDLE_FINGER_MCP_y', 'MIDDLE_FINGER_PIP_x', 'MIDDLE_FINGER_PIP_y', 'MIDDLE_FINGER_DIP_x',\
                         'MIDDLE_FINGER_DIP_y', 'MIDDLE_FINGER_TIP_x', 'MIDDLE_FINGER_TIP_y', 'RING_FINGER_MCP_x', 'RING_FINGER_MCP_y',\
                         'RING_FINGER_PIP_x', 'RING_FINGER_PIP_y', 'RING_FINGER_DIP_x', 'RING_FINGER_DIP_y', 'RING_FINGER_TIP_x',\
                         'RING_FINGER_TIP_y', 'PINKY_MCP_x', 'PINKY_MCP_y', 'PINKY_PIP_x', 'PINKY_PIP_y', 'PINKY_DIP_x', 'PINKY_DIP_y',\
                         'PINKY_TIP_x', 'PINKY_TIP_y']

    # Converting the array of landmarks into an xarray.
    hand_trajectories = xr.DataArray(hand_trajectories,
                                     coords={'landmarks': mp_hand_landmarks, 'time_seconds': time_seconds},
                                     dims=["landmarks", "time_seconds"])

    ####################################################################
     

    # Release the camera and close the camera window.
    cap.release()
    cv2.destroyAllWindows()

    return hand_trajectories


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


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
    While using the speller application, the participant makes a grasp with his right hand which is video-recorded. The video of the hand is
    post-processed from a larger video which also contains the keyboard, participant's hand, and the main screen of the testing machine. The 
    coordinates of each landmark (x- and y-coordinates of the hand joints) are extracted at each frame (using pose-estimation), yielding a 
    time-domain trajectories across the duration of the spelling block. These are then saved in arrays. 
    """
    # EXPERIMENTER INPUTS
    block_start        = 1
    block_end          = 2
    date               = '2022_11_08'
    dir_base           = 'D:/Hopkins/CroneLab2/Projects/CortiComm/Code/VideoInformation/Speller/'
    fps                = 30 
    n_landmarks        = 42
   
    """
    INPUT VARIABLES:
    block_end:   [int]; The number of the last block from the date.
    block_start: [int]; The number of the first block from the date.
    date:        [string]; The date ('yyyy_mm_dd') of the video from which clicks will be extracted.
    dir_base:    [string]; The directory which holds the video for extracting the hand landmark coordinates.
    fps:         [int]; Frames per second of the video camera.
    n_landmarks: [int]; Total number of hand landmarks (x- and y- coordinates).
    """

    # COMPUTATION:

    # Iterating across each experimenter-input block to extract hand landmark coordinates.
    for block in range(block_start,block_end+1):
    
        print('Block: ', block)
            
        # Defining the directory for the video corresponding to the current date and block.
        dir_vid  = dir_base + 'videos/' + date + '/Block' + str(block) + '/'

        # Defining the video file names for hand movement.
        filename_hand = date + '_Speller_block' + str(block) + '_hand.mp4'

        # Defining the pathway from which to extract the hand video movement file.
        path_vid_hand = dir_vid + filename_hand

        # Extracting the total number of frames in the video.
        n_frames = count_frames(path_vid_hand)
        # n_frames = 500

        # Computing the hand trajectories.
        # hand_trajectories = extract_hand_trajectories(n_frames, n_landmarks, path_vid_hand) 
        hand_trajectories = extract_hand_trajectories(fps, n_frames, n_landmarks, path_vid_hand) # DELETE IF NECESSARY

        # Saving the hand trajectories.
        # dir_hand_trajectories = dir_base + 'hand_trajectories/' + date + '/' 
        dir_hand_trajectories = dir_base + 'hand_trajectories/' + date + '/Xarrays/' # DELETE IF NECESSARY

        # Checking whether or not the directory for hand trajectories exists. If it does not exist, it is created here.
        dir_hand_trajectories_exists = os.path.exists(dir_hand_trajectories)
        if not dir_hand_trajectories_exists:
             os.makedirs(dir_hand_trajectories)

        # Defining the file which hold the hand trajectories. 
        # file_hand_trajectories = date + '_Block' + str(block) + '_hand_trajectories' 
        file_hand_trajectories = date + '_Block' + str(block) + '_hand_trajectories.nc' # DELETE IF NECESSARY

        # Defining the file paths for saving the hand trajectories.
        path_hand_trajectories = dir_hand_trajectories + file_hand_trajectories

        # Saving the array of pose-estimation coordinates.
        # np.save(path_hand_trajectories, hand_trajectories, allow_pickle=True) 
        hand_trajectories.to_netcdf(path_hand_trajectories) # DELETE IF NECESSARY
