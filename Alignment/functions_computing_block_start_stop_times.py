
# IMPORTING LIBRARIES
import numpy as np





def extract_start_stop_times_from_txt(block_id, date, dir_intermediates, fps, patient_id):
    """
    DESCRIPTION:
    Importing the starting an ending points of the video that show only the truly relevant movement information for the
    experimenter-specified spelling block. By looking at the video recording from that block, these starting and ending times 
    are determined according to when the participant starts and ends making task-related hand movements.
    
    INPUT VARIABLES:
    block_id:          [String (BlockX, where X is an int)]; Block ID of the task that was run.
    date:              [string (YYYY_MM_DD)]; Date on which the current block was run.
    dir_intermediates: [string]; [string]; Intermediates directory where relevant information is stored.
    fps:               [int (30 or 60)]; Frames per second of of the video feed. Note that 30 FPS was from Aug 2022 - Jan 2023.
    patient_id:        [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    
    OUTPUT VARIABLES:
    frame_block_start: [int]; Starting video frame of the task-related activity. 
    frame_block_end:   [int]; Ending video frame of the task-related activity. 
    t_block_start:     [int (units: s)]; Starting time of the task-related activity. 
    t_block_end:       [int (units: s)]; Ending time of the task-related activity. 
    """
    # COMPUTATION:
    
    # Creating the pathway for the .txt file from which the starting and ending points of the current block are imported.
    this_directory = dir_intermediates + patient_id + '/Speller/BlocksStartAndStops/' + date + '/' 
    this_filename  = date + '_' + block_id + '_StartStop.txt'
    path_startstop = this_directory + this_filename
    
    # Opening up the start-stop text file from the pathway and reading contents. For an un-appended .txt file, only the first
    # three lines (Lines 0-2) should have information.
    txt_file_start_stop = open(path_startstop)        
    text_file_lines = txt_file_start_stop.readlines()

    # Reading in the strings with the starting and stopping times.
    line_time_start = text_file_lines[1]
    line_time_stop  = text_file_lines[2]

    # Extracting the start and stop strings from the corresponding lines.
    time_start = line_time_start[7:18] 
    time_end   = line_time_stop[7:18] 

    # Extracting the hours, minutes, seconds and frames information. 
    [hour_start, min_start, sec_start, frame_start] = time_start.split(':')
    [hour_end, min_end, sec_end, frame_end]         = time_end.split(':')

    # Converting this time information to integers.
    hour_start = int(hour_start)
    min_start  = int(min_start)
    sec_start  = int(sec_start)
    fr_start   = int(frame_start)
    hour_end   = int(hour_end)
    min_end    = int(min_end)
    sec_end    = int(sec_end)
    fr_end     = int(frame_end)

    # Converting the times to frames.
    hours_to_frames = lambda dd: dd * 24 * 60 * 60 * fps # hours x 24 hrs/hour x 60 min/hr x 60 s/min x 30 frames/s = frames
    mins_to_frames  = lambda mm: mm * 60 * fps           # minutes x 60 s/min x 30 frames/s = frames
    secs_to_frames  = lambda ss: ss * fps                # seconds x 30 frames/s = frames

    # Computing the starting and ending frames of the block.
    frame_block_start = hours_to_frames(hour_start) + mins_to_frames(min_start) + secs_to_frames(sec_start) + fr_start
    frame_block_end   = hours_to_frames(hour_end) + mins_to_frames(min_end) + secs_to_frames(sec_end) + fr_end

    # Computing the starting and ending times of the block.
    t_block_start = np.round(frame_block_start/fps,3)
    t_block_end   = np.round(frame_block_end/fps,3)
    
    # Print total block time.
    t_total = t_block_end - t_block_start
    print('Total block time (s): ', round(t_total, 3))

    return frame_block_start, frame_block_end, t_block_start, t_block_end





def writing_to_text_file(block_id, date, dir_intermediates, frame_block_start, frame_block_end, patient_id, t_block_start, t_block_end):
    """
    DESCRIPTION:
    Writing the block start and stop times in units of seconds and frames back to the text file.
    
    INPUT VARIABLES:
    block_id:          [String (BlockX, where X is an int)]; Block ID of the task that was run.
    date:              [string (YYYY_MM_DD)]; Date on which the current block was run.
    dir_intermediates: [string]; [string]; Intermediates directory where relevant information is stored.
    frame_block_start: [int]; Starting video frame of the task-related activity. 
    frame_block_end:   [int]; Ending video frame of the task-related activity. 
    patient_id:        [string]; Patient ID PYyyNnn or CCxx format, where y, n, and x are integers.
    t_block_start:     [int (units: s)]; Starting time of the task-related activity. 
    t_block_end:       [int (units: s)]; Ending time of the task-related activity. 
    """
    
    # COMPUTATION:
    
    # Creating the pathway for the .txt file where the starting and ending times of the current block are stored.
    this_directory = dir_intermediates + patient_id + '/Speller/BlocksStartAndStops/' + date + '/' 
    this_filename  = date + '_' + block_id + '_StartStop.txt'
    path_startstop = this_directory + this_filename

    # Opening up the start-stop text file from the pathway and reading contents. For an un-appended .txt file, only the first
    # three lines (Lines 0-2) should have information.
    txt_file_start_stop = open(path_startstop)        
    text_file_lines = txt_file_start_stop.readlines()

    # Updating the list of text file lines with the start and stop times in units of seconds and frames.
    text_file_lines[5]  = 'start: ' + str(t_block_start) + '\n'
    text_file_lines[6]  = 'stop:  ' + str(t_block_end)  + '\n'
    text_file_lines[9]  = 'start: ' + str(frame_block_start) + '\n'
    text_file_lines[10] = 'stop:  ' + str(frame_block_end)  + '\n'

    # Writing to text file.
    with open(path_startstop, 'w') as f:
        for line in text_file_lines:
            f.write(f"{line}")