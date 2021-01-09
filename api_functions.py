import numpy as np
import pandas as pd
from pupil_functions import *
import platform
from shutil import copyfile
import matplotlib.pyplot as plt
from find_blob import *


#lag correcting
def correct_lag(trial_range, folder_path):
    #returns the average delay between when the experiment was sent and when the stimulus appears on the screen
    #trial_range = range of trials where annotations were added. ex: first 10 trials [1, 10]
    #folder_path = path to pupil player recording 
    col = ["block", "trial", "task", "sent", "received"]
    times = pd.DataFrame(columns = col)
    annotations = load_manual_annotations(folder_path)
    
    #change none's to nearest numbers
    for idx, num in annotations.iloc[:, 4].items():
        if num == None:
            point = idx
            while point >= 0:
                #finds the nearest filled cell that is before it
                point -= 1
                val = annotations.iloc[point, 4]
                if val is not None:
                    annotations.at[idx, "trial"] = val
                    break
    #first two must be manually done
    annotations.at[0, "trial"] = 1
    annotations.at[1, "trial"] = 1

    for idx, event in annotations.iterrows():
    #only look at trials in range
        if float(annotations.iloc[idx, 4]) > trial_range[1]:
            break
        if float(annotations.iloc[idx, 4]) < trial_range[0]:
            pass
        #fixation
        if annotations.iloc[idx, 2] == "Fixation dot":
            task = "Fixation dot"
            trial = annotations.iloc[idx, 4]
            block = annotations.iloc[idx, 0]
            sent = annotations.iloc[idx, 3]
            if annotations.iloc[idx + 1, 1] == "fixation":
                received = annotations.iloc[idx + 1, 3]
            else:
                received = annotations.iloc[idx - 1, 3]
            row = [block, trial, task, sent, received]
            times.loc[len(times), :] = row
        #image onset
        elif annotations.iloc[idx, 2] == "Image onset":
            task = "Image onset"
            trial = annotations.iloc[idx, 4]
            block = annotations.iloc[idx, 0]
            sent = annotations.iloc[idx, 3]
            if annotations.iloc[idx + 1, 1] == "image onset":
                received = annotations.iloc[idx + 1, 3]
            else:
                received = annotations.iloc[idx - 1, 3]        
            row = [block, trial, task, sent, received]
            times.loc[len(times), :] = row
        elif annotations.iloc[idx, 2] == "Adjust task":
            task = "Adjust task"
            trial = annotations.iloc[idx, 4]
            block = annotations.iloc[idx, 0]
            sent = annotations.iloc[idx, 3]
            if annotations.iloc[idx + 1, 1] == "adjust":
                received = annotations.iloc[idx + 1, 3]
            else:
                received = annotations.iloc[idx - 1, 3]
            row = [block, trial, task, sent, received]
            times.loc[len(times), :] = row
        else:
            pass
    times["delay"] = times["sent"] - times["received"]
    avg = times["delay"].mean()
    if avg < 0:
        times["fixedlag"] = times["received"] + avg #avg was negative
    elif avg > 0:
        times["fixedlag"] = times["received"] - avg
    
    return avg

def nearest_ts(time, ts_frame, task):
    #finds the nearest frame to a timestamp
    #time = corrected time after delay, ts_frame=DF with timestamps, task = 'begin' or 'end'
    timediff = []
    for idx, ts in ts_frame.iloc[:, 0].items():
        diff = ts-time
        if diff < 0: 
            timediff.append(diff)
        elif diff > 0:
            if task=='begin':
                best_ts = timediff[-1]
                best_frame = idx + 1
            elif task=='end':
                best_ts = ts
                best_frame = idx + 2
            break
        else: 
            best_frame = idx + 1
            break
    return best_frame


def frame_range_finder(folder_path, trial, block, avg):
    #finds the range of frames where the trial occurs
    #avg: average lag delay
    #define paths to csv files
    timestampframe = pd.read_csv(folder_path + "preprocessed/pl_timestamps.csv", header=None)
    messages = pd.read_csv(folder_path + "preprocessed/pl_msgs.csv")
    #find matching trial/block lines of csv files
    begintask = "Image onset"
    endtask = "Adjust task"
    msg_idx = messages[(messages['task'] == begintask) & (messages['trial'] == trial) & (messages['block'] == block)].index.values
    begin_ts = messages.loc[msg_idx[0], "timestamp"]
    msg_end_idx = messages[(messages['task'] == endtask) & (messages['trial'] == trial) & (messages['block'] == block)].index.values
    end_ts = messages.loc[msg_end_idx[0], 'timestamp']
    #correct the lag
    if avg > 0:
        begin_ts_lag = begin_ts - avg
        end_ts_lag = end_ts - avg
    elif avg < 0:
        begin_ts_lag = begin_ts + avg
        end_ts_lag = end_ts + avg
    #use nearest_ts to find best frame
    begin_frame = nearest_ts(begin_ts_lag, timestampframe, 'begin')
    end_frame = nearest_ts(end_ts_lag, timestampframe, 'end')
    return [begin_frame, end_frame]

def subset_frames(EM_path, frame_range, block, trial):
    #optional function that copies a trial's frames into a new folder
    #EM_path: path to folder with recording data
    #frame_range: range of frames for a specific trial
    #trial/block number
    subset_path = EM_path + 'block' + str(block) + 'trial' + str(trial) + 'frames'
    frames_path = EM_path + 'frames'
    try:
        if not os.path.exists(subset_path):
            os.mkdir(subset_path)
            print("Successfully created the directory %s " % subset_path)
        else:
            print("Directory already exists %s " % subset_path)
    except OSError:
        print("Creation of the directory %s failed" % subset_path)
    #copy files
    for i in range(frame_range[0], frame_range[1] + 1):
        if os.path.isfile(subset_path + str(i) + ".png"):
            pass
        else:
            copyfile(frames_path + "/frame" + str(i) + ".png", subset_path + "/frame" + str(i) + ".png")

#run find_blob on frames in subset
def blob_finder(bframe, eframe, trial, block, folder_path, coords_df, behavioral_path, tsframes):
    behavioral_df = pd.read_csv(behavioral_path)
    timestampframe = pd.read_csv(tsframes)
    row = behavioral_df[(behavioral_df["trialNumber"] == trial) & (behavioral_df['blockNumber'] == block)].index.values[0]
    degree = behavioral_df.loc[row, "stimLocationDeg"]
    correct_tags = [0, 1, 2, 3, 5, 6, 7, 8, 9, 11]
    #do for first 30 frames
    for i in range(bframe, bframe + 31): 
        blob = find_blob(folder_path + "frames/frame" + str(i) + ".png", correct_tags, degree, draw=True, show=True)
        tright= (blob[2][0], blob[1][1])
        bleft = (blob[1][0], blob[2][1])
        values = [trial, i, timestampframe.iloc[i+1, 0], blob[1], tright, bleft, blob[2], blob[0], blob[3]]
        coords_df.loc[len(coords_df), :] = values

def blob_coord_finder(folderpath, frame_range, trial, block, df, behavioral_1npath):
    blob_coords = pd.DataFrame(columns=["trial", "frame", "timestamp", "topleft", "topright", "bottomleft", "bottomright", "blobcenter", "screencenter"])
    correct_tags = [0, 1, 2, 3, 5, 6, 7, 8, 9, 11]
    time_frame = folderpath + 'preprocessed/pl_timestamps.csv'
    blob_finder(frame_range[0], frame_range[1], trial, block, folderpath, blob_coords, behavioral_1npath, time_frame)

