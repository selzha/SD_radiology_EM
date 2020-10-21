import numpy as np
import pandas as pd
from pupil_functions import load_manual_annotations
import os

#need to manually annotate the video first with "fixation", "adjust" and "image onset"

def find_delay(path, trials_annotated):
    col = ["block", "trial", "task", "sent", "received"]
    times = pd.DataFrame(columns = col)
    df = load_manual_annotations(path)
    #change none's to nearest numbers
    for idx, num in df.iloc[:, 4].items():
        if num == None:
            point = idx
            while point >= 0:
            #finds the nearest filled cell that is before it
                point -= 1
                val = df.iloc[point, 4]
                if val is not None:
                    df.at[idx, "trial"] = val
                    break
    #first two must be manually done
    df.at[0, "trial"] = 1
    df.at[1, "trial"] = 1

    for idx, event in df.iterrows():
        #first x trials
        if float(df.iloc[idx, 4]) > trials_annotated:
            break
        #fixation
        if df.iloc[idx, 2] == "Fixation dot":
            task = "Fixation dot"
            trial = df.iloc[idx, 4]
            block = df.iloc[idx, 0]
            sent = df.iloc[idx, 3]
            if df.iloc[idx + 1, 1] == "fixation":
                received = df.iloc[idx + 1, 3]
            else:
                received = df.iloc[idx - 1, 3]
                row = [block, trial, task, sent, received]
                times.loc[len(times), :] = row
        #image onset
        elif df.iloc[idx, 2] == "Image onset":
            task = "Image onset"
            trial = df.iloc[idx, 4]
            block = df.iloc[idx, 0]
            sent = df.iloc[idx, 3]
            if df.iloc[idx + 1, 1] == "image onset":
                received = df.iloc[idx + 1, 3]
            else:
                received = df.iloc[idx - 1, 3]        
                row = [block, trial, task, sent, received]
                times.loc[len(times), :] = row
        elif df.iloc[idx, 2] == "Adjust task":
            task = "Adjust task"
            trial = df.iloc[idx, 4]
            block = df.iloc[idx, 0]
            sent = df.iloc[idx, 3]
            if df.iloc[idx + 1, 1] == "adjust":
                received = df.iloc[idx + 1, 3]
            else:
                received = df.iloc[idx - 1, 3]
                row = [block, trial, task, sent, received]
                times.loc[len(times), :] = row
        else:
            pass
    #mean delay
    times["delay"] = times["sent"] - times["received"]
    avg = times["delay"].mean()
    #fixlag
    if avg < 0:
        times["fixedlag"] = times["received"] + avg #avg was negative
    elif avg > 0:
        times["fixedlag"] = times["received"] - avg
    return avg
