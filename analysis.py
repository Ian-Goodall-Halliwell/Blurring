import pandas as pd
import os
import numpy as np


def delete_empty_folder(folder_path):
    # Check if the folder is empty
    if not os.listdir(folder_path):
        # Delete the folder if it is empty
        os.rmdir(folder_path)
        print(f"Deleted empty folder: {folder_path}")
    else:
        print(f"Folder is not empty: {folder_path}")


hemis = ["L", "R"]
datadir = "E:\data\derivatives\zblur"
for fold in os.listdir(datadir):
    delete_empty_folder(os.path.join(datadir, fold))

totalsessions = []
for fold in os.listdir(datadir):
    sessions = os.listdir(os.path.join(datadir, fold))
    output_sessions = []
    for session in sessions:
        start = session.find(f"{fold}_") + len(f"{fold}_")
        end = session.find("_L") if "_L" in session else session.find("_R")
        session_number = session[start:end]
        print(f"Session: {session}, Extracted Substring: {session_number}")
        output_sessions.append(session_number)
    sessions = list(set(output_sessions))
    for ses in sessions:
        totalsessions.append(ses)


totalsubs = len(totalsessions)
L_intensities_array = np.zeros((4842, 12, totalsubs))
L_distances_array = np.zeros((4842, 11, totalsubs))

R_intensities_array = np.zeros((4842, 12, totalsubs))
R_distances_array = np.zeros((4842, 11, totalsubs))

e = 0
for fold in os.listdir(datadir):
    sessions = os.listdir(os.path.join(datadir, fold))
    output_sessions = []
    for session in sessions:
        start = session.find(f"{fold}_") + len(f"{fold}_")
        end = session.find("_L") if "_L" in session else session.find("_R")
        session_number = session[start:end]
        print(f"Session: {session}, Extracted Substring: {session_number}")
        output_sessions.append(session_number)
    sessions = list(set(output_sessions))
    for session in sessions:
        for hemi in hemis:
            distances_path = os.path.join(
                datadir,
                fold,
                f"sub-{fold}_{session}_{hemi}_T1map_blur_distances.csv",
            )
            intensities_path = os.path.join(
                datadir,
                fold,
                f"sub-{fold}_{session}_{hemi}_T1map_blur_intensities.csv",
            )
            distances = np.genfromtxt(
                distances_path, delimiter=",", skip_header=1
            ).transpose()
            intensities = np.genfromtxt(
                intensities_path, delimiter=",", skip_header=1
            ).transpose()
            if hemi == "L":
                L_intensities_array[:, :, int(e)] = intensities
                L_distances_array[:, :, int(e)] = distances
            else:
                R_intensities_array[:, :, int(e)] = intensities
                R_distances_array[:, :, int(e)] = distances
        e += 1

print("e")
