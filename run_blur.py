from blurring import compute_blurring
import tempfile
import os
import sys
from joblib import Parallel, delayed
import subprocess

patients = {
    "PX001": [],  #
    # # "PX004": [],
    # # "PX005": [],
    # "PX006": [],  #
    # # "PX007": [],
    # # "PX008": [],
    # # "PX010": [],
    # # "PX013": [],
    # # "PX015": [],
    # # "PX016": [],  #
    # # "PX021": [],
    # # "PX023": [],
    # # "PX026": [],  #
    # # "PX028": [],  #
    "PX029": [],  #
    # # "PX044": [],
    # # "PX047": [],
    # "PX049": [],  #
    # "PX051": [],  #
    # # "PX060": [],  #
    "PX069": [],  #
    # # "PX076": [],
}
workingdir = "/data/mica3/BIDS_MICs/derivatives/zbrains_blur"
datadir = "/data/mica3/BIDS_MICs/derivatives"
micapipe = "micapipe_v0.2.0"
freesurfer = "freesurfer"
wb_path = "/usr/bin"
fs_path = "/data/mica1/01_programs/freesurfer-7.4.1/bin"
zbrainsname = "zbrains"
os.environ["FREESURFER_HOME"] = fs_path
os.environ["FS_LICENSE"] = "/home/bic/igoodall/Downloads/license.txt"
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Sort out and loop through paths

import os

if not os.path.exists(workingdir):
    os.mkdir(workingdir)


controls = {}

for path in os.listdir(os.path.join(datadir, freesurfer)):
    if "HC" in path:
        id = path.split("_")[0].split("-")[1]
        if id not in controls.keys():
            controls[id] = []
        controls[id].append(path)
    else:
        for px in patients:
            if px in path:
                patients[px].append(path)

# Convert the dictionary to a list of lists and sort each sublist
patients = {patient: sorted(patients[patient]) for patient in patients}
controls = {control: sorted(controls[control]) for control in controls}


def process_path(
    patient,
    path,
    workingdir,
    datadir,
    micapipe,
    freesurfer,
    wb_path,
    fs_path,
    current_file_directory,
):
    try:
        subprocess.run(
            [
                sys.executable,  # Use the current Python interpreter
                os.path.join(
                    current_file_directory, "parallel_func.py"
                ),  # The script to run
                patient,
                path,
                workingdir,
                datadir,
                micapipe,
                freesurfer,
                wb_path,
                fs_path,
                current_file_directory,
            ],
            check=True,
        )

    except Exception as e:
        print(e)
        print(f"Error with {path}")


# for patient in controls:
#     for path in controls[patient]:
#         process_path(
#             patient,
#             path,
#             workingdir,
#             datadir,
#             micapipe,
#             freesurfer,
#             wb_path,
#             fs_path,
#             current_file_directory,
#         )


Parallel(n_jobs=4)(
    delayed(process_path)(
        patient,
        path,
        workingdir,
        datadir,
        micapipe,
        freesurfer,
        wb_path,
        fs_path,
        current_file_directory,
    )
    for patient in controls
    for path in controls[patient]
)
