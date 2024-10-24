from blurring import compute_blurring
import tempfile
import os

patients = {
    "PX001": [],  #
    # "PX004": [],
    # "PX005": [],
    "PX006": [],  #
    # "PX007": [],
    # "PX008": [],
    # "PX010": [],
    # "PX013": [],
    # "PX015": [],
    # "PX016": [],  #
    # "PX021": [],
    # "PX023": [],
    # "PX026": [],  #
    # "PX028": [],  #
    "PX029": [],  #
    # "PX044": [],
    # "PX047": [],
    "PX049": [],  #
    "PX051": [],  #
    # "PX060": [],  #
    "PX069": [],  #
    # "PX076": [],
}
# workingdir = "/host/verges/tank/data/ian/blur"
workingdir = "/data/mica3/BIDS_MICs/derivatives/zbrains_blur"
datadir = "/data/mica3/BIDS_MICs/derivatives"
micapipe = "micapipe_v0.2.0"
freesurfer = "freesurfer"
wb_path = "/usr/bin"
fs_path = "/data/mica1/01_programs/freesurfer-7.4.1/bin"
zbrainsname = "zbrains"
os.environ["FREESURFER_HOME"] = fs_path
os.environ["FS_LICENSE"] = "/home/bic/igoodall/Downloads/license.txt"
# Sort out and loop through paths

import os

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


with tempfile.TemporaryDirectory(dir=workingdir) as tmpdir:

    for patient in controls:
        os.makedirs(os.path.join(workingdir, patient), exist_ok=True)
        for path in controls[patient]:

            outputfile = compute_blurring(
                input_dir=os.path.join(
                    datadir, micapipe, path.split("_")[0], path.split("_")[1]
                ),
                surf_dir=os.path.join(datadir, freesurfer, path),
                bids_id=path,
                hemi="L",
                feat="T1map",
                workbench_path=wb_path,
                resol="5k",
                fwhm=5,
                tmp_dir=os.path.join(tmpdir),
                fs_path=fs_path,
            )
            os.rename(
                outputfile[0],
                os.path.join(
                    workingdir, patient, f"{path}_L_T1map_blur_NONgrad.func.gii"
                ),
            )
            os.rename(
                outputfile[1],
                os.path.join(
                    workingdir, patient, f"{path}_L_T1map_blur_intensities.csv"
                ),
            )
            os.rename(
                outputfile[2],
                os.path.join(workingdir, patient, f"{path}_L_T1map_blur_distances.csv"),
            )
            outputfile = compute_blurring(
                input_dir=os.path.join(
                    datadir, micapipe, path.split("_")[0], path.split("_")[1]
                ),
                surf_dir=os.path.join(datadir, freesurfer, path),
                bids_id=path,
                hemi="R",
                feat="T1map",
                workbench_path=wb_path,
                resol="5k",
                fwhm=5,
                tmp_dir=os.path.join(tmpdir),
                fs_path=fs_path,
            )
            os.rename(
                outputfile[0],
                os.path.join(
                    workingdir, patient, f"{path}_R_T1map_blur_NONgrad.func.gii"
                ),
            )
            os.rename(
                outputfile[1],
                os.path.join(
                    workingdir, patient, f"{path}_R_T1map_blur_intensities.csv"
                ),
            )
            os.rename(
                outputfile[2],
                os.path.join(workingdir, patient, f"{path}_R_T1map_blur_distances.csv"),
            )
            # compute_blurring(
            #     input_dir=os.path.join(datadir, micapipe, path.split("_")[0], path.split("_")[1]),
            #     surf_dir=os.path.join(datadir, freesurfer, path),
            #     bids_id=path,
            #     hemi="L",
            #     feat="T1map",
            #     workbench_path=wb_path,
            #     resol=1,
            #     fwhm=5,
            #     tmp_dir=os.path.join(tmpdir),
            #     fs_path=fs_path
            # )
            # compute_blurring(
            #     input_dir=os.path.join(datadir, micapipe, path),
            #     surf_dir=os.path.join(datadir, micapipe, path, "freesurfer"),
            #     bids_id=path,
            #     hemi="R",
            #     feat="T1map",
            #     workbench_path=wb_path,
            #     resol=1,
            #     fwhm=5,
            #     tmp_dir=os.path.join(tmpdir),
            #     fs_path=fs_path
            # )
print("e")
