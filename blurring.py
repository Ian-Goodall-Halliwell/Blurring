import nibabel as nib
import numpy as np
from scipy.stats import mode
import subprocess
import os
from sWM import laplace_solver, surface_generator
import scipy
import pandas as pd
import shutil
from utils import reshape_distances

def fixmatrix(path, inputmap, outputmap, basemap, BIDS_ID, temppath, wb_path, mat_path):
    # Load the .mat file
    mat = scipy.io.loadmat(
        os.path.join(
            path,
            "xfm",
            f"{BIDS_ID}_{mat_path}.mat",
        )
    )

    # Extract variables from the .mat file
    affine_transform = mat["AffineTransform_double_3_3"].flatten()
    fixed = mat["fixed"].flatten()

    temp = np.identity(4)
    for i in range(3):
        temp[i, 3] = affine_transform[9 + i] + fixed[i]
        for j in range(3):
            temp[i, j] = affine_transform[i * 3 + j]
            temp[i, 3] -= temp[i, j] * fixed[j]

    flips = np.identity(4)
    flips[0, 0] = -1
    flips[1, 1] = -1

    m_matrix = np.linalg.inv(flips @ temp @ flips)

    print(m_matrix)
    with open(
        os.path.join(temppath, f"{BIDS_ID}_real_world_affine.txt"),
        "w",
    ) as f:
        for row in m_matrix:
            f.write(" ".join(map(str, row)) + "\n")

    # command4 = [
    #     os.path.join(wb_path, "wb_command"),
    #     "-convert-affine",
    #     "-from-world",
    #     os.path.join(temppath, f"{BIDS_ID}_real_world_affine.txt"),
    #     "-to-world",
    #     "-inverse",
    #     os.path.join(temppath, f"{BIDS_ID}_real_world_affine.txt"),
    # ]

    # subprocess.run(command4)

    command3 = [
        os.path.join(wb_path, "wb_command"),
        "-volume-resample",
        inputmap,
        basemap,
        "ENCLOSING_VOXEL",
        outputmap,
        "-affine",
        os.path.join(temppath, f"{BIDS_ID}_real_world_affine.txt"),
    ]

    subprocess.run(command3)


def load_gifti_data(filepath):
    data = nib.load(filepath)
    return data.darrays[0].data


def calcdist(surf1, surf2):

    euclidianDistanceSquared = (surf1 - surf2) ** 2
    euclidianDistanceSummed = np.sum(euclidianDistanceSquared, axis=1)
    return np.sqrt(euclidianDistanceSummed)


def computegrad(data, dists):
    data = np.ediff1d(data)
    data[dists == 0] = 0
    dists[dists == 0] = 1
    return np.divide(data, dists)


def compute_blurring(
    input_dir,
    surf_dir,
    bids_id,
    hemi,
    feat,
    workbench_path,
    tmp_dir,
    fs_path,
    workingdir,
):

    base_path = input_dir
    for _ in range(4):  # Adjust the range to navigate up the desired number of levels
        base_path, _ = os.path.split(base_path)

    micapipe_path = os.path.split(input_dir)[0]

    freesurfer_path = os.path.join(
        base_path, "freesurfer", bids_id, "mri", "aparc+aseg.nii.gz"
    )

    temp_parc_path = os.path.join(
        tmp_dir, f"{bids_id}_{hemi}_surf-fsnative_label-temp.nii.gz"
    )
    print(temp_parc_path)
    output_path = os.path.join(workingdir, "swm", f"{bids_id}-laplace.nii.gz")

    stuctpath = os.path.join(workingdir, "structural")

    if not os.path.exists(stuctpath):
        os.mkdir(stuctpath)

    if not os.path.exists(freesurfer_path):
        subprocess.run(
            [
                os.path.join(fs_path, "mri_convert"),
                os.path.join(
                    micapipe_path,
                    f"{surf_dir}/mri/aparc+aseg.mgz",
                ),
                os.path.join(
                    tmp_dir, f"{bids_id}_{hemi}_surf-fsnative_label-temp-fixed.nii.gz"
                ),
            ]
        )
        freesurfer_path = os.path.join(
            tmp_dir, f"{bids_id}_{hemi}_surf-fsnative_label-temp-fixed.nii.gz"
        )
    if not os.path.exists(temp_parc_path):
        fixmatrix(
            path=input_dir,
            BIDS_ID=bids_id,
            temppath=tmp_dir,
            wb_path=workbench_path,
            inputmap=freesurfer_path,
            outputmap=temp_parc_path,
            basemap=f"{input_dir}/anat/{bids_id}_space-nativepro_T1w_brain.nii.gz",
            mat_path="from-fsnative_to_nativepro_T1w_0GenericAffine",
        )

        if not os.path.exists(os.path.join(workingdir, "swm")):
            os.mkdir(os.path.join(workingdir, "swm"))
        laplace_solver.solve_laplace(temp_parc_path, output_path)
        surface_generator.shift_surface(
            f"{input_dir}/surf/{bids_id}_hemi-{hemi}_space-nativepro_surf-fsnative_label-white.surf.gii",
            output_path,
            f"{workingdir}//swm//{bids_id}_{hemi}_sfwm-",
            [0.5, 1, 1.5, 2, 2.5, 3],
            n_jobs=32,
        )
    if feat.lower() == "t1map":
        volumemap = f"{input_dir}/anat/{bids_id}_space-nativepro_T1w_brain.nii.gz"
    elif feat.lower() != "adc" or feat.lower() != "fa":
        volumemap = f"{input_dir}/maps/{bids_id}_space-nativepro_map-{feat}.nii.gz"
    else:
        volumemap = f"{input_dir}/maps/{bids_id}_space-nativepro_model-DTI_map-{feat.upper()}.nii.gz"

    # pialDataArr = load_gifti_data(
    #     f"{input_dir}/maps/{bids_id}_hemi-{hemi}_surf-fsnative_label-pial_{feat}.func.gii"
    # )
    pialSurfaceArr = load_gifti_data(
        f"{input_dir}/surf/{bids_id}_hemi-{hemi}_space-nativepro_surf-fsnative_label-pial.surf.gii"
    )
    subprocess.run(
        [
            os.path.join(workbench_path, "wb_command"),
            "-volume-to-surface-mapping",
            volumemap,
            f"{input_dir}/surf/{bids_id}_hemi-{hemi}_space-nativepro_surf-fsnative_label-pial.surf.gii",
            f"{tmp_dir}/{bids_id}_hemi-{hemi}_space-nativepro_surf-fsnative_label-pial.func.gii",
            "-trilinear",
        ]
    )
    pialDataArr = load_gifti_data(
        f"{tmp_dir}/{bids_id}_hemi-{hemi}_space-nativepro_surf-fsnative_label-pial.func.gii"
    )
    wmBoundaryDataArr = load_gifti_data(
        f"{input_dir}/maps/{bids_id}_hemi-{hemi}_surf-fsnative_label-white_{feat}.func.gii"
    )
    wmBoundarySurfaceArr = load_gifti_data(
        f"{input_dir}/surf/{bids_id}_hemi-{hemi}_space-nativepro_surf-fsnative_label-white.surf.gii"
    )

    surfarr = [
        [pialDataArr, pialSurfaceArr],
    ]

    for ratio in [0.8, 0.6, 0.4, 0.2]:
        command_new = [
            os.path.join(workbench_path, "wb_command"),
            "-surface-cortex-layer",
            f"{input_dir}/surf/{bids_id}_hemi-{hemi}_space-nativepro_surf-fsnative_label-white.surf.gii",
            f"{input_dir}/surf/{bids_id}_hemi-{hemi}_space-nativepro_surf-fsnative_label-pial.surf.gii",
            str(ratio),
            f"{workingdir}//swm//{bids_id}_{hemi}_cortex-{ratio}.surf.gii",
        ]
        subprocess.run(command_new)
        subprocess.run(
            [
                os.path.join(workbench_path, "wb_command"),
                "-volume-to-surface-mapping",
                volumemap,
                f"{workingdir}//swm//{bids_id}_{hemi}_cortex-{ratio}.surf.gii",
                f"{workingdir}//swm//{bids_id}_{hemi}_{feat}_cortex-{ratio}_metric.func.gii",
                "-trilinear",
            ]
        )
        surfarr.append(
            [
                load_gifti_data(
                    f"{workingdir}//swm//{bids_id}_{hemi}_{feat}_cortex-{ratio}_metric.func.gii"
                ),
                load_gifti_data(
                    f"{workingdir}//swm//{bids_id}_{hemi}_cortex-{ratio}.surf.gii"
                ),
            ]
        )

    surfarr.append([wmBoundaryDataArr, wmBoundarySurfaceArr])

    for surf in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        subprocess.run(
            [
                os.path.join(workbench_path, "wb_command"),
                "-volume-to-surface-mapping",
                volumemap,
                f"{workingdir}//swm//{bids_id}_{hemi}_sfwm-{surf}mm.surf.gii",
                f"{workingdir}//swm//{bids_id}_{hemi}_{feat}_sfwm-{surf}mm_metric.func.gii",
                "-trilinear",
            ]
        )
        surfarr.append(
            [
                load_gifti_data(
                    f"{workingdir}//swm//{bids_id}_{hemi}_{feat}_sfwm-{surf}mm_metric.func.gii"
                ),
                load_gifti_data(
                    f"{workingdir}//swm//{bids_id}_{hemi}_sfwm-{surf}mm.surf.gii"
                ),
            ]
        )

    distances = np.zeros(shape=(len(pialDataArr), len(surfarr) - 1))
    dataArr = np.zeros(shape=(len(pialDataArr), len(surfarr)))
    dataArr_nonmode = np.zeros(shape=(len(pialDataArr), len(surfarr)), dtype=np.float32)
    for e, ds in enumerate(surfarr):
        data, surf = ds

        dataArr_nonmode[:, e] = data
        if e == len(surfarr) - 1:
            break
        nextdata, nextsurt = surfarr[e + 1]
        print(e)
        distance = calcdist(surf, nextsurt)
        distance = reshape_distances(distance)
        distances[:, e] = distance

    data_non_grad = nib.gifti.gifti.GiftiDataArray(
        data=dataArr_nonmode,
        intent="NIFTI_INTENT_NORMAL",
    )

    gii_non_grad = nib.gifti.GiftiImage(darrays=[data_non_grad])
    nib.save(
        gii_non_grad,
        os.path.join(
            workingdir,
            f"{bids_id}_{hemi}_{feat}-surf-fsnative_NONgrad.func.gii",
        ),
    )
    shutil.copy(
        os.path.join(
            input_dir,
            "surf",
            f"{bids_id}_hemi-{hemi}_surf-fsnative_label-sphere.surf.gii",
        ),
        os.path.join(
            stuctpath, f"{bids_id}_hemi-{hemi}_surf-fsnative_label-sphere.surf.gii"
        ),
    )
    data_dist = nib.gifti.gifti.GiftiDataArray(
        data=distances.astype(np.float32),
        intent="NIFTI_INTENT_NORMAL",
    )
    gii_dist = nib.gifti.GiftiImage(darrays=[data_dist])
    nib.save(
        gii_dist,
        os.path.join(
            workingdir,
            f"{bids_id}_{hemi}_{feat}_surf-fsnative_dist.func.gii",
        ),
    )


if __name__ == "__main__":
    sub = "sub-PX103"
    ses = "ses-01"
    surface = "fsnative"
    micapipe = "micapipe"
    hemi = "L"
    input_dir = f"E:/data/derivatives/{micapipe}/{sub}/{ses}/maps/"
    surf_dir = f"E:/data/derivatives/{micapipe}/{sub}/{ses}/surf/"
    output_dir = "."
    bids_id = f"{sub}_{ses}"
    compute_blurring(
        input_dir,
        surf_dir,
        bids_id,
        hemi,
        f"{output_dir}/{bids_id}_hemi-{hemi}_blurring.func.gii",
    )
