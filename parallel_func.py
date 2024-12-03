import os
import tempfile
import argparse
from blurring import compute_blurring


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
        if not os.path.exists(os.path.join(workingdir, patient)):
            os.mkdir(os.path.join(workingdir, patient))
        with tempfile.TemporaryDirectory(
            dir=os.path.join(workingdir, patient)
        ) as tmpdir:
            compute_blurring(
                input_dir=os.path.join(
                    datadir, micapipe, path.split("_")[0], path.split("_")[1]
                ),
                surf_dir=os.path.join(datadir, freesurfer, path),
                bids_id=path,
                hemi="L",
                feat="T1map",
                workbench_path=wb_path,
                tmp_dir=tmpdir,
                fs_path=fs_path,
                workingdir=os.path.join(workingdir, patient),
            )
            compute_blurring(
                input_dir=os.path.join(
                    datadir, micapipe, path.split("_")[0], path.split("_")[1]
                ),
                surf_dir=os.path.join(datadir, freesurfer, path),
                bids_id=path,
                hemi="R",
                feat="T1map",
                workbench_path=wb_path,
                tmp_dir=tmpdir,
                fs_path=fs_path,
                workingdir=os.path.join(workingdir, patient),
            )

    except Exception as e:
        print(e)
        print(f"Error with {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process path for blurring.")
    parser.add_argument("patient", type=str, help="Patient identifier")
    parser.add_argument("path", type=str, help="Path to the data")
    parser.add_argument("workingdir", type=str, help="Working directory")
    parser.add_argument("datadir", type=str, help="Data directory")
    parser.add_argument("micapipe", type=str, help="Micapipe version")
    parser.add_argument("freesurfer", type=str, help="Freesurfer directory")
    parser.add_argument("wb_path", type=str, help="Workbench path")
    parser.add_argument("fs_path", type=str, help="Freesurfer path")
    parser.add_argument(
        "current_file_directory", type=str, help="Current file directory"
    )

    args = parser.parse_args()

    process_path(
        args.patient,
        args.path,
        args.workingdir,
        args.datadir,
        args.micapipe,
        args.freesurfer,
        args.wb_path,
        args.fs_path,
        args.current_file_directory,
    )
