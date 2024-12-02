import nibabel as nib
import numpy as np
import pickle as pkl


def load_surf_gii(file_path):
    """Load a .surf.gii file."""
    return nib.load(file_path)


def create_metric_data(num_vertices):
    """Create some metric data for the vertices."""
    # Example: Create random metric data
    return np.random.rand(num_vertices)


def save_func_gii(file_path, metric_data, hemi):
    """Save the metric data as a .func.gii file."""
    # Create a new GIFTI data array with the metric data
    metadata = nib.gifti.GiftiMetaData()
    if hemi == "L":
        metadata["AnatomicalStructurePrimary"] = "CortexLeft"
    elif hemi == "R":
        metadata["AnatomicalStructurePrimary"] = "CortexRight"
    data_array = nib.gifti.GiftiDataArray(
        metric_data,
        intent="NIFTI_INTENT_SHAPE",
        datatype="NIFTI_TYPE_FLOAT32",
        meta=metadata,
    )

    # Create a new GIFTI image with the data array
    gifti_image = nib.gifti.GiftiImage(darrays=[data_array], meta=metadata)

    # Save the GIFTI image as a .func.gii file
    nib.save(gifti_image, file_path)


def process_hemispheres(diff_from_mean, base_path, output_prefix):
    for hemi in ["L", "R"]:
        # Load the .surf.gii file
        surf_file_path = f"{base_path}/fsLR-5k.{hemi}.surf.gii"
        surf_gii = load_surf_gii(surf_file_path)

        # Get the number of vertices from the loaded surface
        num_vertices = surf_gii.darrays[0].data.shape[0]

        # Create metric data
        if hemi == "L":
            metric_data = diff_from_mean[:num_vertices]
        elif hemi == "R":
            metric_data = diff_from_mean[-num_vertices:]

        # Save the metric data as a .func.gii file
        func_file_path = f"{output_prefix}_{hemi}.func.gii"
        save_func_gii(func_file_path, metric_data, hemi)

        print(f"Saved metric data to {func_file_path}")


if __name__ == "__main__":
    with open("outs.pkl", "rb") as f:
        diff_from_mean = pkl.load(f)
    base_path = "src/data"

    process_hemispheres(diff_from_mean, base_path, "difference_from_mean")

    with open("mean_conf_arr.pkl", "rb") as f:
        mean, confidence, std = pkl.load(f)

    process_hemispheres(mean, base_path, "mean_blur")
    process_hemispheres(std, base_path, "std_blur")

    with open("fullout.pkl", "rb") as f:
        fullout = pkl.load(f)

    process_hemispheres(fullout, base_path, "fullout")

    original_mean = diff_from_mean + mean
    confidence_interval_lower = confidence[:, 1]*1.6 < original_mean


    diff_from_mean[~confidence_interval_lower] = np.nan
    process_hemispheres(diff_from_mean, base_path, "difference_from_mean_confidence")

    print("done")