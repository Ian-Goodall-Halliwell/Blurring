import numpy as np
from scipy import stats
import os
import nibabel as nib


def load_data(datadir, hemis):

    # Count number of sessions
    num_sessions = 0
    for fold in os.listdir(datadir):
        sessions = os.listdir(os.path.join(datadir, fold))
        output_sessions = []
        for session in sessions:
            start = session.find(f"{fold}_") + len(f"{fold}_")
            end = session.find("_L") if "_L" in session else session.find("_R")
            session_number = session[start:end]
            output_sessions.append(session_number)
        sessions = list(set(output_sessions))
        num_sessions += len(sessions)

    L_intensities_array = np.zeros((4842, 16, num_sessions))
    L_distances_array = np.zeros((4842, 15, num_sessions))
    R_intensities_array = np.zeros((4842, 16, num_sessions))
    R_distances_array = np.zeros((4842, 15, num_sessions))

    e = 0
    for fold in os.listdir(datadir):
        sessions = os.listdir(os.path.join(datadir, fold))
        output_sessions = []
        for session in sessions:
            start = session.find(f"{fold}_") + len(f"{fold}_")
            end = session.find("_L") if "_L" in session else session.find("_R")
            session_number = session[start:end]
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

    intensities_array = np.concatenate(
        (L_intensities_array, R_intensities_array), axis=0
    )
    distances_array = np.concatenate((L_distances_array, R_distances_array), axis=0)

    return intensities_array, distances_array


def process_vertex(x, intensities_array, distances_array_reshaped, mask):
    vert_intensities = intensities_array[x, :, :]
    vert_distances = distances_array_reshaped[x, :, :]
    highest_coeff_values = []
    if mask[x]:
        highest_coeff_values.append(np.nan)
        return np.nan, (np.nan, np.nan), np.nan
    for z in range(vert_intensities.shape[1]):

        vert_distances_single = vert_distances[:, z]
        vert_intensities_single = vert_intensities[:, z]

        # Fit a quadratic function
        coeffs = np.polyfit(vert_distances_single, vert_intensities_single, 2)
        highest_coeff = abs(coeffs[0])
        highest_coeff_values.append(highest_coeff)
    highest_coeff_values = np.array(highest_coeff_values)
    # Calculate mean and standard deviation
    mean = np.mean(highest_coeff_values[~np.isnan(highest_coeff_values)])
    stddev = np.std(highest_coeff_values[~np.isnan(highest_coeff_values)])

    # Calculate 95% confidence interval
    confidence_level = 0.95
    degrees_freedom = len(highest_coeff_values[~np.isnan(highest_coeff_values)]) - 1
    confidence_interval = stats.t.interval(
        confidence_level,
        df=degrees_freedom,
        loc=mean,
        scale=stats.sem(highest_coeff_values[~np.isnan(highest_coeff_values)]),
    )
    return mean, confidence_interval, stddev


def reshape_distances(distances_array):
    distances_array_reshaped = np.zeros(
        (
            len(distances_array),
            16,
            distances_array.shape[2] if len(distances_array.shape) > 2 else 1,
        )
    )
    for en, x in enumerate(distances_array):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        for v in range(x.shape[1]):
            for e in range(x.shape[0]):
                if e < 4:
                    distances_array_reshaped[en, e, v] = -(np.sum(x[e:5, v]))
                elif e == 4:
                    distances_array_reshaped[en, e, v] = -x[e, v]
                    distances_array_reshaped[en, e + 1, v] = 0
                else:
                    distances_array_reshaped[en, e + 1, v] = (
                        distances_array_reshaped[en, e, v] + x[e, v]
                    )
    return distances_array_reshaped.squeeze()


def analyze_data(
    distances, intensities, mask, confidence_interval_array, mean_array, fullcort=False
):
    data_mask = np.zeros_like(mean_array)
    distances_reshaped = reshape_distances(distances)
    if not fullcort:
        intensities_full = intensities[~mask]
        distances_reshaped_full = distances_reshaped[~mask]
        confidence_interval_array_full = confidence_interval_array[~mask]
        mean_array_full = mean_array[~mask]
    else:
        intensities_full = intensities
        distances_reshaped_full = distances_reshaped
        confidence_interval_array_full = confidence_interval_array
        mean_array_full = mean_array

    outarray = np.zeros([intensities_full.shape[0]])
    for z in range(intensities_full.shape[0]):
        vert_distances_single = distances_reshaped_full[z, :]
        vert_intensities_single = intensities_full[z, :]

        # Fit a quadratic function
        coeffs = np.polyfit(vert_distances_single, vert_intensities_single, 2)
        highest_coeff = abs(coeffs[0])
        outarray[z] = highest_coeff

    e1 = 0
    outs_full = []
    for i in range(len(outarray)):
        # if outarray[i] < confidence_interval_array_full[i][0]:
        outs_full.append(mean_array_full[i] - outarray[i])
        e1 += 1
        print(f"Outlier at index {i} with value {outarray[i]}")
    if fullcort:
        outs_full, outarray = np.array(outs_full), np.array(outarray)
        outs_full[mask] = np.nan
        outarray[mask] = np.nan
        return outs_full, outarray
    return np.array(outs_full), np.array(outarray)
