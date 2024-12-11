import numpy as np
from scipy import stats
import os
import nibabel as nib
import matplotlib.pyplot as plt


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

    L_intensities_array = np.zeros((4842, 12, num_sessions))
    L_distances_array = np.zeros((4842, 12, num_sessions))
    R_intensities_array = np.zeros((4842, 12, num_sessions))
    R_distances_array = np.zeros((4842, 12, num_sessions))

    e = 0
    for fold in os.listdir(datadir):
        sessions = os.listdir(os.path.join(datadir, fold))
        sessions = [x for x in sessions if ".gii" in x]
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
                    f"sub-{fold}_{session}_{hemi}_T1map_surf-fsnative_dist.func.gii",
                )
                intensities_path = os.path.join(
                    datadir,
                    fold,
                    f"sub-{fold}_{session}_{hemi}_T1map-surf-fsnative_NONgrad.func.gii",
                )

                distances = nib.load(distances_path).darrays[0].data
                intensities = nib.load(intensities_path).darrays[0].data

                distances = distances[:, :12]
                # distances = np.genfromtxt(
                #     distances_path, delimiter=",", skip_header=1
                # ).transpose()
                # intensities = np.genfromtxt(
                #     intensities_path, delimiter=",", skip_header=1
                # ).transpose()
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
        # # Generate x values for plotting the fitted curve
        # x_fit = np.linspace(vert_distances_single.min(), vert_distances_single.max(), 100)
        # y_fit = np.polyval(coeffs, x_fit)

        # # Plot the original data points
        # plt.scatter(vert_distances_single, vert_intensities_single, label='Data Points')

        # # Plot the fitted quadratic function
        # plt.plot(x_fit, y_fit, color='red', label='Fitted Quadratic Curve')

        # # Add labels and legend
        # plt.xlabel('Vertical Distances')
        # plt.ylabel('Vertical Intensities')
        # plt.legend()

        # # Show the plot
        # plt.show()
        highest_coeff = abs(coeffs[0])
        highest_coeff_values.append(highest_coeff)
    highest_coeff_values = np.array(highest_coeff_values)

    # sterr = stats.sem(highest_coeff_values[~np.isnan(highest_coeff_values)], ddof=degrees_freedom),
    # Calculate 95% confidence interval
    # Remove NaN values
    cleaned_data = highest_coeff_values[~np.isnan(highest_coeff_values)]

    # Calculate the sample mean
    mean = np.mean(cleaned_data)
    stddev = np.std(cleaned_data)
    # Calculate the standard error of the mean
    sterr = np.std(cleaned_data, ddof=1) / np.sqrt(len(cleaned_data))

    # Determine the critical value from the t-distribution
    confidence_level = 0.997
    degrees_freedom = len(cleaned_data) - 1
    critical_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)

    # Calculate the margin of error
    margin_of_error = critical_value * sterr

    # Determine the confidence interval
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    return mean, confidence_interval, stddev


def reshape_distances(distances_array):
    distances_array_reshaped = np.zeros(
        (
            len(distances_array),
            12,
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
