import pandas as pd
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import scipy.stats as stats
from joblib import Parallel, delayed


def delete_empty_folder(folder_path):
    if not os.listdir(folder_path):
        os.rmdir(folder_path)
        print(f"Deleted empty folder: {folder_path}")
    else:
        print(f"Folder is not empty: {folder_path}")


def main():
    hemis = ["L", "R"]
    datadir = "E:/data/derivatives/zbrains_blur"

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
    L_intensities_array = np.zeros((4842, 16, totalsubs))
    L_distances_array = np.zeros((4842, 15, totalsubs))
    R_intensities_array = np.zeros((4842, 16, totalsubs))
    R_distances_array = np.zeros((4842, 15, totalsubs))

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

    intensities_array = np.concatenate(
        (L_intensities_array, R_intensities_array), axis=0
    )
    distances_array = np.concatenate((L_distances_array, R_distances_array), axis=0)

    distances_array_reshaped = np.zeros((len(distances_array), 16, totalsubs))
    for en, x in enumerate(distances_array):
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
        # print(x)

    with open("output.pkl", "wb") as f:
        pickle.dump((intensities_array, distances_array_reshaped), f)

    return intensities_array, distances_array_reshaped


if not os.path.exists("output.pkl"):
    intensities_array, distances_array_reshaped = main()
else:
    with open("output.pkl", "rb") as f:
        intensities_array, distances_array_reshaped = pickle.load(f)


intensities_array = intensities_array[:, :-5, :]
distances_array_reshaped = distances_array_reshaped[:, :-5, :]

mean_array = np.zeros([distances_array_reshaped.shape[0]])
confidence_interval_array = np.zeros([distances_array_reshaped.shape[0], 2])


def process_vertex(x):
    vert_intensities = intensities_array[x, :, :]
    vert_distances = distances_array_reshaped[x, :, :]
    highest_coeff_values = []
    for z in range(vert_intensities.shape[1]):
        vert_distances_single = vert_distances[:, z]
        vert_intensities_single = vert_intensities[:, z]

        # Fit a quadratic function
        coeffs = np.polyfit(vert_distances_single, vert_intensities_single, 2)
        highest_coeff = abs(coeffs[0])
        highest_coeff_values.append(highest_coeff)

    # Calculate mean and standard deviation
    mean = np.mean(highest_coeff_values)
    stddev = np.std(highest_coeff_values)

    # Calculate 95% confidence interval
    confidence_level = 0.95
    degrees_freedom = len(highest_coeff_values) - 1
    confidence_interval = stats.t.interval(
        confidence_level,
        df=degrees_freedom,
        loc=mean,
        scale=stats.sem(highest_coeff_values),
    )
    return mean, confidence_interval


from tqdm import tqdm

# Parallelize the computation
results = Parallel(n_jobs=-1)(
    delayed(process_vertex)(x) for x in tqdm(range(intensities_array.shape[0]))
)

# Store the results
for x, (mean, confidence_interval) in enumerate(results):
    mean_array[x] = mean
    confidence_interval_array[x] = confidence_interval

print(f"Mean array: {mean_array}")
print(f"Confidence interval array: {confidence_interval_array}")

# Save variables to a file
with open("output.pkl", "wb") as f:
    pickle.dump((mean_array, confidence_interval_array), f)


### LOAD PATIENT DATA

# Load L and R versions of distances and intensities
L_distances = np.genfromtxt(
    "E:/data/derivatives/zbrains_blur/PX001/sub-PX001_ses-01_L_T1map_blur_distances.csv",
    delimiter=",",
    skip_header=1,
).transpose()
L_intensities = np.genfromtxt(
    "E:/data/derivatives/zbrains_blur/PX001/sub-PX001_ses-01_L_T1map_blur_intensities.csv",
    delimiter=",",
    skip_header=1,
).transpose()

R_distances = np.genfromtxt(
    "E:/data/derivatives/zbrains_blur/PX001/sub-PX001_ses-01_R_T1map_blur_distances.csv",
    delimiter=",",
    skip_header=1,
).transpose()
R_intensities = np.genfromtxt(
    "E:/data/derivatives/zbrains_blur/PX001/sub-PX001_ses-01_R_T1map_blur_intensities.csv",
    delimiter=",",
    skip_header=1,
).transpose()

# Concatenate L and R versions
distances = np.concatenate((L_distances, R_distances), axis=0)
intensities = np.concatenate((L_intensities, R_intensities), axis=0)

# Load mask
mask_R = np.genfromtxt(
    "C:/Users/Ian/Documents/GitHub/Blurring/output/sub-PX001_ses-02_space-nativepro_T1w_brainthresholdedSurface.csv",
    delimiter=",",
    skip_header=1,
).transpose()

mask_L = np.zeros([R_intensities.shape[0]])


# Convert mask to boolean array
mask_R = mask_R.astype(bool)
mask_L = mask_L.astype(bool)

mask = np.concatenate((mask_L, mask_R), axis=0)


def reshape_distances(distances):
    RDavgacrosstrial_reshaped = np.zeros((len(distances), 16))
    for en, x in enumerate(distances):
        for e in range(x.shape[0]):
            if e < 4:
                RDavgacrosstrial_reshaped[en, e] = -(np.sum(x[e:5]))
            elif e == 4:
                RDavgacrosstrial_reshaped[en, e] = -x[e]
                RDavgacrosstrial_reshaped[en, e + 1] = 0
            else:
                RDavgacrosstrial_reshaped[en, e + 1] = (
                    RDavgacrosstrial_reshaped[en, e] + x[e]
                )
    return RDavgacrosstrial_reshaped


distances_reshaped = reshape_distances(distances)

intensities_full = intensities[~mask]


distances_reshaped_full = distances_reshaped[~mask]

confidence_interval_array_full = confidence_interval_array[~mask]

mean_array_full = mean_array[~mask]

outarray = np.zeros([intensities_full.shape[0]])
# Fit a quadratic function
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
    if outarray[i] < confidence_interval_array_full[i][0]:
        outs_full.append(mean_array_full[i] - outarray[i])
        e1 += 1
        print(f"Outlier at index {i} with value {outarray[i]}")


intensities = intensities[mask]


distances_reshaped = distances_reshaped[mask]

confidence_interval_array = confidence_interval_array[mask]

mean_array = mean_array[mask]


outarray = np.zeros([intensities.shape[0]])
# Fit a quadratic function
for z in range(intensities.shape[0]):
    vert_distances_single = distances_reshaped[z, :]
    vert_intensities_single = intensities[z, :]

    # Fit a quadratic function
    coeffs = np.polyfit(vert_distances_single, vert_intensities_single, 2)
    highest_coeff = abs(coeffs[0])
    outarray[z] = highest_coeff

e = 0
outs = []
for i in range(len(outarray)):
    if outarray[i] < confidence_interval_array[i][0]:
        outs.append(mean_array[i] - outarray[i])
        e += 1
        print(f"Outlier at index {i} with value {outarray[i]}")


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Example data for demonstration purposes
data1 = outs  # Replace with your actual data
data2 = outs_full  # Replace with your actual data

# Perform a t-test to compare the two distributions
# t_stat, p_value = stats.ttest_ind(data1, data2)
from scipy.stats import permutation_test


def mean_diff(sample1, sample2):
    return np.mean(sample1) - np.mean(sample2)


results = permutation_test(
    (data1, data2), statistic=mean_diff, n_resamples=100000, alternative="greater"
)
stat = results.statistic
p = results.pvalue
print(f"stat: {stat}")
print(f"P: {p}")

# Plotting the distributions
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot data1 on the first axes
ax1.hist(
    data1, bins=35, alpha=0.7, label="Lesional tissue", color="blue", edgecolor="black"
)
ax1.set_xlabel("Blurstat value")
ax1.set_ylabel("Frequency (lesional tissue)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.set_xlim(0, 100)  # Adjust the limits as needed

# Create a twin axes sharing the same x-axis
ax2 = ax1.twinx()

# Plot data2 on the twin axes
ax2.hist(
    data2,
    bins=200,
    alpha=0.7,
    label="Non-lesional tissue",
    color="red",
    edgecolor="black",
)
ax2.set_ylabel("Frequency (non-lesional tissue)", color="red")
ax2.tick_params(axis="y", labelcolor="red")

# Set the title and legend
fig.suptitle("Distributions of lesional and non-lesional tissue")
fig.legend(loc="upper right")

plt.show()
