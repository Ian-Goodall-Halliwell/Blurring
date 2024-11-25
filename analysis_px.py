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


L_intensities_array = np.zeros((4842, 16))
L_distances_array = np.zeros((4842, 15))

R_intensities_array = np.zeros((4842, 16))
R_distances_array = np.zeros((4842, 15))

distances = np.genfromtxt(
    "E:/data/derivatives/zbrains_blur/PX001/sub-PX001_ses-01_L_T1map_blur_distances.csv",
    delimiter=",",
    skip_header=1,
).transpose()
intensities = np.genfromtxt(
    "E:/data/derivatives/zbrains_blur/PX001/sub-PX001_ses-01_L_T1map_blur_intensities.csv",
    delimiter=",",
    skip_header=1,
).transpose()

mask = np.genfromtxt(
    "C:/Users/Ian/Documents/GitHub/Blurring/output/sub-PX001_ses-02_space-nativepro_T1w_brainthresholdedSurface.csv",
    delimiter=",",
    skip_header=1,
).transpose()

# Convert mask to boolean array
mask = mask.astype(bool)

R_intensities_array[:, :] = intensities
R_distances_array[:, :] = distances


RIavgacrosstrial = R_intensities_array
RDavgacrosstrial = R_distances_array


RDavgacrosstrial_reshaped = np.zeros((len(RDavgacrosstrial), 16))
for en, x in enumerate(R_distances_array):

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
    # print(x)


# threshold = 5.5
# mask = (
#     LDavgacrosstrial_reshaped[:, 11] <= threshold
# )  # Assuming you want to check the first trial


RIavgacrosstrial = RIavgacrosstrial[mask]


RDavgacrosstrial = RDavgacrosstrial[mask]

RDavgacrosstrial_reshaped = RDavgacrosstrial_reshaped[mask]

# test = LDavgacrosstrial_reshaped[-500:-1]
# test2 = LDavgacrosstrial_reshaped[-1000:-501]
# test3 = LIavgacrosstrial[-500:-1]
# test4 = LIavgacrosstrial[-1000:-501]


import matplotlib.pyplot as plt

plt.close()

# Plotting
plt.figure(figsize=(10, 6))
plt.clf()
# Set x-axis limits
plt.xlim(-5, 5)

for row, dists in zip(RIavgacrosstrial, RDavgacrosstrial_reshaped):
    plt.errorbar(dists, row, marker="o", capsize=5)

plt.xlabel("Distance")
plt.ylabel("Value")
plt.title("Line Graph for Each Row in LIavgacrosstrial with Standard Deviations")
# plt.legend([f"Row {i+1}" for i in range(LIavgacrosstrial.shape[0])], loc="upper right")
plt.grid(True)
plt.show()


distances = np.genfromtxt(
    "E:/data/derivatives/zbrains_blur/HC007/sub-HC007_ses-01_L_T1map_blur_distances.csv",
    delimiter=",",
    skip_header=1,
).transpose()
intensities = np.genfromtxt(
    "E:/data/derivatives/zbrains_blur/HC007/sub-HC007_ses-01_L_T1map_blur_intensities.csv",
    delimiter=",",
    skip_header=1,
).transpose()

mask = np.genfromtxt(
    "C:/Users/Ian/Documents/GitHub/Blurring/output/sub-PX001_ses-02_space-nativepro_T1w_brainthresholdedSurface.csv",
    delimiter=",",
    skip_header=1,
).transpose()

# Convert mask to boolean array
mask = mask.astype(bool)

R_intensities_array[:, :] = intensities
R_distances_array[:, :] = distances


RIavgacrosstrial = R_intensities_array
RDavgacrosstrial = R_distances_array


RDavgacrosstrial_reshaped = np.zeros((len(RDavgacrosstrial), 16))
for en, x in enumerate(R_distances_array):

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
    # print(x)


# threshold = 5.5
# mask = (
#     LDavgacrosstrial_reshaped[:, 11] <= threshold
# )  # Assuming you want to check the first trial


RIavgacrosstrial = RIavgacrosstrial[mask]


RDavgacrosstrial = RDavgacrosstrial[mask]

RDavgacrosstrial_reshaped = RDavgacrosstrial_reshaped[mask]

# test = LDavgacrosstrial_reshaped[-500:-1]
# test2 = LDavgacrosstrial_reshaped[-1000:-501]
# test3 = LIavgacrosstrial[-500:-1]
# test4 = LIavgacrosstrial[-1000:-501]


import matplotlib.pyplot as plt

plt.close()

# Plotting
plt.figure(figsize=(10, 6))
plt.clf()
# Set x-axis limits
plt.xlim(-5, 5)

for row, dists in zip(RIavgacrosstrial, RDavgacrosstrial_reshaped):
    plt.errorbar(dists, row, marker="o", capsize=5)

plt.xlabel("Distance")
plt.ylabel("Value")
plt.title("Line Graph for Each Row in LIavgacrosstrial with Standard Deviations")
# plt.legend([f"Row {i+1}" for i in range(LIavgacrosstrial.shape[0])], loc="upper right")
plt.grid(True)
plt.show()
