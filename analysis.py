import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import permutation_test
from utils import load_data, reshape_distances, process_vertex, analyze_data
import nibabel as nib
from random import sample


def delete_empty_folder(folder_path):
    if not os.listdir(folder_path):
        os.rmdir(folder_path)
        print(f"Deleted empty folder: {folder_path}")
    else:
        print(f"Folder is not empty: {folder_path}")


def main():
    hemis = ["L", "R"]
    datadir = "E:/zbrains_blur"

    for fold in os.listdir(datadir):
        delete_empty_folder(os.path.join(datadir, fold))

    intensities_array, distances_array = load_data(datadir, hemis)
    distances_array_reshaped = reshape_distances(distances_array)

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
std_array = np.zeros([distances_array_reshaped.shape[0]])
mask_midline = ~np.concatenate(
    list(
        nib.load(f"src/data/fsLR-5k.{hemi}.mask.shape.gii").darrays[0].data
        for hemi in ["L", "R"]
    )
).astype(bool)


# intensities_mean_clean  = intensities_array.copy()
# distances_mean_clean = distances_array_reshaped.copy()

# intensities_mean_clean[mask_midline,:] = np.nan
# distances_mean_clean[mask_midline,:] = np.nan


# # Calculate the mean across the final dimension
# intensities_mean_clean = np.std(intensities_mean_clean[~mask_midline], axis=0).transpose()
# distances_mean_clean = np.mean(distances_mean_clean[~mask_midline], axis=0).transpose()


# # Plot the data
# plt.figure(figsize=(10, 6), facecolor='black')

# # Loop through each row in the first dimension and plot
# for i in range(intensities_mean_clean.shape[0]):
#     plt.plot(distances_mean_clean[i, :], intensities_mean_clean[i, :], label=f'Row {i+1}')

# # Set the x and y limits
# plt.xlim(-6, 2)

# # Customize the plot appearance
# plt.xlabel('Distances', color='white')
# plt.ylabel('Subject Intensities (SD)', color='white')
# plt.title('SD of Subject Intensities vs. Distances', color='white')

# # Customize the axes
# plt.gca().spines['bottom'].set_color('white')
# plt.gca().spines['top'].set_color('white')
# plt.gca().spines['right'].set_color('white')
# plt.gca().spines['left'].set_color('white')
# plt.gca().tick_params(axis='x', colors='white')
# plt.gca().tick_params(axis='y', colors='white')

# # Add a grid
# plt.grid(color='gray', linestyle='--', linewidth=0.5)

# # Set the background color of the plot area
# plt.gca().set_facecolor('black')

# # Show the plot
# plt.show()


# intensities_mean_clean  = intensities_array.copy()
# distances_mean_clean = distances_array_reshaped.copy()

# intensities_mean_clean[mask_midline,:] = np.nan
# distances_mean_clean[mask_midline,:] = np.nan


# # Calculate the mean across the final dimension
# intensities_mean_clean = np.mean(intensities_mean_clean[~mask_midline], axis=0).transpose()
# distances_mean_clean = np.mean(distances_mean_clean[~mask_midline], axis=0).transpose()

# # Plot the data
# plt.figure(figsize=(10, 6), facecolor='black')

# # Loop through each row in the first dimension and plot
# for i in range(intensities_mean_clean.shape[0]):
#     plt.plot(distances_mean_clean[i, :], intensities_mean_clean[i, :], label=f'Row {i+1}')

# # Set the x and y limits
# plt.xlim(-6, 2)

# # Customize the plot appearance
# plt.xlabel('Distances', color='white')
# plt.ylabel('Vertex Intensities (Mean)', color='white')
# plt.title('Mean Subject Intensities vs. Distances', color='white')

# # Customize the axes
# plt.gca().spines['bottom'].set_color('white')
# plt.gca().spines['top'].set_color('white')
# plt.gca().spines['right'].set_color('white')
# plt.gca().spines['left'].set_color('white')
# plt.gca().tick_params(axis='x', colors='white')
# plt.gca().tick_params(axis='y', colors='white')

# # Add a grid
# plt.grid(color='gray', linestyle='--', linewidth=0.5)

# # Set the background color of the plot area
# plt.gca().set_facecolor('black')

# # Show the plot
# plt.show()

# # Calculate the mean across the final dimension
# intensities_mean_clean = np.std(intensities_array, axis=-1)
# distances_mean_clean = np.mean(distances_array_reshaped, axis=-1)

# intensities_mean_clean[mask_midline,:] = np.nan
# distances_mean_clean[mask_midline,:] = np.nan


# # Plot the data
# plt.figure(figsize=(10, 6), facecolor='black')

# # Loop through each row in the first dimension and plot
# for i in range(intensities_mean_clean.shape[0]):
#     plt.plot(distances_mean_clean[i, :], intensities_mean_clean[i, :], label=f'Row {i+1}')

# # Set the x and y limits
# plt.xlim(-6, 2)

# # Customize the plot appearance
# plt.xlabel('Distances', color='white')
# plt.ylabel('Vertex Intensities (SD)', color='white')
# plt.title('SD of Vertex Intensities vs. Distances', color='white')

# # Customize the axes
# plt.gca().spines['bottom'].set_color('white')
# plt.gca().spines['top'].set_color('white')
# plt.gca().spines['right'].set_color('white')
# plt.gca().spines['left'].set_color('white')
# plt.gca().tick_params(axis='x', colors='white')
# plt.gca().tick_params(axis='y', colors='white')

# # Add a grid
# plt.grid(color='gray', linestyle='--', linewidth=0.5)

# # Set the background color of the plot area
# plt.gca().set_facecolor('black')

# # Show the plot
# plt.show()


# # Calculate the mean across the final dimension
# intensities_mean_clean = np.mean(intensities_array, axis=-1)
# distances_mean_clean = np.mean(distances_array_reshaped, axis=-1)

# intensities_mean_clean[mask_midline,:] = np.nan
# distances_mean_clean[mask_midline,:] = np.nan


# # Plot the data
# plt.figure(figsize=(10, 6), facecolor='black')

# # Loop through each row in the first dimension and plot
# for i in range(intensities_mean_clean.shape[0]):
#     plt.plot(distances_mean_clean[i, :], intensities_mean_clean[i, :], label=f'Row {i+1}')

# # Set the x and y limits
# plt.xlim(-6, 2)

# # Customize the plot appearance
# plt.xlabel('Distances', color='white')
# plt.ylabel('Vertex Intensities (Mean)', color='white')
# plt.title('Mean Vertex Intensities vs. Distances', color='white')

# # Customize the axes
# plt.gca().spines['bottom'].set_color('white')
# plt.gca().spines['top'].set_color('white')
# plt.gca().spines['right'].set_color('white')
# plt.gca().spines['left'].set_color('white')
# plt.gca().tick_params(axis='x', colors='white')
# plt.gca().tick_params(axis='y', colors='white')

# # Add a grid
# plt.grid(color='gray', linestyle='--', linewidth=0.5)

# # Set the background color of the plot area
# plt.gca().set_facecolor('black')

# # Show the plot
# plt.show()

# Parallelize the computation
results = Parallel(n_jobs=-1)(
    delayed(process_vertex)(
        x, intensities_array, distances_array_reshaped, mask_midline
    )
    for x in range(intensities_array.shape[0])
)

# Store the results
for x, (mean, confidence_interval, std) in enumerate(results):
    mean_array[x] = mean
    confidence_interval_array[x] = confidence_interval
    std_array[x] = std


print(f"Mean array: {mean_array}")
print(f"Confidence interval array: {confidence_interval_array}")
print(f"Standard deviation array: {std_array}")

# Save variables to a file
with open("mean_conf_arr.pkl", "wb") as f:
    pickle.dump((mean_array, confidence_interval_array, std_array), f)


# Load L and R versions of distances and intensities
L_distances = np.genfromtxt(
    "PX001/sub-PX001_ses-01_L_T1map_blur_distances.csv",
    delimiter=",",
    skip_header=1,
).transpose()
L_intensities = np.genfromtxt(
    "PX001/sub-PX001_ses-01_L_T1map_blur_intensities.csv",
    delimiter=",",
    skip_header=1,
).transpose()

R_distances = np.genfromtxt(
    "PX001/sub-PX001_ses-01_R_T1map_blur_distances.csv",
    delimiter=",",
    skip_header=1,
).transpose()
R_intensities = np.genfromtxt(
    "PX001/sub-PX001_ses-01_R_T1map_blur_intensities.csv",
    delimiter=",",
    skip_header=1,
).transpose()

# Concatenate L and R versions
distances = np.concatenate((L_distances, R_distances), axis=0)
intensities = np.concatenate((L_intensities, R_intensities), axis=0)

# Load mask
mask_R = np.genfromtxt(
    "output/sub-PX001_ses-02_space-nativepro_T1w_brainthresholdedSurface.csv",
    delimiter=",",
    skip_header=1,
).transpose()
mask_L = np.zeros([R_intensities.shape[0]])

# Convert mask to boolean array
mask_R = mask_R.astype(bool)
mask_L = mask_L.astype(bool)
mask = np.concatenate((mask_L, mask_R), axis=0)
mask_inv = ~mask
nonmask = np.zeros_like(mask).astype(bool)

outs_full, _ = analyze_data(
    distances, intensities, mask * ~mask_midline, confidence_interval_array, mean_array
)

outs, _ = analyze_data(
    distances,
    intensities,
    ~(~mask_inv * ~mask_midline),
    confidence_interval_array,
    mean_array,
)

outs_positive, fullout = analyze_data(
    distances,
    intensities,
    mask_midline,
    confidence_interval_array,
    mean_array,
    fullcort=True,
)


with open("outs.pkl", "wb") as f:
    pickle.dump(outs_positive, f)

with open("fullout.pkl", "wb") as f:
    pickle.dump(fullout, f)


def mean_diff(sample1, sample2):
    return np.mean(sample1) - np.mean(sample2)


def permutation_test(sample1, sample2, num_permutations=10000):
    observed_diff = mean_diff(sample1, sample2)
    combined = np.concatenate([sample1, sample2])
    count = 0

    for _ in range(num_permutations):
        np.random.shuffle(combined)
        perm_sample1 = combined[: len(sample1)]
        perm_sample2 = combined[len(sample1) :]
        perm_diff = mean_diff(perm_sample1, perm_sample2)
        if perm_diff >= observed_diff:
            count += 1

    p_value = count / num_permutations
    return observed_diff, p_value


outs = outs[~np.isnan(outs)]
outs_full = outs_full[~np.isnan(outs_full)]

observed_diff, p_value = permutation_test(outs, outs_full, num_permutations=1000000)


# allvals = outs_positive[~np.isnan(outs_positive)]
# results = permutation_test(
#     (allvals, outs),
#     statistic=mean_diff,
#     n_resamples=100000,
#     alternative="greater",
#     random_state=42,
#     vectorized=False
# )
# stat = results.statistic
# p = results.pvalue
print(f"observed diff: {observed_diff}")
print(f"P: {p_value}")

# Plotting the distributions
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot data1 on the first axes
ax1.hist(
    outs,
    bins=35,
    alpha=0.7,
    label="Lesional tissue",
    color="blue",
    edgecolor="black",
)
ax1.set_xlabel("Quantitative blur")
ax1.set_ylabel("Frequency (lesional tissue)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.set_xlim(-100, 100)  # Adjust the limits as needed

# Create a twin axes sharing the same x-axis
ax2 = ax1.twinx()

# Plot data2 on the twin axes
ax2.hist(
    outs_full,
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
