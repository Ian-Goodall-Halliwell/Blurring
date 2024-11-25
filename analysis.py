import pandas as pd
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler


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


def plot_gpr_samples(gpr_model, n_samples, ax):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    """
    x = np.linspace(-4, 4, 1000)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )
    ax.plot(x, y_mean, color="black", label="Mean")
    ax.fill_between(
        x,
        y_mean - (y_std * 2.94),
        y_mean + (y_std * 2.94),
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.set_ylim([-3, 3])


if not os.path.exists("output.pkl"):
    intensities_array, distances_array_reshaped = main()
else:
    with open("output.pkl", "rb") as f:
        intensities_array, distances_array_reshaped = pickle.load(f)

# Initialize the StandardScaler
scaler = StandardScaler()

# Create an array to store the scaled data
intensities_scaled = np.zeros_like(intensities_array)

# Loop through each item in axis 1 and scale independently
for i in range(intensities_array.shape[1]):
    intarr = intensities_array[:, i, :].flatten().reshape(-1, 1)
    scaler.fit(intarr)
    intensities_scaled[:, i, :] = scaler.transform(intarr).reshape(
        intensities_array.shape[0], -1
    )

print("Original data:", intensities_array)
print("Scaled data:", intensities_scaled)

# Get per-vertex profiles
for x in range(intensities_scaled.shape[0]):
    vert_intensities = intensities_scaled[x, :, :].flatten()
    vert_distances = distances_array_reshaped[x, :, :].flatten().reshape(-1, 1)

    kernel = 1.0 * Matern(
        length_scale=1.0, length_scale_bounds=(1e-2, 10.0), nu=1.5
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-1, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, normalize_y=True)

    fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))
    n_samples = 5
    # plot prior
    plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
    axs[0].set_title("Samples from prior distribution")

    # plot posterior
    gpr.fit(vert_distances, vert_intensities)
    plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
    axs[1].scatter(
        vert_distances[:, 0],
        vert_intensities,
        color="red",
        zorder=10,
        label="Observations",
    )
    axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
    axs[1].set_title("Samples from posterior distribution")

    fig.suptitle("Matérn kernel", fontsize=18)
    plt.tight_layout()
    plt.show()
    print("e")


############################################################################################################
### PLOTTING ###
############################################################################################################
LIavgacrosstrial = np.mean(intensities_array, axis=0).transpose()
LIstdacrosstrial = np.std(intensities_array, axis=0).transpose()

LDavgacrosstrial = np.mean(distances_array_reshaped, axis=0).transpose()
LDstdacrosstrial = np.std(distances_array_reshaped, axis=0).transpose()


plt.close()

# Plotting
plt.figure(figsize=(10, 6))
plt.clf()
# Set x-axis limits
plt.xlim(-4, 5)

for row, std_dev, dists in zip(LIavgacrosstrial, LIstdacrosstrial, LDavgacrosstrial):
    plt.errorbar(dists, row, yerr=std_dev, marker="o", capsize=5)

plt.xlabel("Distance")
plt.ylabel("Value")
plt.title("Line Graph for Each Row in LIavgacrosstrial with Standard Deviations")
# plt.legend([f"Row {i+1}" for i in range(LIavgacrosstrial.shape[0])], loc="upper right")
plt.grid(True)
plt.show()

LIavgacrosstrial = np.mean(intensities_array, axis=2)
LIstdacrosstrial = np.std(intensities_array, axis=2)

LDavgacrosstrial = np.mean(distances_array_reshaped, axis=2)
LDstdacrosstrial = np.std(distances_array_reshaped, axis=2)


lastfew = LDavgacrosstrial[-500:]

# Create a mask where the last column of LDavgacrosstrial is greater than 4.2
mask = (LDavgacrosstrial[:, -1] > 4.2) | (LDavgacrosstrial[:, 0] < -4.8)

# Use np.where to find the indices that satisfy the condition


LIavgacrosstrial = LIavgacrosstrial[~mask]
LIstdacrosstrial = LIstdacrosstrial[~mask]

LDavgacrosstrial = LDavgacrosstrial[~mask]
LDstdacrosstrial = LDstdacrosstrial[~mask]
import matplotlib.pyplot as plt


# Plotting
plt.figure(figsize=(10, 6))

# Set x-axis limits
plt.xlim(-4, 5)

for row, std_dev, dists in zip(LIavgacrosstrial, LIstdacrosstrial, LDavgacrosstrial):
    plt.errorbar(dists, row, yerr=std_dev, marker="o", capsize=5)
# LDavgacrosstrial_reshaped = LDavgacrosstrial_reshaped[mask]
plt.xlabel("Distance")
plt.ylabel("Value")
plt.title("Line Graph for Each Row in LIavgacrosstrial with Standard Deviations")
# plt.legend([f"Row {i+1}" for i in range(LIavgacrosstrial.shape[0])], loc="upper right")
plt.grid(True)
plt.show()
print("e")
