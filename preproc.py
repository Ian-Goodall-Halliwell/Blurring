import os
import numpy as np
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
                for z in range(intensities.shape[1]):

                    vert_distances_single = distances[z]
                    vert_intensities_single = intensities[z]

                    # Fit a quadratic function
                    coeffs = np.polyfit(
                        vert_distances_single, vert_intensities_single, 2
                    )
                    # Generate x values for plotting the fitted curve
                    x_fit = np.linspace(
                        vert_distances_single.min(), vert_distances_single.max(), 100
                    )
                    y_fit = np.polyval(coeffs, x_fit)

                    # Plot the original data points
                    plt.scatter(
                        vert_distances_single,
                        vert_intensities_single,
                        label="Data Points",
                    )

                    # Plot the fitted quadratic function
                    plt.plot(x_fit, y_fit, color="red", label="Fitted Quadratic Curve")

                    # Add labels and legend
                    plt.xlabel("Vertical Distances")
                    plt.ylabel("Vertical Intensities")
                    plt.legend()

                    # Show the plot
                    plt.show()
                    highest_coeff = abs(coeffs[0])
                    # highest_coeff_values.append(highest_coeff)


hemis = ["L", "R"]
datadir = "E:/zbrains_blur"

for fold in os.listdir(datadir):

    intensities_array, distances_array = load_data(datadir, hemis)
