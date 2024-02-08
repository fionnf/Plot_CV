import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Parameters
Std_ox = float(0.3544)
Std_re = float(0.4342)
offset = (Std_re + Std_ox) / 2
electrode_diameter_mm = 3  # Diameter in millimeters

# Plotting
plot_title = "Cyclic Voltammetry"
x_label = "Potential (V vs. Fc/Fc+)"
y_label = "Current (A)"
plot_labels = ["CV"]

aspect_ratio = (10, 6)
compute_current_density = True  # Whether to compute current density
plot_name = "CV_RHB.png"

# file paths
file_paths = [
    r"X:\My Documents\Data\CV\RHB-exp1-CVblank080224.csv"
]

# Set the style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})


def plot_voltammogram(file_path, color, label, compute_current_density=False, electrode_diameter_mm=None):
    # Read the file into a list of lines to find the header
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Locate the line with the header
    header_line = [i for i, line in enumerate(lines) if 'Potential/V, Current/A' in line][0]

    # Read the CSV data, skipping rows before the header
    data = pd.read_csv(file_path, skiprows=header_line, delimiter=',', na_values=["", " "])

    # Ensure that we strip any whitespace from the headers
    data.columns = [col.strip() for col in data.columns]

    # Check if 'Potential/V' and 'Current/A' are in the DataFrame after stripping whitespace
    if 'Potential/V' not in data.columns or 'Current/A' not in data.columns:
        raise ValueError(
            f"Columns 'Potential/V' or 'Current/A' not found in {file_path}. Available columns: {data.columns.tolist()}")

    # Apply the offset to the 'Potential/V' column
    data['Potential/V'] = data['Potential/V'] - offset

    # Convert current to current density if requested
    if compute_current_density and electrode_diameter_mm:
        radius_cm = electrode_diameter_mm / 20  # Convert mm to cm for radius
        electrode_surface_area = np.pi * (radius_cm ** 2)  # Area in cm^2
        data['Current/A'] /= electrode_surface_area
        y_label = "Current Density (A/cm^2)"
    else:
        y_label = "Current (A)"

    # Plot the voltammogram
    plt.plot(data['Potential/V'], data['Current/A'], color=color, label=label)


# Plotting
plt.figure(figsize=aspect_ratio)
colors = sns.color_palette('tab10', len(file_paths))  # Get a color palette with as many colors as files

for file_path, color, label in zip(file_paths, colors, plot_labels):
    plot_voltammogram(file_path, color, label, compute_current_density, electrode_diameter_mm)

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(plot_title)
plt.legend()
plt.tight_layout()

plt.savefig(CV_verdazyl, dpi=600)
plt.show()

