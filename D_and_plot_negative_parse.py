import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# ========== Constants and Parameters ==========

directory_path = r"G:\.shortcut-targets-by-id\1gpf-XKVVvMHbMGqpyQS5Amwp9fh8r96B\RUG shared\Master Project\Experiment files\FF015\CV_data"
filename_prefix = "FF015cvd_n"

TARGET_VOLTAGE_RANGE = 0.5
OFFSET = 0.0
ELECTRODE_DIAMETER = None
PLOT_CURRENT_DENSITY = False

PLOT_SIZE = (16, 6)
COLORS = sns.color_palette('tab10')
SCAN_RATE_AXES = ["Scan Rate (V/s)", "Peak Current (A)"]
VOLTAMMOGRAM_AXES = ["Potential (V vs. Ag/AgCl)", "Current Density (A/cm^2)" if PLOT_CURRENT_DENSITY else "Current (A)"]
SWEEP_SELECTIONS = [2, 3]

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})

# ========== Parsing Functions ==========

def parse_cv_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    scan_rate = None
    for line in lines:
        if "Scan Rate (V/s)" in line:
            scan_rate = float(line.split('=')[1].strip())
            break

    data = {}
    current_segment = None
    for line in lines:
        if "Segment" in line:
            current_segment = int(line.split()[-1].strip(":"))
            data[current_segment] = []
        elif "Ep =" in line and (current_segment == 2 or current_segment == 3):
            ep = float(line.split('=')[1].replace('V', '').strip())
            ip = float(lines[lines.index(line)+1].split('=')[1].replace('A', '').strip())
            data[current_segment].append((ep, ip))

    return scan_rate, data.get(2, []), data.get(3, [])

def parse_all_files(directory, prefix):
    file_pattern = os.path.join(directory, f"{prefix}*.csv")
    files = glob.glob(file_pattern)

    data_collection = {}
    for file in files:
        scan_rate, segment2, segment3 = parse_cv_file(file)
        delta_Ep = None
        if segment2 and segment3:
            delta_Ep = segment3[0][0] - segment2[0][0]
        data_collection[file] = {
            "Scan Rate": scan_rate,
            "Segment 2 Data": segment2,
            "Segment 3 Data": segment3,
            "Delta Ep": delta_Ep
        }

    return data_collection

def get_scan_rates_from_files(directory, prefix):
    file_pattern = os.path.join(directory, f"{prefix}*.csv")
    files = glob.glob(file_pattern)

    scan_rates = []
    for file in files:
        scan_rate, _, _ = parse_cv_file(file)
        if scan_rate:
            scan_rates.append(scan_rate)

    return scan_rates

data_collection = parse_all_files(directory_path, filename_prefix)

FILE_PATHS = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.startswith(filename_prefix) and file.endswith('.csv')]
SCAN_RATES = get_scan_rates_from_files(directory_path, filename_prefix)
PLOT_LABELS = [f"Scan Rate: {rate} V/s" for rate in SCAN_RATES]

PEAK_CURRENTS_OX = [entry["Segment 2 Data"][0][1] for entry in data_collection.values() if entry["Segment 2 Data"]]
PEAK_CURRENTS_RED = [entry["Segment 3 Data"][0][1] for entry in data_collection.values() if entry["Segment 3 Data"]]

data_collection = parse_all_files(directory=directory_path, prefix=filename_prefix)

# Create a DataFrame to store and display information in tabular form
df = pd.DataFrame(columns=["File", "Scan Rate", "Oxidation Peak (Ep, Ip)", "Reduction Peak (Ep, Ip)", "Delta Ep"])

for idx, (file, data) in enumerate(data_collection.items()):
    oxidation_peak = data["Segment 2 Data"][0] if data["Segment 2 Data"] else None 
    reduction_peak = data["Segment 3 Data"][0] if data["Segment 3 Data"] else None
    df.loc[idx] = [file, data["Scan Rate"], oxidation_peak, reduction_peak, data["Delta Ep"]]
    # Extract just the filename from the full path
    df['File'] = df['File'].apply(lambda x: os.path.basename(x))

    # Set the display options to make the output neater
    pd.set_option('display.max_columns', None)  # show all columns
    pd.set_option('display.expand_frame_repr', False)  # prevent wrapping to the next line

# Display the table
print(df)

# Display the table
print(df)



# ========== Plotting Functions ==========
def plot_scan_rate_vs_peak_current(scan_rates, peak_currents_ox, peak_currents_red):
    # Constants
    n = 1
    diameter = 0.3
    A = np.pi * (diameter / 2)**2
    C = 1e-3
    constant = 2.69e5 * n**1.5 * A * C

    sqrt_scan_rates = np.sqrt(scan_rates)

    z_ox, cov_ox = np.polyfit(sqrt_scan_rates, peak_currents_ox, 1, cov=True)
    p_ox = np.poly1d(z_ox)
    R2_ox = 1 - (np.sum((peak_currents_ox - p_ox(sqrt_scan_rates))**2) / ((len(peak_currents_ox) - 1) * np.var(peak_currents_ox, ddof=1)))

    z_red, cov_red = np.polyfit(sqrt_scan_rates, peak_currents_red, 1, cov=True)
    p_red = np.poly1d(z_red)
    R2_red = 1 - (np.sum((peak_currents_red - p_red(sqrt_scan_rates))**2) / ((len(peak_currents_red) - 1) * np.var(peak_currents_red, ddof=1)))

    D_ox = (z_ox[0] / constant)**2
    D_red = (z_red[0] / constant)**2

    plt.annotate(f"D_ox = {D_ox:.2e} cm^2/s", xy=(0.6, 0.9), xycoords="axes fraction")
    plt.annotate(f"D_red = {D_red:.2e} cm^2/s", xy=(0.6, 0.85), xycoords="axes fraction")

    plt.scatter(sqrt_scan_rates, peak_currents_ox, label='Oxidation Peaks', color='blue')
    plt.plot(sqrt_scan_rates, p_ox(sqrt_scan_rates), "b--", label=f"Oxidation Fit (R^2={R2_ox:.2f})")
    plt.scatter(sqrt_scan_rates, peak_currents_red, label='Reduction Peaks', color='red')
    plt.plot(sqrt_scan_rates, p_red(sqrt_scan_rates), "r--", label=f"Reduction Fit (R^2={R2_red:.2f})")
    plt.xlabel("Square Root of Scan Rate (V/s)^0.5")
    plt.ylabel("Peak Current (A)")
    plt.legend()
    plt.title('Square Root of Scan Rate vs Peak Current')
    plt.grid(True)
    plt.show()


def plot_voltammogram(file_path, color, label):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    start_line = lines.index('Potential/V, Current/A\n')
    data = pd.read_csv(file_path, skiprows=start_line, delimiter=',', na_values=["", " "])
    data.columns = [col.strip() for col in data.columns]

    plt.plot(data['Potential/V'], data['Current/A'], color=color, label=f"{label} mV/s")
    plt.xlabel("Potential (V vs. Ag/AgCl)")
    plt.ylabel("Current (A)")
    plt.title("Cyclic Voltammetry")
    plt.legend()


# Use the function:
data_collection = parse_all_files(directory=directory_path, prefix=filename_prefix)

# You can then use the data in 'data_collection' for other analyses, or use the plotting functions.

for file, data in data_collection.items():
    print(f"File: {file}")
    print("Scan Rate:", data["Scan Rate"])
    print("Segment 2 Data:", data["Segment 2 Data"])
    print("Segment 3 Data:", data["Segment 3 Data"])
    print("Difference in Ep (Segment 3 - Segment 2):", data["Delta Ep"])
    print("===================================")

# Generate side by side plots
fig, ax = plt.subplots(1, 2, figsize=PLOT_SIZE)

# Left plot (Cyclic Voltammograms)
plt.sca(ax[0])
for file_path, color, label in zip(FILE_PATHS, COLORS, PLOT_LABELS):
    plot_voltammogram(file_path, color, label, ELECTRODE_DIAMETER, OFFSET, PLOT_CURRENT_DENSITY)

# Right plot (Scan Rate vs Peak Current)
plt.sca(ax[1])
plot_scan_rate_vs_peak_current(SCAN_RATES, PEAK_CURRENTS_OX, PEAK_CURRENTS_RED)

plt.tight_layout()

# Save the figure before displaying it
plt.savefig('negative_D.png', dpi=300, bbox_inches='tight')

plt.show()