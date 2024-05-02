import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import zscore
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit

# Load data
data = pd.read_csv('/home/saumya_rlc/Downloads/data1.csv')

# Filter data for id 0 (wrist)
wrist_data = data[data['id'] == 0].copy()

# Calculate z-scores for outlier removal
for coord in ['x', 'y', 'z']:
    wrist_data[f'{coord}_z_score'] = zscore(wrist_data[coord])
    # Remove outliers (threshold = 3 standard deviations)
    wrist_data = wrist_data[np.abs(wrist_data[f'{coord}_z_score']) <= 3]

# Smooth the XYZ data using Savitzky-Golay filter
window_length = 51  # Ensure this is less than the number of data points and is an odd number
poly_order = 3
if len(wrist_data) > window_length:
    for coord in ['x', 'y', 'z']:
        wrist_data[f'{coord}_smoothed'] = savgol_filter(wrist_data[coord], window_length, poly_order)
else:
    for coord in ['x', 'y', 'z']:
        wrist_data[f'{coord}_smoothed'] = wrist_data[coord]  # If not enough data, skip smoothing

# Plotting the smoothed 3D trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Converting pandas series to numpy arrays for compatibility
x_smoothed = wrist_data['x_smoothed'].to_numpy()
y_smoothed = wrist_data['y_smoothed'].to_numpy()
z_smoothed = wrist_data['z_smoothed'].to_numpy()

# Plotting the data using numpy arrays
ax.plot(x_smoothed, y_smoothed, z_smoothed, label='Smoothed Trajectory', color='r')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Trajectory of Wrist')
ax.legend()

plt.show()
