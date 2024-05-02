import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/home/saumya_rlc/Downloads/data1.csv')

# Filter data for id 0 (waist)
waist_data = data[data['id'] == 0].copy()

# Calculate angles in degrees from X, Y coordinates
waist_data['angle_degrees'] = np.degrees(np.arctan2(waist_data['y'], waist_data['x']))

# Standardize angles to range from 0 to 360
waist_data['angle_degrees'] = waist_data['angle_degrees'] % 360

# Calculate change in angle relative to the first measurement
initial_angle = waist_data['angle_degrees'].iloc[0]
waist_data['relative_angle_change'] = waist_data['angle_degrees'] - initial_angle

# Adjust for wrapping (handle transitions through 0 degrees)
waist_data['relative_angle_change'] = waist_data['relative_angle_change'].apply(lambda x: x if x >= -180 else x + 360)
waist_data['relative_angle_change'] = waist_data['relative_angle_change'].apply(lambda x: x if x <= 180 else x - 360)

# Convert Series to numpy arrays for plotting
marker_time_array = waist_data['marker_time'].to_numpy()
relative_angle_change_array = waist_data['relative_angle_change'].to_numpy()

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(marker_time_array, relative_angle_change_array, label='Relative Angle Change', marker='o')
plt.title('Relative Angle Change of the Waist over Time')
plt.xlabel('Marker Time')
plt.ylabel('Angle Change (degrees)')
plt.grid(True)
plt.legend()
plt.show()
