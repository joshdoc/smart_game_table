import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the column names for the CSV file
column_names = ["Time", "UserID", "CircleRadius", "MouseMode",
                "TargetsHit", "Misclicks", "MeanTapTime", "MedianTapTime"]

# Load the CSV data
csv_file = 'log/results.csv'  # Replace with your actual CSV file path
data = pd.read_csv(csv_file, header=None, names=column_names)

# Sort data by CircleRadius for consistency
data = data.sort_values(by='CircleRadius')

# Get unique circle radii
unique_radii = sorted(data['CircleRadius'].unique())

# Prepare the lists to hold calculations per radius
mean_tap_times = []
mean_misclicks = []

# User tests are grouped in blocks of 15: 5 for each input method
num_tests_per_user = 15
num_methods = 3
tests_per_method = 5  # Mouse, One-Handed, Two-Handed

# Check if data frame was read correctly
if data.empty:
    print("Data frame is empty. Please check the CSV file path and format.")
else:
    print(f"Data successfully loaded with shape {data.shape}")

# Calculate values for bar plots
for radius in unique_radii:
    mouse_times = []
    one_hand_times = []
    two_hand_times = []
    
    mouse_clicks = []
    one_hand_clicks = []
    two_hand_clicks = []

    # Iterate over each user (assuming tests are sequential by user)
    for start in range(0, len(data), num_tests_per_user):
        user_data = data.iloc[start:start + num_tests_per_user]
        
        # Filter user's data by circle radius
        user_radius_data = user_data[user_data['CircleRadius'] == radius]
        if len(user_radius_data) < tests_per_method * num_methods:
            continue
        
        # Calculate mean times and misclicks for each method
        mouse_times.append(user_radius_data.iloc[0:5]['MeanTapTime'].mean())
        one_hand_times.append(user_radius_data.iloc[5:10]['MeanTapTime'].mean())
        two_hand_times.append(user_radius_data.iloc[10:15]['MeanTapTime'].mean())

        mouse_clicks.append(user_radius_data.iloc[0:5]['Misclicks'].mean())
        one_hand_clicks.append(user_radius_data.iloc[5:10]['Misclicks'].mean())
        two_hand_clicks.append(user_radius_data.iloc[10:15]['Misclicks'].mean())

    # Append mean results for this radius
    mean_tap_times.append([
        np.mean(mouse_times),
        np.mean(one_hand_times),
        np.mean(two_hand_times)
    ])
    mean_misclicks.append([
        np.mean(mouse_clicks),
        np.mean(one_hand_clicks),
        np.mean(two_hand_clicks)
    ])

# Convert lists to numpy arrays for plotting
mean_tap_times = np.array(mean_tap_times)
mean_misclicks = np.array(mean_misclicks)

# Plot settings
bar_width = 0.2
indices = np.arange(len(unique_radii))

# Plot 1: Mean Tap Time
plt.figure(figsize=(10, 6))
plt.bar(indices, mean_tap_times[:, 0], bar_width, label='Mouse')
plt.bar(indices + bar_width, mean_tap_times[:, 1], bar_width, label='One Hand')
plt.bar(indices + 2 * bar_width, mean_tap_times[:, 2], bar_width, label='Two Hands')

plt.xlabel('Circle Radius')
plt.ylabel('Mean Tap Time')
plt.ylim(0.0, 0.8)  # Set the y-axis scale
plt.title('Plot 1: Mean Tap Time per Circle Radius')
plt.xticks(indices + bar_width, unique_radii)
plt.legend()
plt.tight_layout()

# Plot 2: Mean Misclicks
plt.figure(figsize=(10, 6))
plt.bar(indices, mean_misclicks[:, 0], bar_width, label='Mouse')
plt.bar(indices + bar_width, mean_misclicks[:, 1], bar_width, label='One Hand')
plt.bar(indices + 2 * bar_width, mean_misclicks[:, 2], bar_width, label='Two Hands')

plt.xlabel('Circle Radius')
plt.ylabel('Mean Misclicks')
# plt.ylim(0.0, 0.8)  # Remove this line for automatic scaling
plt.title('Plot 2: Mean Misclicks per Circle Radius')
plt.xticks(indices + bar_width, unique_radii)
plt.legend()
plt.tight_layout()

# Show the plots
plt.show()