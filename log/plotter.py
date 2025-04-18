import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the column names for the CSV file
column_names = ["Time", "UserID", "CircleRadius", "MouseMode",
                "TargetsHit", "Misclicks", "MeanTapTime", "MedianTapTime", "Mode"]

# Load the CSV data
csv_file = 'results.csv'  # Replace with your actual CSV file path
data = pd.read_csv(csv_file, header=None, names=column_names)

unique_radii       =       [10,15,20,25,30]
mouse_times =   np.array([0.0,0.0,0.0,0.0,0.0])
one_hand_times =np.array([0.0,0.0,0.0,0.0,0.0])
two_hand_times =np.array([0.0,0.0,0.0,0.0,0.0])
 
TPU = 15
BAR_WIDTH = 0.2

users = len(data)//TPU

print("Users: ", users)

def average_column(col_name:str):
    mouse_times =   np.array([0.0,0.0,0.0,0.0,0.0])
    one_hand_times =np.array([0.0,0.0,0.0,0.0,0.0])
    two_hand_times =np.array([0.0,0.0,0.0,0.0,0.0])
    for start in range(0, len(data), TPU):
            # Calculate mean times and misclicks for each method
            for i in range(TPU):
                if i<5:
                    mouse_times[i] += data[col_name][start+i]
                elif i>=5 and i<10:
                    one_hand_times[i-5] += data[col_name][start+i]
                else:
                    two_hand_times[i-10] += data[col_name][start+i]
    return [mouse_times/TPU, one_hand_times/TPU, two_hand_times/TPU]

# Define colors for each method
COLOR_MOUSE = '#FFCB05'     # Yellow
COLOR_1H = '#3066C6'        # Light Blue
COLOR_2H = '#162E5A'        # Dark Blue

# Function to add values on top of bars
def add_values(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.2f}', ha='center', va='bottom')

def plot_column(col_name:str, plot_name):
    # Plot settings
    ret = average_column(col_name)

    plt.rcParams['font.family'] = 'Calibri'


    indices = indices = np.arange(len(unique_radii))
    fig = plt.figure(figsize=(8, 8))
    b1 = plt.bar(indices, ret[0], BAR_WIDTH, label='Mouse', 
            color = COLOR_MOUSE)
    b2 = plt.bar(indices+BAR_WIDTH, ret[1], BAR_WIDTH, label='One Hand',
            color = COLOR_1H)
    b3 = plt.bar(indices+2*BAR_WIDTH, ret[2], BAR_WIDTH, label='Two Hand',
            color = COLOR_2H)

    plt.xlabel('Circle Radius', fontsize=24)
    plt.ylabel(plot_name, fontsize=24)
    #plt.ylim(0.0, 0.8)  # Set the y-axis scale
    #plt.title('Plot 1: Mean ' + plot_name + ' per Circle Radius')
    plt.yticks(fontsize=18)
    plt.xticks(indices + BAR_WIDTH, unique_radii, fontsize=18)
    #legend = plt.legend()
    plt.tight_layout()

    plt.savefig("Figures/"+ plot_name+'.png', format='png', dpi=600)

def saveLegend():
    # Define some dummy data and labels for demonstration

    plt.rcParams['font.family'] = 'Calibri'

    categories = ['Mouse', 'One Hand', 'Two Hands']
    colors = ['#FFCB05', '#3066C6', '#162E5A']  # Yellow, Light Blue, Dark Blue

    # Create a dummy figure and axis
    fig, ax = plt.subplots()

    # Create the bars just to get the legend
    for i, color in enumerate(colors):
        ax.bar(0, 0, color=color, label=categories[i])  # Zero-width bars

    # Add the legend
    legend = ax.legend(loc='center', frameon=False)

    # Create a new figure for the legend
    fig_legend = plt.figure(figsize=(2*10, 1*10)) # Size can be adjusted
    fig_legend.legend(handles=ax.get_legend_handles_labels()[0], labels=categories, loc='center', frameon=False)

    # Save only the legend as a separate image file
    fig_legend.savefig('Figures/legend.png', format='png', bbox_inches='tight', dpi=600)


# Create the directory
try:
    os.mkdir("Figures/")
    print(f"Directory Figures/ created successfully.")
except FileExistsError:
    pass

plot_column("MeanTapTime", "Mean time between taps")
plot_column("TargetsHit", "Targets hit")
plot_column("Misclicks", "Misclicks")

saveLegend()
#plot_column("MedianTapTime", "Median Tap Time")