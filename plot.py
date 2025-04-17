import pandas as pd
import matplotlib.pyplot as plt

# === Step 1: Load data ===
file_path = "log/results.csv"  # Update this path if needed
columns = [
    "Time", "UserID", "CircleRadius", "MouseMode",
    "TargetsHit", "MissClicks", "MeanTapTime", "MedianTapTime"
]
df = pd.read_csv(file_path, header=None, names=columns)

# === Step 2: Assign Modes ===
# Assumes each user has 15 tests in order:
# First 5 = MouseMode On, next 5 = One Hand, last 5 = Two Hands
def assign_modes_fixed(group):
    group = group.reset_index(drop=True)
    group["Mode"] = ["MouseMode On"] * 5 + ["One Hand"] * 5 + ["Two Hands"] * 5
    return group

df_labeled = df.groupby("UserID", group_keys=False).apply(assign_modes_fixed)

# === Step 3: Compute mean tap times per circle radius and mode ===
#summary = df_labeled.groupby(["CircleRadius", "Mode"])["MeanTapTime"].mean().unstack()
summary = df_labeled.groupby(["CircleRadius", "Mode"])["MissClicks"].mean().unstack()



# === Step 4: Plot single bar chart ===
plt.figure(figsize=(12, 6))
summary.plot(kind="bar", ax=plt.gca())
plt.title("Mean misclicks")
plt.xlabel("Circle Radius")
plt.ylabel("Average Tap Time (s)")
plt.xticks(rotation=0)
plt.legend(title="Input Mode")
plt.tight_layout()
plt.grid(axis='y')
plt.show()
