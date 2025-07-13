import json
import matplotlib.pyplot as plt

# Correct path to the data file (same directory)
DATA_FILE = "trajectories.json"

# Load the data
try:
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"🚫 Data file not found: {DATA_FILE}")

trajectories = data["trajectories"]

# Plot the trajectories
colors = ['r', 'g', 'b']
labels = ['Body 1', 'Body 2', 'Body 3']

plt.figure(figsize=(8, 6))
for i in range(3):
    x = [pos[0] for pos in trajectories[i]]
    y = [pos[1] for pos in trajectories[i]]
    plt.plot(x, y, color=colors[i], label=labels[i])

plt.title("Three-Body Trajectories")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.savefig("three_body_plot.png")
plt.show()
