"""
🔭 Generator: Three-Body System Trajectories
Generates numerical simulation data for a Newtonian three-body problem.
Outputs the full trajectory data in JSON format for later visualization.

Author: Mohamed Orhan Zeinel
"""

import numpy as np
import json
import os

# === Configuration ===
G = 1.0              # Normalized gravitational constant
dt = 0.005           # Time step size
steps = 10000        # Total number of simulation steps
epsilon = 1e-9       # To avoid division by zero

# === Initial conditions ===
bodies = [
    {
        "mass": 1.0,
        "position": np.array([-1.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 0.4, 0.0])
    },
    {
        "mass": 1.0,
        "position": np.array([1.0, 0.0, 0.0]),
        "velocity": np.array([0.0, -0.4, 0.0])
    },
    {
        "mass": 1.0,
        "position": np.array([0.0, 1.0, 0.0]),
        "velocity": np.array([0.3, 0.0, 0.0])
    }
]

# === Storage ===
trajectories = [[] for _ in bodies]

# === Simulation loop ===
for step in range(steps):
    positions = [body["position"] for body in bodies]
    velocities = [body["velocity"] for body in bodies]
    masses = [body["mass"] for body in bodies]
    forces = [np.zeros(3) for _ in bodies]

    # Compute gravitational force on each body
    for i in range(len(bodies)):
        for j in range(len(bodies)):
            if i != j:
                r_vec = positions[j] - positions[i]
                dist = np.linalg.norm(r_vec) + epsilon
                force = G * masses[i] * masses[j] * r_vec / dist**3
                forces[i] += force

    # Update positions and velocities
    for i in range(len(bodies)):
        velocities[i] += (forces[i] / masses[i]) * dt
        positions[i] += velocities[i] * dt
        bodies[i]["position"] = positions[i]
        bodies[i]["velocity"] = velocities[i]
        trajectories[i].append(positions[i].tolist())

# === Export to JSON ===
output_data = {
    "bodies": [{"positions": traj} for traj in trajectories]
}

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "trajectories.json")

with open(output_path, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"✅ Trajectories saved successfully to '{output_path}'")
