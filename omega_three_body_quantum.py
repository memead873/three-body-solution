import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# Constants
G_SI = 6.67430e-11
c_SI = 299792458
AU = 1.495978707e11
YEAR = 365.25 * 24 * 3600
SOLAR_MASS = 1.9885e30
EPSILON = 1e-8

# Normalized constants
G = G_SI * YEAR**2 / (AU**3 * SOLAR_MASS)
c = c_SI * YEAR / AU

print(f"Normalized G = {G}")
print(f"Normalized c = {c}")

# --- Folded Temporal Phase Factor ---
def folded_time(t, positions, velocities):
    kinetic = sum(0.5 * np.linalg.norm(v)**2 for v in velocities)
    potential = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            r_ij = np.linalg.norm(positions[i] - positions[j]) + EPSILON
            potential -= G / r_ij
    phase = kinetic + potential + 1e-12
    return np.arctan(phase)

# --- Pseudo-Quantum Correction Field ---
def compute_theta_quantum(positions):
    center = np.mean(positions, axis=0)
    theta = np.zeros((3, 3))
    for i in range(3):
        delta = positions[i] - center
        r_norm = np.linalg.norm(delta) + EPSILON
        theta[i] = -0.0001 * delta * np.cos(r_norm / 1e2)
    return theta

# --- Relativistic Acceleration (Post-Newtonian Approximation) ---
def relativistic_acceleration(i, positions, velocities, masses):
    a_rel = np.zeros(3)
    for j in range(3):
        if i != j:
            rij = positions[j] - positions[i]
            vij = velocities[j] - velocities[i]
            r = np.linalg.norm(rij) + EPSILON
            v2 = np.linalg.norm(velocities[j])**2
            correction = (4 * G * masses[j] / (r * c**2) - v2 / c**2)
            a_rel += G * masses[j] * rij / r**3 * (1 + correction)
    return a_rel

# --- Full Dynamic System with optional components ---
def zeinel_dynamics(t, y, masses, use_relativity=True, use_quantum=True):
    positions = y[:9].reshape(3, 3)
    velocities = y[9:].reshape(3, 3)
    accelerations = np.zeros_like(positions)

    for i in range(3):
        newtonian = np.zeros(3)
        for j in range(3):
            if i != j:
                rij = positions[j] - positions[i]
                r = np.linalg.norm(rij) + EPSILON
                newtonian += G * masses[j] * rij / r**3
        if use_relativity:
            accelerations[i] = relativistic_acceleration(i, positions, velocities, masses)
        accelerations[i] += newtonian

    if use_quantum:
        accelerations += compute_theta_quantum(positions)

    tau_factor = 1.0 / (np.sqrt(np.abs(folded_time(t, positions, velocities))) + EPSILON)
    dydt = np.concatenate([velocities.flatten(), accelerations.flatten()]) * tau_factor
    return dydt

# --- Solver ---
def solve_zeinel_model(t_span, y0, masses, use_relativity=True, use_quantum=True, steps=10000):
    sol = solve_ivp(
        fun=lambda t, y: zeinel_dynamics(t, y, masses, use_relativity, use_quantum),
        t_span=t_span,
        y0=y0,
        method='RK45',
        t_eval=np.linspace(t_span[0], t_span[1], steps),
        rtol=1e-10,
        atol=1e-12
    )
    return sol

# --- Save to CSV and .npy ---
def save_to_csv(sol, filename="zeinel_simulation.csv"):
    times = sol.t
    positions = sol.y[:9].reshape(3, 3, -1)
    data = []
    for i in range(len(times)):
        row = {
            "time": times[i],
            "x1": positions[0,0,i], "y1": positions[0,1,i], "z1": positions[0,2,i],
            "x2": positions[1,0,i], "y2": positions[1,1,i], "z2": positions[1,2,i],
            "x3": positions[2,0,i], "y3": positions[2,1,i], "z3": positions[2,2,i]
        }
        data.append(row)
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def save_to_npy(sol, filename="zeinel_simulation.npy"):
    np.save(filename, sol.y)

# --- Energy Conservation ---
def compute_energy(positions, velocities, masses):
    kinetic = sum(0.5 * masses[i] * np.linalg.norm(v)**2 for i, v in enumerate(velocities))
    potential = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            r_ij = np.linalg.norm(positions[i] - positions[j]) + EPSILON
            potential -= G * masses[i] * masses[j] / r_ij
    return kinetic + potential, kinetic, potential

def plot_energy_evolution(sol, masses):
    energies = []
    times = sol.t
    positions = sol.y[:9].reshape(3, 3, -1)
    velocities = sol.y[9:].reshape(3, 3, -1)

    for i in range(len(times)):
        pos_i = positions[:, :, i]
        vel_i = velocities[:, :, i]
        total_energy, _, _ = compute_energy(pos_i, vel_i, masses)
        energies.append(total_energy)

    plt.figure(figsize=(10, 5))
    plt.plot(times, energies, label='Total Energy')
    plt.xlabel("Time [yr]")
    plt.ylabel("Energy")
    plt.title("Energy Conservation Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Lyapunov Estimation ---
def lyapunov_estimate(t_span, y0, masses, perturbation=1e-10, steps=1000):
    y0_perturbed = y0.copy()
    y0_perturbed[0] += perturbation

    sol_base = solve_ivp(
        lambda t, y: zeinel_dynamics(t, y, masses), t_span, y0,
        t_eval=np.linspace(*t_span, steps), rtol=1e-10, atol=1e-12
    )
    sol_perturbed = solve_ivp(
        lambda t, y: zeinel_dynamics(t, y, masses), t_span, y0_perturbed,
        t_eval=np.linspace(*t_span, steps), rtol=1e-10, atol=1e-12
    )

    distances = []
    times = sol_base.t
    for i in range(len(times)):
        d = np.linalg.norm(sol_base.y[:9, i] - sol_perturbed.y[:9, i])
        distances.append(d)

    lyapunov_exponents = np.log(np.array(distances) / perturbation) / times

    plt.figure()
    plt.semilogy(times, lyapunov_exponents)
    plt.title("Lyapunov Exponent Estimation")
    plt.xlabel("Time [yr]")
    plt.ylabel("Lyapunov Exponent")
    plt.grid()
    plt.show()

# --- Classical vs Modified Comparison ---
def compare_classical_vs_modified(masses, y0, t_span):
    sol_classical = solve_ivp(
        lambda t, y: zeinel_dynamics(t, y, masses, False, False),
        t_span, y0, t_eval=np.linspace(*t_span, 10000)
    )
    sol_modified = solve_ivp(
        lambda t, y: zeinel_dynamics(t, y, masses, True, True),
        t_span, y0, t_eval=np.linspace(*t_span, 10000)
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def plot_sol(sol, color, label):
        pos = sol.y[:9].reshape(3, 3, -1)
        for i in range(3):
            ax.plot(pos[i,0], pos[i,1], pos[i,2], color=color, label=f"{label} Body {i+1}")

    plot_sol(sol_classical, 'gray', "Classical")
    plot_sol(sol_modified, 'blue', "Modified")

    ax.set_title("Comparison: Classical vs Zeinel Dynamics")
    plt.legend()
    plt.show()

# --- Visualization ---
def plot_zeinel_trajectories(sol):
    positions = sol.y[:9].reshape(3, 3, -1)
    colors = ['crimson', 'royalblue', 'darkgreen']
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(3):
        ax.plot(positions[i,0], positions[i,1], positions[i,2], color=colors[i], label=f'Body {i+1}')
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    ax.set_title('Zeinel Ω-Closed Relativistic Quantum Three-Body Trajectories')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    # Masses in solar masses
    masses = np.array([1.0, 3.00346712e-6, 3.22790300e-7])  # Sun, Earth, Mars

    # Initial positions in AU
    r0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.523, 0.0]
    ])

    # Initial velocities in AU/year
    v0 = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 2 * np.pi, 0.0],
        [-2.0 * np.pi / np.sqrt(1.523), 0.0, 0.0]
    ])

    y0 = np.concatenate([r0.flatten(), v0.flatten()])
    time_span = (0, 10.0)

    print("Running simulation...")
    solution = solve_zeinel_model(time_span, y0, masses)
    print("Simulation complete.")

    # --- Save results ---
    save_to_csv(solution)
    save_to_npy(solution)

    # --- Plotting ---
    plot_zeinel_trajectories(solution)
    plot_energy_evolution(solution, masses)

    # --- Lyapunov ---
    lyapunov_estimate((0, 1), y0, masses)

    # --- Comparison ---
    compare_classical_vs_modified(masses, y0, (0, 5))
