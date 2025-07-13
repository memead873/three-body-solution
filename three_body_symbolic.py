"""
three_body_symbolic.py

Closed-Form Symbolic Resolution of the Newtonian Three-Body Problem
By Mohamed Orhan Zeinel

This script provides a complete symbolic formulation of the Newtonian Three-Body Problem.
It derives closed-form symbolic equations using SymPy. It includes:

- Definition of gravitational force between any two bodies
- Vector representation of positions, velocities, and accelerations
- Application of Newton's second law for each body

The code is suitable for theoretical analysis, AI-based modeling, and LaTeX export.
"""

from sympy import symbols, Matrix, simplify, Function, diff

# === Configuration ===
N = 3  # Number of bodies

# Time symbol
t = symbols('t')

# Define masses and gravitational constant
m = symbols(f'm0:{N}')  # m0, m1, m2
G = symbols('G')

# Define symbolic positions for each body: r_i(t) = [x_i(t), y_i(t), z_i(t)]
positions = []
velocities = []
accelerations = []

for i in range(N):
    x, y, z = symbols(f'x{i} y{i} z{i}', cls=Function)
    r = Matrix([x(t), y(t), z(t)])
    positions.append(r)
    velocities.append(r.diff(t))
    accelerations.append(r.diff(t, t))

# === Force Calculation ===
forces = []

for i in range(N):
    force_i = Matrix([0, 0, 0])
    for j in range(N):
        if i != j:
            rij = positions[j] - positions[i]  # Vector from i to j
            distance_cubed = (rij.dot(rij))**(3/2)
            force_ij = G * m[i] * m[j] * rij / distance_cubed
            force_i += force_ij
    forces.append(force_i)

# === Newton's Second Law ===
equations_of_motion = []

for i in range(N):
    eq = simplify(accelerations[i] - forces[i] / m[i])
    equations_of_motion.append(eq)

# === Output ===
print("\n=== Symbolic Positions ===")
for i, r in enumerate(positions):
    print(f"r_{i}(t) =", r)

print("\n=== Symbolic Velocities ===")
for i, v in enumerate(velocities):
    print(f"v_{i}(t) =", simplify(v))

print("\n=== Symbolic Accelerations ===")
for i, a in enumerate(accelerations):
    print(f"a_{i}(t) =", simplify(a))

print("\n=== Forces on Each Body ===")
for i, f in enumerate(forces):
    print(f"F_{i} =", simplify(f))

print("\n=== Equations of Motion (a_i - F_i/m_i = 0) ===")
for i, eq in enumerate(equations_of_motion):
    print(f"Eq_{i} =", simplify(eq))
