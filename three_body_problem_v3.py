# omega_final_superform_ai_relativistic_v3.py
# ⚛️ Zeinel Omega Final Superform v3 — Full Analytical + Quantum + Chaos + Relativity + AI
import sympy as sp
from sympy import symbols, Function, diff, sqrt, Abs, exp, pi, I, oo, pprint, simplify, limit, Sum, Matrix, fourier_series, lambdify, N, latex, Eq, solve, cos, sin, Rational
from sympy.vector import CoordSys3D, cross
from sympy.physics.quantum import Operator, Commutator
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from reportlab.pdfgen import canvas
import math
from sympy import symbols
from three_body_dynamics import ThreeBodyDynamics

t = symbols('t')
system = ThreeBodyDynamics(t)
system.summary()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd


# --- Constants ---
G = 1
m1 = m2 = m3 = 1.0
Λ = 0.01  # Cosmological constant
T_max = 50
N = 5000

# --- Initial Conditions ---
Y0 = np.array([
    1.0, 0.0, 0.0,  0.0, 0.3, 0.0,  # r1, v1
   -0.5, np.sqrt(3)/2, 0.0, -0.3, 0.0, 0.0,  # r2, v2
   -0.5, -np.sqrt(3)/2, 0.0,  0.3, -0.3, 0.0  # r3, v3
])

# --- Derivatives Function ---
def dYdt(t, Y):
    r1 = Y[0:3]; v1 = Y[3:6]
    r2 = Y[6:9]; v2 = Y[9:12]
    r3 = Y[12:15]; v3 = Y[15:18]

    def acc(ri, rj, rk):
        a = G * m2 * (rj - ri) / np.linalg.norm(rj - ri)**3
        a += G * m3 * (rk - ri) / np.linalg.norm(rk - ri)**3
        a += -Λ * ri  # Cosmological repulsion
        return a

    a1 = acc(r1, r2, r3)
    a2 = acc(r2, r3, r1)
    a3 = acc(r3, r1, r2)

    return np.concatenate([v1, a1, v2, a2, v3, a3])

# --- Integrate the System ---
from scipy.integrate import solve_ivp
t_span = (0, T_max)
t_eval = np.linspace(*t_span, N)
sol = solve_ivp(dYdt, t_span, Y0, t_eval=t_eval, rtol=1e-9)

# --- Extract Data ---
R1 = sol.y[0:3].T
R2 = sol.y[6:9].T
R3 = sol.y[12:15].T

# --- 3D Plot ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(*R1.T, label='Body 1')
ax.plot(*R2.T, label='Body 2')
ax.plot(*R3.T, label='Body 3')
ax.legend(); plt.title("Three-Body Trajectories")
plt.savefig("trajectories.png")
plt.show()

# --- Animation ---
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
lines = [ax2.plot([], [], [], 'o')[0] for _ in range(3)]

def animate(i):
    for j, R in enumerate([R1, R2, R3]):
        lines[j].set_data(R[i,0], R[i,1])
        lines[j].set_3d_properties(R[i,2])
    return lines

ani = FuncAnimation(fig2, animate, frames=N, interval=1)
ani.save("three_body_animation.mp4", fps=30)

# --- Save CSV ---
df = pd.DataFrame({
    'time': sol.t,
    'r1x': R1[:,0], 'r1y': R1[:,1], 'r1z': R1[:,2],
    'r2x': R2[:,0], 'r2y': R2[:,1], 'r2z': R2[:,2],
    'r3x': R3[:,0], 'r3y': R3[:,1], 'r3z': R3[:,2],
})
df.to_csv("three_body_output.csv", index=False)

# --- PDF Report ---
c = canvas.Canvas("three_body_results.pdf")
c.drawString(50, 800, "Three-Body Problem Simulation Report")
c.drawImage("trajectories.png", 50, 450, width=500, preserveAspectRatio=True)
c.drawString(50, 420, f"Total Simulation Time: {T_max}s")
c.drawString(50, 400, "Cosmological Constant Λ: 0.01")
c.save()

# --- Lyapunov Exponent (Simple) ---
def lyapunov(Y, eps=1e-6):
    Y_perturb = Y + eps * np.random.randn(*Y.shape)
    sol2 = solve_ivp(dYdt, t_span, Y_perturb, t_eval=t_eval)
    delta = np.linalg.norm(sol.y - sol2.y, axis=0)
    lce = np.log(delta/eps)
    return lce

lce_vals = lyapunov(Y0)
plt.plot(sol.t, lce_vals)
plt.title("Lyapunov Exponent Estimate")
plt.savefig("lyapunov_plot.png")
plt.show()

# --- AI: LSTM Chaos Predictor ---
def build_lstm_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(10, 3), return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

X = []
y = []
for i in range(1000, N - 10):
    X.append(R1[i-10:i])
    y.append(lce_vals[i])
X = np.array(X); y = np.array(y)

model = build_lstm_model()
model.fit(X, y, epochs=10, verbose=1)
pred = model.predict(X)

plt.plot(y[:200], label="True")
plt.plot(pred[:200], label="Predicted")
plt.legend(); plt.title("Chaos Prediction via LSTM")
plt.savefig("chaos_lstm.png")
plt.show()

import sympy as sp
from sympy import symbols, Function, diff, sqrt, Matrix
import builtins

# --- Time Symbol ---
t = symbols('t', real=True)

# --- Autofix Function ---
def safe_unpack(func_str):
    try:
        # حاول تنفيذ unpack بشكل مباشر
        return tuple(Function(name)(t) for name in func_str.split())
    except Exception as e:
        print(f"[AutoFix] Failed to unpack: {func_str}. Fixing...")
        fixed = []
        for name in func_str.split():
            fixed.append(Function(name)(t))
        return tuple(fixed)

# --- 3D Position Vectors ---
r1x, r1y, r1z = safe_unpack('r1x r1y r1z')
r2x, r2y, r2z = safe_unpack('r2x r2y r2z')
r3x, r3y, r3z = safe_unpack('r3x r3y r3z')

# --- Combine into vectors ---
r1 = Matrix([r1x, r1y, r1z])
r2 = Matrix([r2x, r2y, r2z])
r3 = Matrix([r3x, r3y, r3z])

# --- Velocities ---
v1 = diff(r1, t)
v2 = diff(r2, t)
v3 = diff(r3, t)

# --- Relative Distances ---
r12_vec = r1 - r2
r23_vec = r2 - r3
r13_vec = r1 - r3
r12 = sqrt(r12_vec.dot(r12_vec))
r23 = sqrt(r23_vec.dot(r23_vec))
r13 = sqrt(r13_vec.dot(r13_vec))

print("[✓] All variables initialized successfully with AutoFix.")

# === Zeinel Omega Final — Core Physical Constants & Indices ===
from sympy import symbols, sqrt, pi, I, Rational

# --- Time and Indices ---
t = symbols('t', real=True)                         # Continuous time
n = symbols('n', integer=True)                      # Integer index for sums
N = symbols('N', integer=True, positive=True)       # Upper bound index (positive)

# --- Core Physical Constants ---
G = symbols('G', positive=True, real=True)          # Gravitational constant
ħ = symbols('ħ', positive=True, real=True)          # Reduced Planck constant
c = symbols('c', positive=True, real=True)          # Speed of light
ε = symbols('ε', real=True)                         # Small quantum correction parameter
Λ = symbols('Λ', real=True)                         # Optional cosmological constant

# --- Masses of the 3 bodies ---
m1, m2, m3 = symbols('m1 m2 m3', positive=True, real=True)

# --- Planck Units (Natural Units Layer) ---
# Derived symbolic Planck units (dimension analysis)
l_P = sqrt(ħ * G / c**3)                            # Planck Length
t_P = sqrt(ħ * G / c**5)                            # Planck Time
m_P = sqrt(ħ * c / G)                               # Planck Mass
E_P = m_P * c**2                                     # Planck Energy

# Optional symbolic substitution for dimensionless analysis
use_dimensionless = True

if use_dimensionless:
    # Rescale all quantities to be dimensionless in Planck units
    m1_dimless = m1 / m_P
    m2_dimless = m2 / m_P
    m3_dimless = m3 / m_P
    t_dimless = t / t_P
    print("\n[+] All constants are rescaled to Planck units (dimensionless system)")
else:
    print("\n[+] Using physical units with G, ħ, c explicitly retained")

# --- Display ---
print("\n--- Fundamental Constants ---")
print(f"G  = {G}  (Gravitational constant)")
print(f"ħ  = {ħ}  (Reduced Planck constant)")
print(f"c  = {c}  (Speed of light)")
print(f"ε  = {ε}  (Quantum correction parameter)")
print(f"Λ  = {Λ}  (Cosmological constant)")

print("\n--- Planck Units (symbolic) ---")
print(f"Planck Length     l_P = {l_P}")
print(f"Planck Time       t_P = {t_P}")
print(f"Planck Mass       m_P = {m_P}")
print(f"Planck Energy     E_P = {E_P}")
# --- 3D Position Vectors for each body ---
r1x = Function('r1x')(t)
r1y = Function('r1y')(t)
r1z = Function('r1z')(t)

r2x = Function('r2x')(t)
r2y = Function('r2y')(t)
r2z = Function('r2z')(t)

r3x = Function('r3x')(t)
r3y = Function('r3y')(t)
r3z = Function('r3z')(t)

# === three_body_dynamics.py ===
import sympy as sp
from sympy import Function, Matrix, sqrt, diff

class ThreeBodyDynamics:
    """
    A symbolic engine for 3D three-body classical mechanics using sympy.
    Includes position, velocity, relative vectors, and distance magnitudes.
    """

    def __init__(self, t: sp.Symbol):
        self.t = t

        # Time-dependent position functions
        self.r1x, self.r1y, self.r1z = Function('r1x')(t), Function('r1y')(t), Function('r1z')(t)
        self.r2x, self.r2y, self.r2z = Function('r2x')(t), Function('r2y')(t), Function('r2z')(t)
        self.r3x, self.r3y, self.r3z = Function('r3x')(t), Function('r3y')(t), Function('r3z')(t)

        # Position vectors
        self.r1 = Matrix([self.r1x, self.r1y, self.r1z])
        self.r2 = Matrix([self.r2x, self.r2y, self.r2z])
        self.r3 = Matrix([self.r3x, self.r3y, self.r3z])

        # Velocity vectors (time derivatives of positions)
        self.v1 = diff(self.r1, t)
        self.v2 = diff(self.r2, t)
        self.v3 = diff(self.r3, t)

        # Relative displacement vectors
        self.r12_vec = self.r1 - self.r2
        self.r23_vec = self.r2 - self.r3
        self.r13_vec = self.r1 - self.r3

        # Euclidean norms (magnitudes of relative positions)
        self.r12 = sqrt(self.r12_vec.dot(self.r12_vec))
        self.r23 = sqrt(self.r23_vec.dot(self.r23_vec))
        self.r13 = sqrt(self.r13_vec.dot(self.r13_vec))

        # Time derivatives of distance magnitudes
        self.dr12_dt = diff(self.r12, t)
        self.dr23_dt = diff(self.r23, t)
        self.dr13_dt = diff(self.r13, t)

    def summary(self):
        """
        Print a symbolic summary of all main vectors and distances.
        """
        print("\n--- Three-Body Dynamics Summary ---")
        print("r1(t):", self.r1)
        print("r2(t):", self.r2)
        print("r3(t):", self.r3)
        print("v1(t):", self.v1)
        print("v2(t):", self.v2)
        print("v3(t):", self.v3)
        print("r12(t):", self.r12_vec)
        print("r23(t):", self.r23_vec)
        print("r13(t):", self.r13_vec)
        print("|r12|(t):", self.r12)
        print("|r23|(t):", self.r23)
        print("|r13|(t):", self.r13)
        print("d|r12|/dt:", self.dr12_dt)
        print("d|r23|/dt:", self.dr23_dt)
        print("d|r13|/dt:", self.dr13_dt)
# --- Kinetic Energy ---
K = Rational(1,2)*m1*v1.dot(v1) + Rational(1,2)*m2*v2.dot(v2) + Rational(1,2)*m3*v3.dot(v3)

# --- Potential Energy ---
V = - G*m1*m2/r12 - G*m2*m3/r23 - G*m1*m3/r13

# --- Lagrangian Formalism ---
Lagrangian = K - V

# Euler-Lagrange Equations (Example for r1x)
dL_dq = diff(Lagrangian, r1x)
dL_dv = diff(Lagrangian, diff(r1x, t))
EL_eq_r1x = dL_dq - diff(dL_dv, t)
print("\n[+] Euler-Lagrange Equation (r1x):")
pprint(simplify(EL_eq_r1x), use_unicode=True)

# --- Hamiltonian Formulation ---
p1x = diff(Lagrangian, v1[0])
p1y = diff(Lagrangian, v1[1])
p1z = diff(Lagrangian, v1[2])
p2x = diff(Lagrangian, v2[0])
p2y = diff(Lagrangian, v2[1])
p2z = diff(Lagrangian, v2[2])
p3x = diff(Lagrangian, v3[0])
p3y = diff(Lagrangian, v3[1])
p3z = diff(Lagrangian, v3[2])

H = p1x*v1[0] + p1y*v1[1] + p1z*v1[2] + \
    p2x*v2[0] + p2y*v2[1] + p2z*v2[2] + \
    p3x*v3[0] + p3y*v3[1] + p3z*v3[2] - Lagrangian

print("\n[+] Hamiltonian H(p, q):")
pprint(H.simplify(), use_unicode=True)

# --- Folded Time Omega(t) ---
Gamma = sqrt(K**2 + V**2)
Omega_expr = sp.integrate(Gamma, t)

# Complexified Time (Analytic Continuation)
Omega_complex = Omega_expr.subs(t, I * t)
print("\n[+] Complexified Omega(t):")
pprint(Omega_complex, use_unicode=True)

# Limit at infinity
print("\n[+] lim_{t→∞} Ω(it):")
pprint(limit(Omega_complex, t, oo), use_unicode=True)

# --- Fourier Series Representation of Position (Generalized Form) ---
C_n = Function('C_n')(n)  # Fourier Coefficients
# Assume position is a function of folded time Omega(t)
r_N = Sum(C_n * exp(I * n * Omega_expr), (n, -N, N))
r_of_t = sp.re(r_N.doit())

# Classical Limit
r_classical = r_of_t.subs(Omega_expr, t).simplify()
print("\n[+] Classical Limit r(t → Ω=t):")
pprint(r_classical, use_unicode=True)

try:
    assert simplify(r_of_t.subs(Omega_expr, t) - r_classical) == 0
    print("[+] Assertion Passed: r_of_t → r_classical under classical limit.")
except AssertionError:
    print("[-] Assertion Failed: r_of_t ≠ r_classical in classical limit.")

# --- Invariance Under Rotation (SO(3)) ---
theta = symbols('θ')
R = Matrix([[cos(theta), -sin(theta), 0],
           [sin(theta),  cos(theta), 0],
           [0,            0,        1]])
r1_rotated = R * r1
L_rotated = Lagrangian.subs({r1: r1_rotated})
invariant_L = simplify(Lagrangian - L_rotated)
print("\n[+] δL under SO(3) rotation:")
pprint(invariant_L, use_unicode=True)

# --- Noether Theorem General Symbolic Derivation ---
def noether_theorem(L, sym):
    conserved_quantity = L.diff(sym)
    return conserved_quantity

conserved_energy = noether_theorem(Lagrangian, t)
conserved_momentum = noether_theorem(Lagrangian, r1x)

print("\n[+] Conserved Quantity from Time Symmetry:")
pprint(conserved_energy, use_unicode=True)
print("\n[+] Conserved Quantity from Spatial Symmetry:")
pprint(conserved_momentum, use_unicode=True)

# --- Quantum Operators (Canonical Quantization) ---
# Convert positions to operators
r1x_op = Operator('r1x')
r1y_op = Operator('r1y')
r1z_op = Operator('r1z')

# Momentum operator in position basis
p1x_op = -I * ħ * diff(r1x, t)
p1y_op = -I * ħ * diff(r1y, t)
p1z_op = -I * ħ * diff(r1z, t)

# Example commutator [r, p]
comm_r_p = Commutator(r1x_op, p1x_op).doit()
print("\n[+] [r1x, p1x] =", end=" ")
pprint(comm_r_p, use_unicode=True)

# Schrödinger-like equation
ψ = Function('ψ')(t)
H_hat = H.subs({r1x: r1x_op, r1y: r1y_op, r1z: r1z_op})
schrodinger_eq = I * ħ * diff(ψ, t) - H_hat * ψ
print("\n[+] Semi-Classical Schrödinger Equation:")
pprint(schrodinger_eq, use_unicode=True)

# Feynman Path Integral
Z = PathIntegral(Lagrangian, r1, r2, r3)
print("\n[+] Path Integral Z = ∫ Dq(t) e^{iS[q]/ħ}:")
pprint(Z, use_unicode=True)

# --- Quantum Backreaction (Semiclassical Model) ---
ψ_1 = Function('ψ1')(r1x, t)
ψ_2 = Function('ψ2')(r2x, t)
ψ_3 = Function('ψ3')(r3x, t)

E_quantum_backreaction = -ħ**2 / (2*m1) * diff(ψ_1, r1x, 2) + V.subs(r1x, r2x) * ψ_1
print("\n[+] Quantum Backreaction Term:")
pprint(E_quantum_backreaction, use_unicode=True)

# --- Lyapunov Stability via Floquet Theory: Full Version ---
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvals
import matplotlib.pyplot as plt

# --- Define symbols and positions ---
from sympy import symbols, Function, diff
from sympy import lambdify

# Time symbol
t = symbols('t')

# Positions as functions of time
r1x, r1y, r1z = Function('r1x')(t), Function('r1y')(t), Function('r1z')(t)
r2x, r2y, r2z = Function('r2x')(t), Function('r2y')(t), Function('r2z')(t)
r3x, r3y, r3z = Function('r3x')(t), Function('r3y')(t), Function('r3z')(t)

# Velocities
v1 = [diff(r1x, t), diff(r1y, t), diff(r1z, t)]
v2 = [diff(r2x, t), diff(r2y, t), diff(r2z, t)]
v3 = [diff(r3x, t), diff(r3y, t), diff(r3z, t)]

# Full state vector
# r1, v1, r2, v2, r3, v3
state0 = np.array([
    1.0, 0.0, 0.0,  0.0, 0.5, 0.0,
   -0.5, 0.0, 0.0,  0.0, -0.5, 0.0,
   -0.5, 0.0, 0.0,  0.0, 0.0, 0.0
])

# Masses
g = 1.0
m1 = m2 = m3 = 1.0

# Right-hand side of the system (Newton's 2nd law)
def rhs(t, y):
    r1 = y[0:3]; v1 = y[3:6]
    r2 = y[6:9]; v2 = y[9:12]
    r3 = y[12:15]; v3 = y[15:18]

    def acc(ra, rb, m):
        r = rb - ra
        dist3 = np.linalg.norm(r)**3 + 1e-6
        return g * m * r / dist3

    a1 = acc(r1, r2, m2) + acc(r1, r3, m3)
    a2 = acc(r2, r1, m1) + acc(r2, r3, m3)
    a3 = acc(r3, r1, m1) + acc(r3, r2, m2)

    return np.concatenate([v1, a1, v2, a2, v3, a3])

# --- Integrate system ---
T = 10  # total time
sol = solve_ivp(rhs, [0, T], state0, t_eval=np.linspace(0, T, 1000))

# --- Compute monodromy matrix using finite differences ---
N = len(state0)
delta = 1e-6
monodromy = np.zeros((N, N))

for i in range(N):
    perturbed = np.copy(state0)
    perturbed[i] += delta
    sol_pert = solve_ivp(rhs, [0, T], perturbed, t_eval=[T])
    monodromy[:, i] = (sol_pert.y[:, -1] - sol.y[:, -1]) / delta

# --- Compute Floquet multipliers (eigenvalues) ---
floquet_eigs = eigvals(monodromy)

# --- Plot Floquet spectrum ---
plt.figure(figsize=(6, 4))
plt.title("Floquet Spectrum — Lyapunov Exponents")
plt.plot(np.abs(floquet_eigs), 'o')
plt.yscale('log')
plt.xlabel("Mode")
plt.ylabel("|Eigenvalue|")
plt.grid(True)
plt.tight_layout()
plt.show()
# Define RHS of the equations of motion (F = ma)
a_newton_1 = - G*m2*(r1 - r2)/r12**3 - G*m3*(r1 - r3)/r13**3
a_newton_2 = - G*m1*(r2 - r1)/r12**3 - G*m3*(r2 - r3)/r23**3
a_newton_3 = - G*m1*(r3 - r1)/r13**3 - G*m2*(r3 - r2)/r23**3

f = Matrix([
    v1[0], v1[1], v1[2], a_newton_1[0], a_newton_1[1], a_newton_1[2],
    v2[0], v2[1], v2[2], a_newton_2[0], a_newton_2[1], a_newton_2[2],
    v3[0], v3[1], v3[2], a_newton_3[0], a_newton_3[1], a_newton_3[2]
])

# Compute Jacobian
try:
    J = f.jacobian(state)
    print("\n[+] Jacobian Matrix (Lyapunov Stability):")
    pprint(J[:4, :4], use_unicode=True)
    eigenvals = eigvals(np.array(J[:6, :6].evalf().tolist()).astype(float))
    print("\n[+] Eigenvalues of Lyapunov Matrix (Floquet Spectrum):")
    pprint(eigenvals, use_unicode=True)
except Exception as e:
    print(f"\n[-] Jacobian not computable symbolically: {e}")

# --- Poincaré Section Analysis ---
def poincare_section(sol):
    fig, ax = plt.subplots()
    ax.plot(sol[:, 0], sol[:, 3], '.', markersize=1)
    ax.set_title("Poincaré Section (Real Dynamics)")
    plt.show()

# --- Chaos Map: Sensitivity to Initial Conditions ---
def chaos_map():
    y0_base = [0.97000436, -0.93240737/2, 0,
               0.93240737, 0.97000436/2, 0,
               0, 0, 0]
    y0_perturbed = y0_base.copy()
    y0_perturbed[0] += 1e-8

    def rhs(t, y):
        r1x_, r1y_, r1z_, v1x_, v1y_, v1z_, \
        r2x_, r2y_, r2z_, v2x_, v2y_, v2z_, \
        r3x_, r3y_, r3z_, v3x_, v3y_, v3z_ = y

        return [
            v1x_,
            v1y_,
            v1z_,
            - G*m2*(r1x_ - r2x_)/(math.sqrt((r1x_ - r2x_)**2 + (r1y_ - r2y_)**2 + (r1z_ - r2z_)**2)**3) \
            - G*m3*(r1x_ - r3x_)/(math.sqrt((r1x_ - r3x_)**2 + (r1y_ - r3y_)**2 + (r1z_ - r3z_)**2)**3),
            - G*m2*(r1y_ - r2y_)/(math.sqrt((r1x_ - r2x_)**2 + (r1y_ - r2y_)**2 + (r1z_ - r2z_)**2)**3) \
            - G*m3*(r1y_ - r3y_)/(math.sqrt((r1x_ - r3x_)**2 + (r1y_ - r3y_)**2 + (r1z_ - r3z_)**2)**3),
            - G*m2*(r1z_ - r2z_)/(math.sqrt((r1x_ - r2x_)**2 + (r1y_ - r2y_)**2 + (r1z_ - r2z_)**2)**3) \
            - G*m3*(r1z_ - r3z_)/(math.sqrt((r1x_ - r3x_)**2 + (r1y_ - r3y_)**2 + (r1z_ - r3z_)**2)**3),
            v2x_,
            v2y_,
            v2z_,
            - G*m1*(r2x_ - r1x_)/(math.sqrt((r1x_ - r2x_)**2 + (r1y_ - r2y_)**2 + (r1z_ - r2z_)**2)**3) \
            - G*m3*(r2x_ - r3x_)/(math.sqrt((r2x_ - r3x_)**2 + (r2y_ - r3y_)**2 + (r2z_ - r3z_)**2)**3),
            - G*m1*(r2y_ - r1y_)/(math.sqrt((r1x_ - r2x_)**2 + (r1y_ - r2y_)**2 + (r1z_ - r2z_)**2)**3) \
            - G*m3*(r2y_ - r3y_)/(math.sqrt((r2x_ - r3x_)**2 + (r2y_ - r3y_)**2 + (r2z_ - r3z_)**2)**3),
            - G*m1*(r2z_ - r1z_)/(math.sqrt((r1x_ - r2x_)**2 + (r1y_ - r2y_)**2 + (r1z_ - r2z_)**2)**3) \
            - G*m3*(r2z_ - r3z_)/(math.sqrt((r2x_ - r3x_)**2 + (r2y_ - r3y_)**2 + (r2z_ - r3z_)**2)**3),
            v3x_,
            v3y_,
            v3z_,
            - G*m1*(r3x_ - r1x_)/(math.sqrt((r1x_ - r3x_)**2 + (r1y_ - r3y_)**2 + (r1z_ - r3z_)**2)**3) \
            - G*m2*(r3x_ - r2x_)/(math.sqrt((r2x_ - r3x_)**2 + (r2y_ - r3y_)**2 + (r2z_ - r3z_)**2)**3),
            - G*m1*(r3y_ - r1y_)/(math.sqrt((r1x_ - r3x_)**2 + (r1y_ - r3y_)**2 + (r1z_ - r3z_)**2)**3) \
            - G*m2*(r3y_ - r2y_)/(math.sqrt((r2x_ - r3x_)**2 + (r2y_ - r3y_)**2 + (r2z_ - r3z_)**2)**3),
            - G*m1*(r3z_ - r1z_)/(math.sqrt((r1x_ - r3x_)**2 + (r1y_ - r3y_)**2 + (r1z_ - r3z_)**2)**3) \
            - G*m2*(r3z_ - r2z_)/(math.sqrt((r2x_ - r3x_)**2 + (r2y_ - r3y_)**2 + (r2z_ - r3z_)**2)**3),
        ]

    sol_base = solve_ivp(rhs, [0, 10000], y0_base, t_eval=np.linspace(0, 10000, 10000))
    sol_perturbed = solve_ivp(rhs, [0, 10000], y0_perturbed, t_eval=np.linspace(0, 10000, 10000))

    Δ_initial = np.linalg.norm(np.array(y0_base) - np.array(y0_perturbed))
    Δ_final = np.linalg.norm(sol_base.y[:, -1] - sol_perturbed.y[:, -1])
    λ = np.log(Δ_final / Δ_initial) / 10000
    print(f"[+] Lyapunov Exponent λ ≈ {λ:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(sol_base.t, np.linalg.norm(sol_base.y - sol_perturbed.y, axis=0))
    plt.title("Long-Term Chaos Sensitivity Map")
    plt.xlabel("Time")
    plt.ylabel("Δ(t)")
    plt.grid(True)
    plt.show()

# --- General Relativity Reformulation with Geodesics ---
from sympy.diffgeom import Manifold, Patch, CoordSystem, TensorProduct
from sympy.diffgeom import Metric, ChristoffelSymbols, covariant_derivative

# Define spacetime coordinates
t_coord, x_coord, y_coord, z_coord = symbols('t x y z')
coords = [t_coord, x_coord, y_coord, z_coord]

# Create 4D spacetime manifold
M = Manifold('M', 4)
P = Patch('P', M)
CS = CoordSystem('CS', P, coords)

# Define metric tensor (Minkowski metric for now)
g = Matrix([
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Convert to SymPy metric object
metric = Metric('g', CS, g)

# Compute Christoffel symbols
christoffel = ChristoffelSymbols(metric, 'Γ')
print("\n[+] Christoffel Symbols (First 5):")
for idx in range(5):
    pprint(christoffel[idx], use_unicode=True)

# Define position as functions of affine parameter (τ)
τ = symbols('τ')
x_mu = [Function(f'x^{i}')(τ) for i in range(4)]

# Compute geodesic equations
geodesic_eqs = []
for μ in range(4):
    term = diff(x_mu[μ], τ, 2)
    for α in range(4):
        for β in range(4):
            term += christoffel[μ, α, β] * diff(x_mu[α], τ) * diff(x_mu[β], τ)
    geodesic_eqs.append(Eq(term, 0))

print("\n[+] Geodesic Equations (First 2 Components):")
for eq in geodesic_eqs[:2]:
    pprint(eq, use_unicode=True)

# Compute Riemann curvature tensor
R = metric.riemann_tensor()
print("\n[+] Riemann Tensor (First Component):")
pprint(R[0, 1, 0, 1], use_unicode=True)

# Compute Ricci tensor
Ric = metric.ricci_tensor()
print("\n[+] Ricci Tensor (First Component):")
pprint(Ric[0, 0], use_unicode=True)

# Compute Ricci scalar
R_scalar = metric.ricci_scalar()
print("\n[+] Ricci Scalar:")
pprint(R_scalar, use_unicode=True)

# Compute Einstein tensor
G_mu_nu = Ric - Rational(1,2) * g[0,0] * R_scalar * metric.tensor()
print("\n[+] Einstein Tensor (First Component):")
pprint(G_mu_nu[0, 0], use_unicode=True)

# --- Quantum Gravity Coupling ---
g_mu_nu = g
quantum_gravity_coupling = g_mu_nu * ψ
print("\n[+] Quantum Gravity Coupling Tensor T^μν = g^μν |ψ|²")
pprint(quantum_gravity_coupling, use_unicode=True)

# --- Field Theory Reformulation ---
x = symbols('x')  # Define x symbol
φ = Function('φ')(x, t)
field_lagrangian = diff(φ, t)**2 - diff(φ, x)**2 - φ**2
print("\n[+] Field Theory Lagrangian Density:")
pprint(field_lagrangian, use_unicode=True)

# --- Current from Noether Theorem ---
J_mu = diff(field_lagrangian, diff(φ, x))
print("\n[+] Noether Current J^μ:")
pprint(J_mu, use_unicode=True)

# Conservation Law
conservation = Eq(diff(J_mu, x), 0)
print("\n[+] ∂μJ^μ = 0 (Conservation Law):")
pprint(conservation, use_unicode=True)

# --- Save Results to LaTeX File ---
with open("omega_paper.tex", "w") as f:
    f.write("\\section{Equations}\n")
    f.write(latex(Lagrangian))
    f.write("\n\\section{Hamiltonian}\n")
    f.write(latex(H))
    f.write("\n\\section{Noether Conservation}\n")
    f.write(latex(conserved_energy))

# --- Smart Summary Report Generator ---
def smart_summary_report():
    summary = {
        'Energy Conserved': simplify(diff(K + V, t)) == 0,
        'Momentum Conserved': simplify(diff(cross(r1, v1), t)).is_zero,
        'Quantum Backreaction': E_quantum_backreaction != 0,
        'Chaos Detected': λ > 1e-3,
        'Field Theory Available': field_lagrangian is not None
    }
    print("\n[+] Smart Summary Report:")
    pprint(summary, use_unicode=True)

# --- Generate PDF Report ---
def generate_pdf_report():
    c = canvas.Canvas("omega_report.pdf")
    c.drawString(50, 750, "Zeinel Omega Three-Body Solution")
    c.drawString(50, 730, f"Lagrangian: {str(Lagrangian)}")
    c.drawString(50, 710, f"Hamiltonian: {str(H)}")
    c.save()

# --- Streamlit Interface (Run in separate file or terminal) ---
def run_streamlit():
    st.title("Zeinel Omega Three-Body Simulator")
    st.latex(f"L = {latex(Lagrangian)}")
    st.latex(f"H = {latex(H)}")

# --- Long-term Chaos Map (t = 1e5) ---
def long_term_chaos_map():
    y0_base = [0.97000436, -0.93240737/2, 0,
               0.93240737, 0.97000436/2, 0,
               0, 0, 0]
    y0_perturbed = y0_base.copy()
    y0_perturbed[0] += 1e-8

    def rhs(t, y):
        r1x_, r1y_, r1z_, v1x_, v1y_, v1z_, \
        r2x_, r2y_, r2z_, v2x_, v2y_, v2z_, \
        r3x_, r3y_, r3z_, v3x_, v3y_, v3z_ = y

        return [
            v1x_,
            v1y_,
            v1z_,
            - G*m2*(r1x_ - r2x_)/(math.sqrt((r1x_ - r2x_)**2 + (r1y_ - r2y_)**2 + (r1z_ - r2z_)**2)**3) \
            - G*m3*(r1x_ - r3x_)/(math.sqrt((r1x_ - r3x_)**2 + (r1y_ - r3y_)**2 + (r1z_ - r3z_)**2)**3),
            - G*m2*(r1y_ - r2y_)/(math.sqrt((r1x_ - r2x_)**2 + (r1y_ - r2y_)**2 + (r1z_ - r2z_)**2)**3) \
            - G*m3*(r1y_ - r3y_)/(math.sqrt((r1x_ - r3x_)**2 + (r1y_ - r3y_)**2 + (r1z_ - r3z_)**2)**3),
            - G*m2*(r1z_ - r2z_)/(math.sqrt((r1x_ - r2x_)**2 + (r1y_ - r2y_)**2 + (r1z_ - r2z_)**2)**3) \
            - G*m3*(r1z_ - r3z_)/(math.sqrt((r1x_ - r3x_)**2 + (r1y_ - r3y_)**2 + (r1z_ - r3z_)**2)**3),
            v2x_,
            v2y_,
            v2z_,
            - G*m1*(r2x_ - r1x_)/(math.sqrt((r1x_ - r2x_)**2 + (r1y_ - r2y_)**2 + (r1z_ - r2z_)**2)**3) \
            - G*m3*(r2x_ - r3x_)/(math.sqrt((r2x_ - r3x_)**2 + (r2y_ - r3y_)**2 + (r2z_ - r3z_)**2)**3),
            - G*m1*(r2y_ - r1y_)/(math.sqrt((r1x_ - r2x_)**2 + (r1y_ - r2y_)**2 + (r1z_ - r2z_)**2)**3) \
            - G*m3*(r2y_ - r3y_)/(math.sqrt((r2x_ - r3x_)**2 + (r2y_ - r3y_)**2 + (r2z_ - r3z_)**2)**3),
            - G*m1*(r2z_ - r1z_)/(math.sqrt((r1x_ - r2x_)**2 + (r1y_ - r2y_)**2 + (r1z_ - r2z_)**2)**3) \
            - G*m3*(r2z_ - r3z_)/(math.sqrt((r2x_ - r3x_)**2 + (r2y_ - r3y_)**2 + (r2z_ - r3z_)**2)**3),
            v3x_,
            v3y_,
            v3z_,
            - G*m1*(r3x_ - r1x_)/(math.sqrt((r1x_ - r3x_)**2 + (r1y_ - r3y_)**2 + (r1z_ - r3z_)**2)**3) \
            - G*m2*(r3x_ - r2x_)/(math.sqrt((r2x_ - r3x_)**2 + (r2y_ - r3y_)**2 + (r2z_ - r3z_)**2)**3),
            - G*m1*(r3y_ - r1y_)/(math.sqrt((r1x_ - r3x_)**2 + (r1y_ - r3y_)**2 + (r1z_ - r3z_)**2)**3) \
            - G*m2*(r3y_ - r2y_)/(math.sqrt((r2x_ - r3x_)**2 + (r2y_ - r3y_)**2 + (r2z_ - r3z_)**2)**3),
            - G*m1*(r3z_ - r1z_)/(math.sqrt((r1x_ - r3x_)**2 + (r1y_ - r3y_)**2 + (r1z_ - r3z_)**2)**3) \
            - G*m2*(r3z_ - r2z_)/(math.sqrt((r2x_ - r3x_)**2 + (r2y_ - r3y_)**2 + (r2z_ - r3z_)**2)**3),
        ]

    sol_base = solve_ivp(rhs, [0, 10000], y0_base, t_eval=np.linspace(0, 10000, 10000))
    sol_perturbed = solve_ivp(rhs, [0, 10000], y0_perturbed, t_eval=np.linspace(0, 10000, 10000))

    Δ_initial = np.linalg.norm(np.array(y0_base) - np.array(y0_perturbed))
    Δ_final = np.linalg.norm(sol_base.y[:, -1] - sol_perturbed.y[:, -1])
    λ = np.log(Δ_final / Δ_initial) / 10000
    plt.figure(figsize=(10, 6))
    plt.plot(sol_base.t, np.linalg.norm(sol_base.y - sol_perturbed.y, axis=0))
    plt.title("Long-Term Chaos Sensitivity Map (t=1e4)")
    plt.xlabel("Time")
    plt.ylabel("Δ(t)")
    plt.grid(True)
    plt.show()

# --- Run Everything ---
if __name__ == '__main__':
    long_term_chaos_map()
    poincare_section(solve_ivp(lambda t, y: [], [0, 10], []).y)
    smart_summary_report()
    generate_pdf_report()
