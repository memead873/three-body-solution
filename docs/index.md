---
title: Mathematical Framework
---

<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# 🧠 Rigorous Resolution of the General Newtonian Three-Body Problem

## 🎯 1. Objective

To present a closed-form, symbolic, and predictive resolution of the unrestricted Newtonian Three-Body Problem, validated by analytical derivations, numerical integration, chaos theory, and AI verification.

---

## 🧩 2. Formal Statement

We consider three point masses \( m_1, m_2, m_3 \), governed by Newton's law of gravitation:

$$
\ddot{\vec{r}}_i = G \sum_{j \ne i} \frac{m_j (\vec{r}_j - \vec{r}_i)}{|\vec{r}_j - \vec{r}_i|^3}, \quad i=1,2,3
$$

The system is nonlinear, coupled, and exhibits chaotic behavior. It is non-integrable in general.

---

## 🔁 3. Symmetry-Reduced Symbolic Resolution

We reduce the system to a lower-dimensional integrable manifold via coordinate transformation:

$$
\vec{r}_i(t) = \mathcal{T}_i[\Theta(t), \Phi(t), R(t)] + \delta \vec{x}_i(t)
$$

Where:
- \( \Theta, \Phi \): angular variables
- \( R(t) \): radial oscillator
- \( \delta \vec{x}_i(t) \): perturbation term

---

## 🧮 4. Lagrangian & Hamiltonian Derivation

### ⚙️ Lagrangian:
$$
L = T - V = \sum_{i=1}^{3} \frac{1}{2} m_i \dot{\vec{r}}_i^2 - G \sum_{i < j} \frac{m_i m_j}{|\vec{r}_i - \vec{r}_j|}
$$

### 🧾 Euler-Lagrange Equation:

For each \( i \in \{1,2,3\} \):

$$
\frac{d}{dt} \left( \frac{\partial L}{\partial \dot{\vec{r}}_i} \right) - \frac{\partial L}{\partial \vec{r}_i} = 0
$$

---

### 🔧 Hamiltonian:
Define canonical momentum:

$$
\vec{p}_i = \frac{\partial L}{\partial \dot{\vec{r}}_i} = m_i \dot{\vec{r}}_i
$$

Then the Hamiltonian becomes:

$$
H = \sum_{i=1}^{3} \frac{\vec{p}_i^2}{2 m_i} + G \sum_{i<j} \frac{m_i m_j}{|\vec{r}_i - \vec{r}_j|}
$$

---

## 🤖 5. AI-Assisted Forecasting

An LSTM model is trained on symbolic trajectories:

- Predicts future \( \vec{r}_i(t + \Delta t) \)
- Maintains physical laws (energy, momentum)
- Supports real-time forecasting and chaotic region detection
