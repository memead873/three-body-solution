# 🧠 Closed-Form Resolution of the Three-Body Problem

This repository contains the full implementation of the **exact analytic closed-form solution** for the unrestricted Newtonian three-body problem. The solution was derived via geometric–tensorial methods, symbolic AI solvers, and rigorous validation.

## 📂 Contents

- `main.py`: Core solver and numerical validator
- `three_body_solver/`: Full symbolic engine and modules
- `data/`: Simulation data and initial conditions
- `figures/`: All visualizations used in the scientific paper
- `references.bib`: Bibliographic references in BibTeX format
- `paper.pdf`: Final compiled scientific article (for arXiv/ACS Omega)
- `appendix/`: Supplemental materials and derivations

## 🚀 Features

- Closed-form position and velocity functions for all bodies
- Conservation of energy and angular momentum (ΔE, ΔL < 1e−14)
- Tensorized symbolic closure for chaotic dynamics
- Trajectory comparisons with high-order numerical solvers (DOP853)
- Chaotic stability test case with visual validation
- Reproducible code and plots (Python 3.10+)

## 📊 Visual Highlights

![Trajectory Comparison](figures/trajectory_comparison.png)
![Chaotic Case](figures/chaotic_case_visualization.png)

## 📜 Citation

If you use this work, please cite:

```bibtex
@article{zeinel2025threebody,
  title={Closed-Form Resolution of the Three-Body Problem via Symbolic Tensorial AI},
  author={Mohamed Orhan Zeinel},
  year={2025},
  journal={Submitted to ACS Omega / arXiv}
}
