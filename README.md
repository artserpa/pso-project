# Particle Swarm Optimization (PSO) in Python

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python implementation of Particle Swarm Optimization (PSO) supporting any number of dimensions, logarithmic scaling for parameters with different magnitudes, early stopping based on convergence, and visualization of particle trajectories and contour plots for multi dimensions optimization problems.

---

## Features

- Any number of dimensions supported (2D, 3D, and higher).
- Logarithmic scaling for parameters with vastly different orders of magnitude.
- Early stopping if the best value does not improve over several iterations.
- Visualization of particle trajectories and contour plots.
- Configurable PSO parameters: number of particles, maximum iterations, inertia weight, cognitive and social coefficients.
- Legends and colorbars organized for clear visualization.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/pso-project.git
cd pso-project
pip install -r requirements.txt\
```

---

## Usage 

Edit main.py to define your own objective function and parameter bounds.
