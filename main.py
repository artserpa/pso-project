from pso import PSO
import numpy as np

# -----------------------
# Definition of the objective function
# -----------------------
def rastrigin_3d(x):
    A = 10
    n = len(x)
    return A*n + sum(x**2 - A*np.cos(2*np.pi*x))

# -----------------------
# Boundareis for each parameter
# -----------------------
bounds = [(1e-3, 1e3), (1, 1e5), (0.01, 10), (0.01, 10)]

# -----------------------
# PSO configuration
# w=0.7, c1=1.5, c2=1.5 as default
# -----------------------
pso = PSO(
    objective_function=rastrigin_3d,
    dim=len(bounds),
    bounds=bounds,
    num_particles=50,
    max_iter=100
)

# -----------------------
# PSO execution
# tolerance and patience parameters work for the adaptatice early
# stop if the minimum number of iterations has been reached
# -----------------------
best_pos, best_val = pso.optimize(tol=1e-6, patience=10, min_iter=20)

print("\nOptimized position: ", best_pos)
print("Best objective function value: ", best_val)

# -----------------------
# Convergence and trajectories plots
# -----------------------
pso.plot_metrics()
