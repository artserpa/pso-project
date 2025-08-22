import numpy as np
import matplotlib.pyplot as plt
import itertools

class Particle:
    def __init__(self, dim, log_bounds):
        self.dim = dim
        self.log_bounds = log_bounds
        self.log_position = np.random.uniform(log_bounds[:, 0], log_bounds[:, 1], dim)
        self.log_velocity = np.random.uniform(-1, 1, dim)
        self.best_log_position = np.copy(self.log_position)
        self.best_value = float("inf")

    @property
    def position(self):
        return 10 ** self.log_position

    def evaluate(self, objective_function):
        value = objective_function(self.position)
        if value < self.best_value:
            self.best_value = value
            self.best_log_position = np.copy(self.log_position)
        return value

class PSO:
    def __init__(self, objective_function, dim, bounds, num_particles=30, max_iter=100, 
                 w=0.7, c1=1.5, c2=1.5):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.log_bounds = np.log10(self.bounds)
        self.swarm = [Particle(dim, self.log_bounds) for _ in range(num_particles)]

        self.global_best_log_position = None
        self.global_best_value = float("inf")
        self.history = []
        self.best_path = []

    @property
    def global_best_position(self):
        return 10 ** self.global_best_log_position

    def optimize(self, tol=1e-6, patience=10, min_iter=20):
        no_improve_counter = 0
        last_best = float("inf")
        for iteration in range(self.max_iter):
            for particle in self.swarm:
                value = particle.evaluate(self.objective_function)
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_log_position = np.copy(particle.log_position)
                    self.best_path.append(particle.position)

            for particle in self.swarm:
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (particle.best_log_position - particle.log_position)
                social = self.c2 * r2 * (self.global_best_log_position - particle.log_position)
                particle.log_velocity = self.w * particle.log_velocity + cognitive + social
                particle.log_position = particle.log_position + particle.log_velocity
                particle.log_position = np.clip(particle.log_position, self.log_bounds[:,0], self.log_bounds[:,1])

            self.history.append(self.global_best_value)
            print(f"Iteration {iteration+1}/{self.max_iter}, Objective function value = {self.global_best_value:.6f}")

            # Adaptative early stop
            # Stops the simulation if there is no significant change

            if iteration >= min_iter:
                if abs(last_best - self.global_best_value) < tol:
                    no_improve_counter += 1
                else:
                    no_improve_counter = 0
                last_best = self.global_best_value
                if no_improve_counter >= patience:
                    print(f"\nConverged after {iteration+1} iterations.")
                    break

        return self.global_best_position, self.global_best_value

    def plot_metrics(self):
        path = np.array(self.best_path)

        # Convergence plot

        plt.figure(figsize=(6.4, 4.8))
        plt.plot(self.history, marker='o', linewidth=1.5)
        plt.title("PSO convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.show()

        # Trajectories plot

        if self.dim == 2:
            x = np.linspace(self.bounds[0,0], self.bounds[0,1], 200)
            y = np.linspace(self.bounds[1,0], self.bounds[1,1], 200)
            X, Y = np.meshgrid(x, y)
            Z = np.array([self.objective_function(np.array([xx, yy]))
                          for xx, yy in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

            plt.figure(figsize=(6.4, 4.8))
            plt.contourf(X, Y, Z, levels=50, cmap="viridis")
            plt.contour(X, Y, Z, levels=20, colors="black", alpha=0.3)
            plt.plot(path[:,0], path[:,1], 'r-', label="Trajectory")
            plt.scatter(path[0,0], path[0,1], color="blue", marker="s", s=100, label="Start")
            plt.scatter(path[-1,0], path[-1,1], color="green", marker="*", s=150, label="Optimized")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title("Best particle's trajectory")
            plt.legend(loc="upper right")
            plt.show()

        elif self.dim > 2:
            pairs = list(itertools.combinations(range(self.dim),2))
            n_pairs = len(pairs)
            ncols = 3
            nrows = int(np.ceil(n_pairs / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols,5*nrows))
            axes = np.array(axes).reshape(-1)

            for idx,(i,j) in enumerate(pairs):
                ax = axes[idx]

                xi = np.logspace(np.log10(self.bounds[i,0]), np.log10(self.bounds[i,1]), 150)
                yj = np.logspace(np.log10(self.bounds[j,0]), np.log10(self.bounds[j,1]), 150)
                X,Y = np.meshgrid(xi, yj)

                fixed = np.copy(self.global_best_position)
                Z = np.zeros_like(X)
                for a in range(X.shape[0]):
                    for b in range(X.shape[1]):
                        point = np.copy(fixed)
                        point[i] = X[a,b]
                        point[j] = Y[a,b]
                        Z[a,b] = self.objective_function(point)

                cf = ax.contourf(X,Y,Z,levels=30,cmap="viridis")
                ax.contour(X,Y,Z,levels=15,colors="black",alpha=0.3)
                ax.plot(path[:,i], path[:,j], 'r-', label="Trajectory")
                ax.scatter(path[0,i], path[0,j], color="blue", marker="s", s=80, label="Start")
                ax.scatter(path[-1,i], path[-1,j], color="green", marker="*", s=120, label="Optimized")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(f"x{i+1}")
                ax.set_ylabel(f"x{j+1}")
                ax.set_title(f"Trajectory (x{i+1}, x{j+1})")

            for k in range(idx+1,len(axes)):
                fig.delaxes(axes[k])

            plt.tight_layout(rect=[0,0.15,1,1])
            # legenda das partículas à esquerda
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.05, 0.05), ncol=3, frameon=False)
            # colorbar à direita
            cbar_ax = fig.add_axes([0.55, 0.05, 0.4, 0.03])
            cbar = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal")
            cbar.set_label("f(x)")
            plt.show()
