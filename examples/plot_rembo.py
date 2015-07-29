"""
=====================
Illustration of REMBO
=====================

Compare Bayesian Optimization and Random EMbedding Bayesian Optimization
(REMBO) on function with many dimensions but a low effective
dimensionality (2d). In this low-dimensional subspace, the Branin-Hoo
function needs to be optimized.
"""

import numpy as np
from scipy.optimize import rosen
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

from bayesian_optimization import (REMBOOptimizer, BayesianOptimizer,
    GaussianProcessModel, UpperConfidenceBound)

n_dims = 20
n_embedding_dims = 2
n_repetitions = 15
n_trials = 100
kappa = 2.5
colors = {"rembo": "g", "bo": "r"}
for it in range(n_repetitions):
    ind = np.random.RandomState(it).choice(n_dims, 2, replace=False)
    def f(X):  # target function (branin-hoo)
        a = 1
        b = 5.1 / (4*np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8*np.pi)
        x1 = X[ind[0]]
        x2 = X[ind[1]]
        return -(a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1- t)*np.cos(x1) + s) \
            + 0.397887 # Adjust the optimal value to be 0.

    for name in ["rembo", "bo"]:
        # Configuration
        if name == "rembo":
            kernel = C(1.0, (0.01, 1000.0)) \
                * Matern(l=1.0, l_bounds=[(0.001, 100)])
            model = GaussianProcessModel(kernel=kernel)
            acquisition_function = UpperConfidenceBound(model, kappa=kappa)
            opt = REMBOOptimizer(
                n_dims=n_dims, n_embedding_dims=n_embedding_dims, model=model,
                acquisition_function=acquisition_function, optimizer="direct",
                random_state=it)
        else:
            kernel = C(1.0, (0.01, 1000.0)) \
                * Matern(l=[1.0] * n_dims, l_bounds=[(0.001, 100)] * n_dims)
            model = GaussianProcessModel(kernel=kernel)
            acquisition_function = UpperConfidenceBound(model, kappa=kappa)
            opt = BayesianOptimizer(model=model,
                                    acquisition_function=acquisition_function,
                                    optimizer="direct", random_state=it)

        # Perform trials
        for i in range(n_trials):
            X_query = \
                opt.select_query_point(boundaries=np.array([[-5, 15]]*n_dims))
            y_query = f(X_query)
            opt.update(X_query, y_query)

        plt.plot(np.maximum.accumulate(opt.y_), c=colors[name],
                 label=name if it == 0 else "")

plt.yscale("symlog", linthreshy=1e-10)
plt.legend(loc="best")
plt.show()
