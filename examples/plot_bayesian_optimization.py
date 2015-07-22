
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

from bayesian_optimization import (BayesianOptimizer, GaussianProcessModel,
    UpperConfidenceBound)

kernel = C(1.0, (0.01, 1000.0)) * Matern(l=1.0, l_bounds=[(0.01, 100)])
model = GaussianProcessModel(kernel=kernel)
kappa = 5.0
acquisition_function = UpperConfidenceBound(model, kappa=kappa)

bayes_opt = BayesianOptimizer(model=model,
                              acquisition_function=acquisition_function,
                              optimizer="direct")

def f(X):
    return -np.linalg.norm(X)

for i in range(10):
    X_query = bayes_opt.select_query_point(boundaries=np.array([[-1, 1]]))
    y_query = f(X_query)
    bayes_opt.update(X_query, y_query)

X_ = np.linspace(-1, 1, 100)[:, None]
y_pred, y_std = bayes_opt.model.predictive_distribution(X_)
plt.plot(X_[:, 0], y_pred, c='b', label="GP mean")
plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                   color='b', alpha=0.3, label="GP mean+-std")
plt.plot(X_[:, 0], y_pred + kappa * y_std, color='r',
           label="UCB(kappa=%s)" % kappa)
plt.scatter(bayes_opt.X_, bayes_opt.y_)
plt.legend(loc="best")
plt.xlabel("Data space")
plt.xlabel("Target value")
plt.show()
