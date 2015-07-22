
import numpy as np

from .utils.optimization import global_optimization


class BayesianOptimizer(object):

    def __init__(self, model, acquisition_function, optimizer, maxf=1000,
                 initial_random_samples=5, seed=0):
        self.model = model
        self.acquisition_function = acquisition_function
        self.optimizer = optimizer
        self.maxf = maxf
        self.initial_random_samples = initial_random_samples

        self.rng = np.random.RandomState(seed)

        self.X_ = []
        self.y_ = []

    def select_query_point(self, boundaries):
        boundaries = np.asarray(boundaries)

        if len(self.X_) < self.initial_random_samples:
            X_query = self.rng.uniform(size=boundaries.shape[0]) \
                * (boundaries[:, 1] - boundaries[:, 0]) + boundaries[:, 0]
        else:
            def objective_function(x):
                # Check boundaries
                if not np.all(np.logical_and(x >= boundaries[:, 0],
                                             x <= boundaries[:, 1])):
                    return -np.inf

                return self.acquisition_function(x, baseline_value=max(self.y_))

            X_query = global_optimization(
                objective_function, boundaries=boundaries,
                optimizer=self.optimizer, maxf=self.maxf, random=self.rng)

        # Clip to hard boundaries
        return np.clip(X_query, boundaries[:, 0], boundaries[:, 1])

    def update(self, X, y):
        self.X_.append(X)
        self.y_.append(y)
        self.model.fit(self.X_, self.y_)
