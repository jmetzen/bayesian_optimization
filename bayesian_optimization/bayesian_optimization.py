# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>

import numpy as np

from sklearn.utils import check_random_state

from .utils.optimization import global_optimization


class BayesianOptimizer(object):
    """Bayesian optimization for global black-box optimization

    Bayesian optimization models the landscape of the function to be optimized
    internally by a surrogate model (typically a Gaussian process) and
    evaluates always those parameters which are considered as global optimum
    of an acquisition function defined over this surrogate model. Different
    acquisition functions and optimizers can be used internally.

    Bayesian optimization aims at reducing the number of evaluations of the
    actual function, which is assumed to be costly. To achieve this, a large
    computational budget is allocated at modelling the true function and finding
    potentially optimal positions based on this model.

    .. seealso:: Brochu, Cora, de Freitas
                 "A tutorial on Bayesian optimization of expensive cost
                  functions, with application to active user modelling and
                  hierarchical reinforcement learning"

    Parameters
    ----------
    model : surrogate model object
        The surrogate model which is used to model the objective function. It
        needs to provide a methods fit(X, y) for training the model and
        predictive_distribution(X) for determining the predictive distribution
        (mean, std-dev) at query point X.

    acquisition_function : acquisition function object
        When called, this function returns the acquisitability of a query point
        i.e., how favourable it is to perform an evaluation at the query point.
        For this, internally the trade-off between exploration and exploitation
        is handled.

    optimizer: string, default: "direct"
        The optimizer used to identify the maximum of the acquisition function.
        The optimizer is specified by a string which may be any of "direct",
        "direct+lbfgs", "random", "random+lbfgs", "cmaes", or "cmaes+lbfgs".

    maxf: int, default: 1000
        The maximum number of evaluations of the acquisition function by the
        optimizer.

    initial_random_samples: int, default: 5
        The number of initial sample, in which random query points are selected
        without using the acquisition function. Setting this to values larger
        than 0 might be required if the surrogate model needs to be trained
        on datapoints before evaluating it.

    random_state : RandomState or int (default: None)
        Seed for the random number generator.
    """
    def __init__(self, model, acquisition_function, optimizer="direct",
                 maxf=1000, initial_random_samples=5, random_state=0):
        self.model = model
        self.acquisition_function = acquisition_function
        self.optimizer = optimizer
        self.maxf = maxf
        self.initial_random_samples = initial_random_samples

        self.rng = check_random_state(random_state)

        self.X_ = []
        self.y_ = []

    def select_query_point(self, boundaries,
                           incumbent_fct=lambda y: np.max(y)):
        """ Select the next query point in boundaries based on acq. function.

        Parameters
        ----------
        boundaries : ndarray-like, shape: [n_dims, 2]
            Box constraint on allowed query points. First axis corresponds
            to dimensions of the search space and second axis to minimum and
            maximum allowed value in the respective dimensions.

        incumbent_fct: function, default: returns maximum observed value
            A function which is used to determine the incumbent for the
            acquisition function. Defaults to the maximum observed value.
        """
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

                incumbent = incumbent_fct(self.y_)
                return self.acquisition_function(x, incumbent=incumbent)

            X_query = global_optimization(
                objective_function, boundaries=boundaries,
                optimizer=self.optimizer, maxf=self.maxf, random=self.rng)

        # Clip to hard boundaries
        return np.clip(X_query, boundaries[:, 0], boundaries[:, 1])

    def update(self, X, y):
        """ Update internal model for observed (X, y) from true function. """
        self.X_.append(X)
        self.y_.append(y)
        self.model.fit(self.X_, self.y_)

    def best_params():
        """ Returns the best parameters found so far."""
        return self.X_[np.argmax(self.y_)]

    def best_value():
        """ Returns the optimal value found so far."""
        return np.max(self.y_)
