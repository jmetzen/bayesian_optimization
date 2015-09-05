# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>

from itertools import cycle

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
                 maxf=1000, initial_random_samples=5, random_state=0,
                 *args, **kwargs):
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
            self.acquisition_function.set_boundaries(boundaries)

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

    def best_params(self):
        """ Returns the best parameters found so far."""
        return self.X_[np.argmax(self.y_)]

    def best_value(self):
        """ Returns the optimal value found so far."""
        return np.max(self.y_)


class REMBOOptimizer(BayesianOptimizer):
    """ Random EMbedding Bayesian Optimization (REMBO).

    This extension of Bayesian Optimizer (BO) is better suited for
    high-dimensional problems with a low effective dimensionality than BO.
    This is achieved by restricting the optimization to a low-dimensional
    linear manifold embedded in the higher dimensional space. Theoretical
    results suggest that even if the manifold is chosen randomly, the
    optimum on this manifold equals the global optimum if the function is
    indeed of the same intrinsic dimensionality as the manifold.

    .. seealso:: Wang, Zoghi, Hutter, Matheson, de Freitas
                 "Bayesian Optimization in High Dimensions via Random
                 Embeddings", International Joint Conferences on Artificial
                 Intelligence (IJCAI), 2013

    Parameters
    ----------
    n_dims : int
        The dimensionality of the actual search space

    n_embedding_dims : int, default: 2
        The dimensionality of the randomly chosen linear manifold on which the
        optimization is performed

    data_space: array-like, shape=[n_dims, 2], default: None
        The boundaries of the data-space. This is used for scaling the mapping
        from embedded space to data space, which is useful if dimensions of the
        data space have different ranges or are not centred around 0.

    n_keep_dims : int, default: 0
        The number of dimensions which are not embedded in the manifold but are
        kept 1-to-1 in the representation. This can be useful if some
        dimensions are known to be relevant. Note that it is expected that
        those dimensions come first in the data representation, i.e., the first
        n_keep_dims dimensions are maintained.

    Further parameters are the same as in BayesianOptimizer
    """

    def __init__(self, n_dims, n_embedding_dims=2, data_space=None,
                 n_keep_dims=0, *args, **kwargs):
        super(REMBOOptimizer, self).__init__(*args, **kwargs)

        self.n_dims = n_dims
        self.n_embedding_dims = n_embedding_dims
        self.data_space = data_space
        self.n_keep_dims = n_keep_dims
        if self.data_space is not None:
            self.data_space = np.asarray(self.data_space)
            if self.data_space.shape[0] != self.n_dims - n_keep_dims:
                raise Exception("Data space must be specified for all input "
                                "dimensions which are not kept.")

        # Determine random embedding matrix
        self.A = self.rng.normal(size=(self.n_dims - self.n_keep_dims,
                                       self.n_embedding_dims))
        #self.A /= np.linalg.norm(self.A, axis=1)[:, np.newaxis]  # XXX

        self.X_embedded_ = []
        self.boundaries_cache = {}

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
        if not boundaries.shape[0] == self.n_dims:
            raise Exception("Dimensionality of boundaries should be %d"
                            % self.n_dims)

        # Compute boundaries on embedded space
        boundaries_embedded = self._compute_boundaries_embedding(boundaries)

        if len(self.X_) < self.initial_random_samples:
            # Select query point randomly
            X_query_embedded = \
                self.rng.uniform(size=boundaries_embedded.shape[0]) \
                * (boundaries_embedded[:, 1] - boundaries_embedded[:, 0]) \
                    + boundaries_embedded[:, 0]
        else:
            # Select query point by finding optimum of acquisition function
            # within boundaries
            def objective_function(x):
                # Check boundaries
                if not np.all(np.logical_and(x >= boundaries_embedded[:, 0],
                                             x <= boundaries_embedded[:, 1])):
                    return -np.inf

                incumbent = incumbent_fct(self.y_)
                return self.acquisition_function(x, incumbent=incumbent)

            X_query_embedded = global_optimization(
                objective_function, boundaries=boundaries_embedded,
                optimizer=self.optimizer, maxf=self.maxf, random=self.rng)

        self.X_embedded_.append(X_query_embedded)

        # Map to higher dimensional space and clip to hard boundaries
        X_query = np.clip(self._map_to_dataspace(X_query_embedded),
                          boundaries[:, 0], boundaries[:, 1])
        return X_query

    def update(self, X, y):
        """ Update internal model for observed (X, y) from true function. """
        self.X_.append(X)
        self.y_.append(y)
        self.model.fit(self.X_embedded_, self.y_)

    def _map_to_dataspace(self, X_embedded):
        """ Map data from manifold to original data space. """
        X_query_kd = self.A.dot(X_embedded[self.n_keep_dims:])
        if self.data_space is not None:
            X_query_kd = (X_query_kd + 1) / 2 \
                * (self.data_space[:, 1] - self.data_space[:, 0]) \
                + self.data_space[:, 0]
        X_query = np.hstack((X_embedded[:self.n_keep_dims], X_query_kd))

        return X_query

    def _compute_boundaries_embedding(self, boundaries):
        """ Approximate box constraint boundaries on low-dimensional manifold"""
        # Check if boundaries have been determined before
        boundaries_hash = hash(boundaries[self.n_keep_dims:].tostring())
        if boundaries_hash in self.boundaries_cache:
            boundaries_embedded = \
                np.array(self.boundaries_cache[boundaries_hash])
            boundaries_embedded[:self.n_keep_dims] = \
                boundaries[:self.n_keep_dims]  # Overwrite keep-dim's boundaries
            return boundaries_embedded

        # Determine boundaries on embedded space
        boundaries_embedded = \
            np.empty((self.n_keep_dims + self.n_embedding_dims, 2))
        boundaries_embedded[:self.n_keep_dims] = boundaries[:self.n_keep_dims]
        for dim in range(self.n_keep_dims,
                         self.n_keep_dims + self.n_embedding_dims):
            x_embedded = np.zeros(self.n_keep_dims + self.n_embedding_dims)
            while True:
                x = self._map_to_dataspace(x_embedded)
                if np.sum(np.logical_or(
                    x[self.n_keep_dims:] < boundaries[self.n_keep_dims:, 0],
                    x[self.n_keep_dims:] > boundaries[self.n_keep_dims:, 1])) \
                   > (self.n_dims - self.n_keep_dims) / 2:
                    break
                x_embedded[dim] -= 0.01
            boundaries_embedded[dim, 0] = x_embedded[dim]

            x_embedded = np.zeros(self.n_keep_dims + self.n_embedding_dims)
            while True:
                x = self._map_to_dataspace(x_embedded)
                if np.sum(np.logical_or(
                    x[self.n_keep_dims:] < boundaries[self.n_keep_dims:, 0],
                    x[self.n_keep_dims:] > boundaries[self.n_keep_dims:, 1])) \
                   > (self.n_dims - self.n_keep_dims) / 2:
                    break
                x_embedded[dim] += 0.01
            boundaries_embedded[dim, 1] = x_embedded[dim]

        self.boundaries_cache[boundaries_hash] = boundaries_embedded

        return boundaries_embedded


class InterleavedREMBOOptimizer(BayesianOptimizer):
    """ Interleaved Random EMbedding Bayesian Optimization (REMBO).

    In this extension of REMBO, several different random embeddings are chosen
    and the optimization is performed on all embeddings interleaved (in a
    round-robin fashion). This way, the specific choice of one random embedding
    becomes less relevant. On the other hand, less evaluations on each
    particular embedding can be performed.

    .. seealso:: Wang, Zoghi, Hutter, Matheson, de Freitas
                 "Bayesian Optimization in High Dimensions via Random
                 Embeddings", International Joint Conferences on Artificial
                 Intelligence (IJCAI), 2013

    Parameters
    ----------
    interleaved_runs : int
        The number of interleaved runs (each on a different random embedding).
        This parameter is denoted as k by Wang et al.

    Further parameters are the same as in REMBOOptimizer
    """

    def __init__(self, interleaved_runs=2, *args, **kwargs):
        random_state = kwargs.pop("random_state", 0)

        self.rembos = [REMBOOptimizer(random_state=random_state + 100 + run,
                                      *args, **kwargs)
                       for run in range(interleaved_runs)]
        self.rembos = cycle(self.rembos)
        self.current_rembo = self.rembos.next()

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
        return self.current_rembo.select_query_point(boundaries, incumbent_fct)

    def update(self, X, y):
        """ Update internal REMBO responsible for observed (X, y). """
        self.X_.append(X)
        self.y_.append(y)

        self.current_rembo.update(X, y)
        self.current_rembo = self.rembos.next()
