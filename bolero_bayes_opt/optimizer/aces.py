# Author: Jan Hendrik Metzen <janmetzen@mailbox.org>
# Date: 01/07/2015

from copy import deepcopy

import numpy as np

from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from bayesian_optimization import create_acquisition_function
from bayesian_optimization.utils.optimization import global_optimization

from .bocps import BOCPSOptimizer


class ACESOptimizer(BOCPSOptimizer):
    """(Active) contextual entropy search

    Behaves like ContextualBayesianOptimizer but is additionally also able to
    actively select the context for the next trial.

    Parameters
    ----------
    context_boundaries : list of pair of floats
        The boundaries of the context space in which the best context for the
        next trial is searched

    n_query_points : int, default: 100
        The number of candidate query points (context-parameter pairs) which
        are determined by the base acquisition function and evaluated using
        ACES

    active : bool
        Whether the context for the next trial is actively selected. This might
        improve the performance in tasks where a uniform selection of contexts
        is suboptimal. However, it also increases the dimensionality of the
        search space.

    For further parameters, we refer to the doc of ContextualBayesianOptimizer.
    """
    def __init__(self, context_boundaries, active=True, **kwargs):
        super(ACESOptimizer, self).__init__(**kwargs)
        self.context_boundaries = context_boundaries
        self.active = active


    def init(self, n_params, n_context_dims):
        super(ACESOptimizer, self).init(n_params, n_context_dims)
        if len(self.context_boundaries) == 1:
            self.context_boundaries = \
                np.array(self.context_boundaries * self.context_dims)
        elif len(self.context_boundaries) == self.context_dims:
            self.context_boundaries = np.array(self.context_boundaries)
        else:
            raise Exception("Context-boundaries not specified for all "
                            "context dimensions.")

        self.cx_boundaries = np.empty((self.context_dims + self.dimension, 2))
        self.cx_boundaries[:self.context_dims] = self.context_boundaries
        self.cx_boundaries[self.context_dims:] = self.boundaries

    def get_desired_context(self):
        """Chooses desired context for next evaluation.

        Returns
        -------
        context : ndarray-like, default=None
            The context in which the next rollout shall be performed. If None,
            the environment may select the next context without any
            preferences.
        """
        if self.active:
            # Active task selection: determine next context via Bayesian
            # Optimization
            self.context, self.parameters = \
                self._determine_contextparams(self.bayes_opt)
        else:
            raise NotImplementedError("Passive ACES not implemented")
            ## Choose context randomly and only choose next parameters
            #self.context = self.rng.uniform(size=self.context_dims) \
            #    * (self.context_boundaries[:, 1]
            #        - self.context_boundaries[:, 0]) \
            #    + self.context_boundaries[:, 0]
            ## Repeat context self.n_query_points times, s.t. ACES can only
            ## select parameters for this context
            #contexts = np.repeat(self.context, self.n_query_points)
            #contexts = contexts.reshape(-1, self.n_query_points).T
            #_, self.parameters = \
            #    self._determine_contextparams(self.bayes_opt, contexts)
        # Return only context, the parameters are later returned in
        # get_next_parameters
        return self.context

    def set_context(self, context):
        """ Set context of next evaluation"""
        assert np.all(context == self.context)  # just a sanity check

    def get_next_parameters(self, params, explore=True):
        """Return parameter vector that shall be evaluated next.

        Parameters
        ----------
        params : array-like
            The selected parameters will be written into this as a side-effect.

        explore : bool
            Whether exploration in parameter selection is enabled
        """
        # We have selected the parameter already along with
        # the context in get_desired_context()
        params[:] = self.parameters

    def _determine_contextparams(self, optimizer):
        """Select context and params jointly using ACES."""
        # Determine optimal parameters for fixed context
        cx = optimizer.select_query_point(self.cx_boundaries)
        return cx[:self.context_dims], cx[self.context_dims:]

    def _create_acquisition_function(self, name, model, **kwargs):
        if not name in ["ContextualEntropySearch",
                        "ContextualEntropySearchLocal"]:
            raise ValueError("%s acquisition function not supported."
                             % name)
        return create_acquisition_function(name, model, **kwargs)


class SurrogateACESOptimizer(ACESOptimizer):
    def __init__(self, context_boundaries, n_context_samples, kappa,
                 active=True, **kwargs):
        super(SurrogateACESOptimizer, self).__init__(
            context_boundaries=context_boundaries, active=active, **kwargs)
        self.n_context_samples = n_context_samples
        self.kappa = kappa

    def init(self, n_params, n_context_dims):
        super(SurrogateACESOptimizer, self).init(n_params, n_context_dims)

    def _determine_contextparams(self, optimizer):
        """Select context and params jointly using ACES."""
        # Choose the first samples uniform randomly
        if len(optimizer.X_) < optimizer.initial_random_samples:
            cx = np.random.uniform(self.cx_boundaries[:, 0],
                                   self.cx_boundaries[:, 1])
            return cx[:self.context_dims], cx[self.context_dims:]

        # Prepare entropy search objective
        self._init_es_ensemble()
        # Generate data for function mapping
        # query_context x query_parameters x eval_context -> entropy reduction
        n_query_points = 500
        n_data_dims = 2 * self.context_dims + self.dimension
        X = np.empty((n_query_points, n_data_dims))
        y = np.empty(n_query_points)
        for i in range(n_query_points):
            # Select query point and evaluation context randomly
            query = np.random.uniform(self.cx_boundaries[:, 0],
                                      self.cx_boundaries[:, 1])
            ind = np.random.choice(self.n_context_samples)
            # Store query point in X and value of entropy-search in y
            X[i, :self.context_dims + self.dimension] = query
            X[i, self.context_dims + self.dimension:] = \
                self.context_samples[ind] - query[:self.context_dims]
            y[i] = self.entropy_search_ensemble[ind](query)[0]

        # Fit GP model to this data
        kernel = C(1.0, (1e-10, 100.0)) \
            * RBF(length_scale=(1.0,)*n_data_dims,
                  length_scale_bounds=[(0.01, 10.0),]*n_data_dims) \
            + WhiteKernel(1.0, (1e-10, 100.0))
        self.es_surrogate = GaussianProcessRegressor(kernel=kernel)
        self.es_surrogate.fit(X, y)

        # Select query based on mean entropy reduction in surrogate model
        # predictions
        contexts = np.random.uniform(self.context_boundaries[:, 0],
                                     self.context_boundaries[:, 1],
                                     (250, self.context_dims))
        def objective_function(cx):
            X_query = np.empty((250, n_data_dims))
            X_query[:, :self.context_dims + self.dimension] = cx
            X_query[:, self.context_dims + self.dimension:] = \
                contexts - cx[:self.context_dims]
            es_pred, es_cov = \
                self.es_surrogate.predict(X_query, return_cov=True)
            return es_pred.mean() + self.kappa * np.sqrt(es_cov.mean())

        cx = global_optimization(
                objective_function, boundaries=self.cx_boundaries,
                optimizer=self.optimizer, maxf=optimizer.maxf)
        return cx[:self.context_dims], cx[self.context_dims:]


    def _init_es_ensemble(self):
        # Determine samples at which CES will be evaluated by
        # 1. uniform random sampling
        self.context_samples = \
            np.random.uniform(self.context_boundaries[:, 0],
                              self.context_boundaries[:, 1],
                              (self.n_context_samples*25, self.context_dims))
        # 2. subsampling via k-means clustering
        kmeans = KMeans(n_clusters=self.n_context_samples, n_jobs=1)
        self.context_samples = \
            kmeans.fit(self.context_samples).cluster_centers_

        # 3. Create entropy search ensemble
        self.entropy_search_ensemble = []
        for i in range(self.n_context_samples):
            cx_boundaries_i = np.copy(self.cx_boundaries)
            cx_boundaries_i[:self.context_dims] = \
                self.context_samples[i][:, np.newaxis]
            entropy_search_fixed_context = deepcopy(self.acquisition_function)
            entropy_search_fixed_context.set_boundaries(cx_boundaries_i)

            self.entropy_search_ensemble.append(entropy_search_fixed_context)

    def _create_acquisition_function(self, name, model, **kwargs):
        if not name in ["EntropySearch", "MinimalRegretSearch"]:
            raise ValueError("%s acquisition function not supported."
                             % name)
        return create_acquisition_function(name, model, **kwargs)