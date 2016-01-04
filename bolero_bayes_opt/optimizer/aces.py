# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# Date: 01/07/2015

import warnings
from copy import deepcopy

import numpy as np
from scipy.stats import entropy, norm

from bayesian_optimization import create_acquisition_function
from bayesian_optimization.acquisition_functions import AcquisitionFunction
from bayesian_optimization.model import ParametricModelApproximation

from .bocps import BOCPSOptimizer
from ..representation.ul_policies \
    import model_free_policy_training, model_based_policy_training_pretrained


class ACESOptimizer(BOCPSOptimizer):
    """(Active) contextual entropy search

    Behaves like ContextualBayesianOptimizer but is additionally also able to
    actively select the context for the next trial.

    Parameters
    ----------
    policy : UpperLevelPolicy-object
        The given upper-level-policy object, which is optimized in
        best_policy() such that the average reward in the internal GP-model is
        maximized. The policy representation is also used in the ACEPS
        acquisition function.

    context_boundaries : list of pair of floats
        The boundaries of the context space in which the best context for the
        next trial is searched

    n_query_points : int, default: 100
        The number of candidate query points (context-parameter pairs) which
        are determined by the base acquisition function and evaluated using
        ACEPS

    active : bool
        Whether the context for the next trial is actively selected. This might
        improve the performance in tasks where a uniform selection of contexts
        is suboptimal. However, it also increases the dimensionality of the
        search space.

    For further parameters, we refer to the doc of ContextualBayesianOptimizer.
    """
    def __init__(self, policy, context_boundaries, n_query_points=100,
                 active=True, aceps_params=None, **kwargs):
        super(ACESOptimizer, self).__init__(policy=policy, **kwargs)
        self.context_boundaries = context_boundaries
        self.n_query_points = n_query_points
        self.active = active
        self.aceps_params = aceps_params if aceps_params is not None else {}

        if self.policy is None:
            raise ValueError("The policy in ACEPS must not be None.")

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
            # Choose context randomly and only choose next parameters
            self.context = self.rng.uniform(size=self.context_dims) \
                * (self.context_boundaries[:, 1]
                    - self.context_boundaries[:, 0]) \
                + self.context_boundaries[:, 0]
            # Repeat context self.n_query_points times, s.t. ACEPS can only
            # select parameters for this context
            contexts = np.repeat(self.context, self.n_query_points)
            contexts = contexts.reshape(-1, self.n_query_points).T
            _, self.parameters = \
                self._determine_contextparams(self.bayes_opt, contexts)
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
        """Select context and params jointly using ACEPS."""
        cx_boundaries = np.empty((self.context_dims + self.dimension, 2))
        cx_boundaries[:self.context_dims] = self.context_boundaries
        cx_boundaries[self.context_dims:] = self.boundaries

        # Determine optimal parameters for fixed context
        cx = optimizer.select_query_point(cx_boundaries)
        return cx[:self.context_dims], cx[self.context_dims:]

    def _create_acquisition_function(self, name, model, **kwargs):
        if not name in ["ContextualEntropySearch",
                        "ContextualEntropySearchLocal", "ACEPS"]:
            raise ValueError("%s acquisition function not supported."
                             % name)

        return create_acquisition_function(name, model, **kwargs)

