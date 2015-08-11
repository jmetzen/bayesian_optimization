# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# Date: 01/07/2015

import warnings
from copy import deepcopy

import numpy as np
from scipy.stats import entropy, norm

from bayesian_optimization.model import ParametricModelApproximation

from .bocps import BOCPSOptimizer
from ..representation.ul_policies \
    import model_free_policy_training, model_based_policy_training_pretrained


class ACEPSOptimizer(BOCPSOptimizer):
    """(Active) contextual entropy policy search

    Behaves like ContextualBayesianOptimizer but is additionally also able to
    actively select the context for the next trial.

    Note: The parameter "acquisition_function" is ignored in ACEPSOptimizer
          since ACEPSOptimizer can only be used with the ACEPS acquisition
          function.

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

    active : bool
        Whether the context for the next trial is actively selected. This might
        improve the performance in tasks where a uniform selection of contexts
        is suboptimal. However, it also increases the dimensionality of the
        search space.

    For further parameters, we refer to the doc of ContextualBayesianOptimizer.
    """
    def __init__(self, policy, context_boundaries, active=True, **kwargs):
        kwargs["acquisition_function"] = "aceps"
        super(ACEPSOptimizer, self).__init__(policy=policy, **kwargs)
        self.context_boundaries = context_boundaries
        self.active = active

        if self.policy is None:
            raise ValueError("The policy in ACEPS must not be None.")

    def init(self, n_params, n_context_dims):
        super(ACEPSOptimizer, self).init(n_params, n_context_dims)
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
            self.context = self.random.uniform(size=self.context_dims) \
                * (self.context_boundaries[:, 1]
                    - self.context_boundaries[:, 0]) \
                + self.context_boundaries[:, 0]
            self.parameters = self._determine_params(self.context, self.bayes_opt)
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

    def _determine_params(self, context, optimizer):
        # Prepend fixed context to search space
        cx_boundaries = np.empty((self.context_dims + self.dimension, 2))
        cx_boundaries[:self.context_dims] = context[:, np.newaxis]
        cx_boundaries[self.context_dims:] = self.boundaries

        # Determine optimal parameters for fixed context
        cx = optimizer.select_query_point(cx_boundaries)
        return cx[self.context_dims:]

    def _determine_contextparams(self, optimizer):
        # Optimize context and params jointly
        cx_boundaries = np.vstack((self.context_boundaries, self.boundaries))
        cx = optimizer.select_query_point(cx_boundaries)

        context = cx[:self.context_dims]
        params = cx[self.context_dims:]

        return context, params

    def _create_acquisition_function(self, name, model, **kwargs):
        # Determine boundaries
        cx_boundaries = np.vstack((self.context_boundaries, self.boundaries))

        acq_function = \
            ACEPS(self.model, self.policy, cx_boundaries,  # XXX: self.model.gp
                  n_context_dims=self.context_dims, **kwargs)
        return acq_function


class ACEPS(object):
    """ (Active) contextual entropy policy search (ACEPS) acquisition function.

    The ACEPS acquisition function prefers query points which will reduce the
    uncertainty (entropy) of the optimal policy according to the current model
    (in expectation, estimated by sampling from the model's posterior).
    It is thus suited only for contextual policy search problems and requires
    that a parametric class of policies is defined.

    Parameters
    ----------
    model : surrogate model object
        The surrogate model which is used to model the objective function. It
        needs to provide a methods fit(X, y) for training the model and
        predictive_distribution(X) for determining the predictive distribution
        (mean, std-dev) at query point X.
    policy : UpperLevelPolicy-object
        Representation of the upper-level behavior policy (mapping from context
        to low-level parameters)
    bounds : ndarray-like, shape: (n_context_dims + n_param_dims, 2)
        Box constraints on context and parameter space. Lower boundaries are
        stored as bounds[:, 0], upper boundaries as bounds[:, 1]. The first
        n_context_dims rows define the box constraint on the context space, the
        other n_param_dims rows the box constraint on the parameter space
    n_context_dims: int
        The number of context dimensions.
    n_context_samples: int, default: 20
        The number of context sampled from context space on which sample
        policies are evaluated
    n_policy_samples: int, default: 20
        The number of policies sampled internally, on which the distribution
        of optimal policies is evaluated
    n_gp_samples: int, default: 1000
        The number of samples drawn from the GP posterior on which the policy
        samples are evaluated
    n_samples_y: int, default: 10
        The number of simulated output values drawn from GP posterior which
        are used for each query point
    use_finite_parameterization: bool, default: False
        Whether the model is approximated by a finite parametrization,
        which might improve speed but be less accurate.
    seed: int, default: None
        Seed of random number generator. If None, the random numbers are
        not reproducible.
    """
    def __init__(self, model, policy, bounds, n_context_dims,
                 n_context_samples=20, n_policy_samples=20, n_gp_samples=1000,
                 n_samples_y=10, use_finite_parameterization=False, seed=None):
        self.model = model
        self.policy = deepcopy(policy)  # XXX
        self.bounds = bounds
        self.n_context_dims = n_context_dims
        self.n_param_dims = self.bounds.shape[0] - n_context_dims
        self.context_bounds = self.bounds[:self.n_context_dims]
        self.param_bounds = self.bounds[self.n_context_dims:]

        self.n_context_samples = n_context_samples
        self.n_policy_samples = n_policy_samples
        self.n_gp_samples = n_gp_samples
        self.n_samples_y = n_samples_y
        self.use_finite_parameterization = use_finite_parameterization

        self.rng = np.random.RandomState(seed)

        # Sample contexts at which policies will be evaluated and be compared
        self.context_samples = \
            self.rng.uniform(self.context_bounds[:, 0],
                             self.context_bounds[:, 1],
                             (self.n_context_samples, n_context_dims))

        self.n_seen_datapoints = 0

    def _update(self):
        if self.model.last_training_size > self.n_seen_datapoints:
            # Select parameters that policy samples would choose at context samples
            self.selected_params = self._sample_policy_parameters()

            # Create array of evaluation points (context samples, policy parameters)
            self.eval_points = \
                np.hstack((np.tile(self.context_samples, (self.n_policy_samples, 1)),
                           self.selected_params.reshape(-1, self.n_param_dims)))

            if self.use_finite_parameterization:
                self.pma = \
                    ParametricModelApproximation(self.model.gp, self.bounds, 100, 0)  # XXX
            else:
                # Compute mean and covariance of GP model over the evaluation points
                self.y_mean, self.y_cov = \
                    self.model.gp.predict(self.eval_points, return_cov=True)

            self.n_seen_datapoints = self.model.last_training_size

    def _sample_policy_parameters(self):
        """ Sample close-to-optimal policies and let them select parameters.

        We determine a set of policies which is close-to-optimal according to
        samples drawn from the model's posterior and let these policies
        determine parameters
        """
        # Compute policy which is close to optimal according to current model
        contexts = self.model.gp.X_train_[:, :self.n_context_dims]
        self.policy.fit(contexts,
                        [self.param_bounds.mean(1)]*contexts.shape[0],
                        weights=np.ones(contexts.shape[0]))
        self.policy = model_based_policy_training_pretrained(
            policy=self.policy, model=self.model.gp,
            contexts=contexts, boundaries=self.param_bounds)

        # Draw context samples, let policy select parameters for these context
        # (with exploration), and sample multiple possible outcomes for these
        # (context, parameter) samples from the GP model.
        while True:  # XXX
            n_samples = 250  # XXX
            X_sample = np.empty((n_samples, self.bounds.shape[0]))
            X_sample[:, :self.n_context_dims] = \
                self.rng.uniform(self.context_bounds[:, 0],
                                 self.context_bounds[:, 1],
                                 (n_samples, self.n_context_dims))
            X_sample[:, self.n_context_dims:] = \
                [self.policy(X_sample[i, :self.n_context_dims])
                 for i in range(n_samples)]
            try:
                y_sample = self.model.gp.sample_y(X_sample, self.n_policy_samples)
                break
            except np.linalg.LinAlgError:
                continue

        # Train for each possible outcome one policy and evaluate this policy
        # on the context samples
        selected_params = []  # XXX: Vectorize
        for i in range(y_sample.shape[1]):
            policy_sample = model_free_policy_training(
                self.policy, X_sample[:, range(self.n_context_dims)],
                X_sample[:, range(self.n_context_dims, X_sample.shape[1])],
                y_sample[:, i])

            params = [policy_sample(np.atleast_1d(self.context_samples[i]),
                                    explore=False).ravel()
                      for i in range(self.context_samples.shape[0])]
            selected_params.append(params)

        selected_params = np.array(selected_params)
        # Enforce lower and upper bound on possible parameters
        for i in range(selected_params.shape[2]):
            selected_params[selected_params[:, :, i] < self.param_bounds[i, 0]] = \
                self.param_bounds[i, 0]
            selected_params[selected_params[:, :, i] > self.param_bounds[i, 1]] = \
                self.param_bounds[i, 1]
        return selected_params

    def __call__(self, X_query, *args, **kwargs):
        self._update()  # XXX

        X_query = np.atleast_2d(X_query)

        if self.use_finite_parameterization:
            y_samples = self._fp_samples(X_query)
        else:
            try:
                y_samples = self._gp_samples(X_query)
            except np.linalg.LinAlgError:
                warnings.warn("LinAlgError: SVD did not converge")
                return -1e10  # XXX

        # determine for each GP sample which policy was best
        best_policies = np.argmax(y_samples.mean(1), 0)

        # Compute entropies of sampled best policy distribution
        entropies = np.array([
            entropy(np.bincount(best_policies[:, i],
                                minlength=self.n_policy_samples))
            for i in range(best_policies.shape[1])])

        # Return negative mean entropy.
        # Since we maximize acquisition functions and want to minimize entropy,
        # we have to multiply the mean entropy with -1
        return -np.mean(entropies)

    def _gp_samples(self, X_query):
        # Predict covariance between X_query and itself
        y_cov_query_self = self.model.gp.predict(X_query, return_cov=True)[1]
        # Compute covariance between X_query and the eval_points
        y_cov_query_cross = self.model.gp.predict(np.vstack((X_query, self.eval_points)),
                                       return_cov=True)[1][1:, [0]]  # XXX

        # Compute change of covariance at evaluation points when performing
        # an additional evaluation at X_query
        y_cov_delta = \
            -(y_cov_query_cross / y_cov_query_self).dot(y_cov_query_cross.T)

        # Draw random-samples from N(y_mean, y_cov + y_cov_delta)
        y_samples = self.rng.multivariate_normal(
            self.y_mean, self.y_cov + y_cov_delta, self.n_gp_samples).T

        # Adapt samples for different possible values of y at X_query
        # according to predictive distribution of GP at X_query
        percent_points = norm.ppf(np.linspace(0.05, 0.95, self.n_samples_y))
        y_delta = np.sqrt(y_cov_query_self + self.model.gp.sigma_squared_n)[:, 0] \
            * percent_points
        y_mean_delta = (y_cov_query_cross / y_cov_query_self) * y_delta

        y_samples = y_samples[:, :, None] + y_mean_delta[:, None]
        return y_samples.reshape(self.n_policy_samples, self.n_context_samples,
                                 self.n_gp_samples, y_delta.shape[0])

    def _fp_samples(self, X_query):
        # Determine possible results (according to current model) of performing
        # an evaluation at X_query
        mean, std = self.model.gp.predict(X_query, return_std=True)
        y_query_samples = \
            norm(mean, std).ppf(np.linspace(0.05, 0.95, self.n_samples_y))
        # Determine coefficients for parametric model for the given simulated
        # query
        coefs = self.pma.determine_coefs(X_query, y_query_samples,
                                         n_samples=self.n_gp_samples)
        # Determine outputs at evaluation points as predicted by the parametric
        # model approximation
        y_samples = self.pma(self.eval_points, coefs)

        return y_samples.reshape(self.n_policy_samples, self.n_context_samples,
                                 self.n_gp_samples, self.n_samples_y)
