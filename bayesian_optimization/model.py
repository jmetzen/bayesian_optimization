# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>

import numpy as np
import copy
import logging
from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
from bolero.utils.validation import check_random_state


class GaussianProcessModel(object):
    """Learn a model of the return landscape using a Gaussian process.

    The main purpose of this model is the usage within a Bayesian optimizer.

    Parameters
    ----------
    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.

    sigma_squared_n : float or array-like, optional (default: 1e-10)
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to increased noise level in the observations
        and reduce potential numerical issue during fitting. If an array is
        passed, it must have the same number of entries as the data used for
        fitting and is used as datapoint-dependent noise level.

    optimizer : string or callable, optional (default: "fmin_l_bfgs_b")
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the  signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the 'fmin_l_bfgs_b' algorithm from scipy.optimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

    n_restarts_optimizer: int, optional (default: 1)
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 1, all bounds
        must be finite.

    normalize_y: boolean, optional (default: False)
        Whether the target values y are normalized, i.e., the mean of the
        observed target values become zero. This parameter should be set to
        True if the target values' mean is expected to differ considerable from
        zero. When enabled, the normalization effectively modifies the GP's
        prior based on the data, which contradicts the likelihood principle;
        normalization is thus disabled per default.

    reestimate_hyperparams:
        Function deciding (based on size of training data) whether
        hyperparameters are reestimated. Note that this exploits the
        incremental nature of Bayesian optimization.

    bayesian_gp:
        Whether hyperparameters of the Gaussian process are inferred using
        Bayesian inference.

    random_state : optional, int
        Seed for the random number generator.
    """
    def __init__(self, kernel=None, sigma_squared_n=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=1,
                 normalize_y=False,
                 reestimate_hyperparams=None, bayesian_gp=False,
                 random_state=None):
        self.kernel_ = kernel
        self.sigma_squared_n = sigma_squared_n
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y

        self.gp_reestimate_hyperparams = reestimate_hyperparams
        if self.gp_reestimate_hyperparams is None:
            self.gp_reestimate_hyperparams = lambda _: True
        self.bayesian_gp = bayesian_gp

        self.random = check_random_state(random_state)

        self.gp = None
        self.last_training_size = 0

    def train(self, inputs, outputs):
        """ Fits a new Gaussian process model on (inputs, outputs) data. """
        # Only train GP model if the training set has actually changed
        if inputs.shape[0] <= self.last_training_size:
            return

        # Create and fit Gaussian Process model
        self.gp = self._create_gp(training_size=inputs.shape[0])
        self.gp.fit(inputs, outputs)

        self.kernel_ = self.gp.kernel_
        self.last_training_size = inputs.shape[0]

    def _create_gp(self, training_size):
        # Decide whether to perform hyperparameter-optimization of GP
        if self.gp_reestimate_hyperparams(training_size):
            gp = GaussianProcessRegressor(
                kernel=self.kernel_, sigma_squared_n=self.sigma_squared_n,
                optimizer=self.optimizer,
                n_restarts_optimizer=self.n_restarts_optimizer,
                normalize_y=self.normalize_y, random_state=self.random)
        else:
            gp = GaussianProcessRegressor(
                kernel=self.kernel_, sigma_squared_n=self.sigma_squared_n,
                optimizer=None,  # do not modify kernel's hyperparameters
                n_restarts_optimizer=self.n_restarts_optimizer,
                normalize_y=self.normalize_y, random_state=self.random)

        # Select if standard or Bayesian Gaussian process are used
        if self.bayesian_gp:
            raise NotImplementedError("Bayesian GP is not yet implemented.")

        return gp

    def __getstate__(self):
        """ Return a pickable state for this object """
        odict = self.__dict__.copy()  # copy the dict since we change it
        if "gp_reestimate_hyperparams" in odict:
            odict.pop("gp_reestimate_hyperparams")
        return odict


def model_free_policy_training(policy, contexts, parameters, returns=None,
                               epsilon=1.0, min_eta=1e-6):
    """Train a contextual policy in a model-free way using REPS.

    Use the given (context, parameter, return) triples for training a
    contextual policy using REPS. The contexts are passed through a
    context_transformation function.

    Parameters
    ----------
    policy: UpperLevelPolicy
        A contextual upper level policy that shall be updated based on samples.

    contexts: array-like, [n_samples, context_dims]
        The contexts used for training

    parameters: array-like, [n_samples, param_dims]
        The parameters of the low-level policy used in the respective context.

    returns: array-like, [n_samples]
        The returns obtained by the low-level policy in the respective context.
        If None, then all training examples are used with equal weight;
        otherwise the returns are used to determine the weights of the samples
        by using REPS.

    epsilon: float > 0.0
        The maximum the KL divergence between training and policy distribution
        might take on. Defaults to 1.0

    min_eta: float > 0.0
        eta is the Lagrangian multiplier connected to the KL-bound constraint.
        If it becomes too small, numerical instability might occur. Because of
        this, min_eta can be set to a value larger than zero which acts as a
        lower bound for eta. Defaults to 1e-6.

    Returns
    -------
    policy : LinearGaussianPolicy
        Upper level policy that maps contexts to parameter vectors
    """
    parameters = np.asarray(parameters)
    features = np.array([policy.transform_context(c) for c in contexts])
    if returns is not None:
        returns = np.asarray(returns)
        from bolero.optimizer.creps import solve_dual_contextual_reps
        d, _ = solve_dual_contextual_reps(features, returns, epsilon=epsilon,
                                          min_eta=min_eta)
    else:
        d = np.ones(contexts.shape[0]) / contexts.shape[0]

    policy.fit(contexts, parameters, d)

    return policy


def model_based_policy_training(policy, contexts, parameters, returns,
                                boundaries=None, policy_initialized=False,
                                model_conf={}, maxfun=5000,
                                variance=0.01, *args, **kwargs):
    """Train a contextual policy in a model-based way on GP model.

    Use the given (context, parameter, return) triples for training a
    model (based on Gaussian processes) of the function
    context x parameter > return. Optimize the parameters of a parametric
    policy such that it would maximize the accrued return in the model.

    Parameters
    ----------
    policy: UpperLevelPolicy
        A contextual upper level policy that shall be updated based on samples.

    contexts: array-like, [n_samples, context_dims]
        The contexts used for training

    parameters: array-like, [n_samples, param_dims]
        The parameters of the low-level policy used in the respective context.

    returns: array-like, [n_samples]
        The returns obtained by the low-level policy in the respective context.
        If None, then all training examples are used with equal weight;
        otherwise the returns are used to determine the weights of the samples
        by using REPS.

    boundaries : array-like or None, shape=[param_dims, 2] (default: None)
        The boundaries of the action space.

    policy_initialized : bool (default: False)
        Whether the policy has already been initialized reasonably or needs to
        be initialized first before performing model-based optimization

    model_conf: dict (default : {})
        Passed as keyword arguments to the GP model.

    maxfun: int, optional (default: 5000)
        The maximal number of policy parameters after which the optimizer
        terminates

    variance: float, optional (default: 0.01)
        The initial exploration variance of CMA-ES

    Returns
    -------
    policy : LinearGaussianPolicy
        Upper level policy that maps contexts to parameter vectors
    """
    contexts = np.asarray(contexts)
    parameters = np.asarray(parameters)

    # Initialize policy
    if boundaries is not None:
        # Compute scale factor per dimension such that isotropic exploration
        # is meaningful
        scale_factor = (boundaries[:, 1] - boundaries[:, 0])[:, None]
        if not policy_initialized:
            # Initialize policy such that outputs are maximally far from
            # boundary
            policy.fit(contexts, [boundaries.mean(1)]*contexts.shape[0],
                       weights=np.ones(contexts.shape[0]))
    else:
        scale_factor = 1  # Don't do scaling since the scales are unknown
        if not policy_initialized:
            # Let policy return mean parameter vector
            policy.fit(contexts, parameters.mean(),
                       weights=np.ones(contexts.shape[0]))

    # Train model approximating the context x parameter -> return landscape
    inputs = np.hstack([contexts, parameters])
    model = GaussianProcessModel(**model_conf)
    model.train(inputs, returns)

    def average_return(policy_params):
        """ Return predicted return of policy for given policy parameters. """
        # Instantiate policy
        policy.W = np.array(policy_params.reshape(policy.W.shape))
        policy.W *= scale_factor
        # Determine parameters selected by policy for given contexts
        params = policy(contexts, explore=False)
        if boundaries is not None:  # check boundaries
            params = np.minimum(np.maximum(params, boundaries[:, 0]),
                                boundaries[:, 1])
        # Compute mean output of GP model for contexts and selected params
        values = model.gp.predict(np.hstack((contexts, params)))
        # Return mean over return obtained on all training contexts
        return values.mean()

    # Refine policy determined in model-free way by performing L-BFGS on
    # the model.
    from bolero.optimizer import fmin as fmin_cmaes
    policy.W /= scale_factor
    x_lbfgs, _ = fmin_cmaes(
        average_return, x0=policy.W.flatten(), maxfun=maxfun,
        eval_initial_x=True, variance=variance, maximize=True, *args, **kwargs)

    # Set weights of linear policy
    policy.W = x_lbfgs.reshape(policy.W.shape)
    policy.W *= scale_factor

    return policy
