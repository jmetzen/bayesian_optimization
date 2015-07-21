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

    def fit(self, X, y):
        """ Fits a new Gaussian process model on (X, y) data. """
        # Only train GP model if the training set has actually changed
        if inputs.shape[0] <= self.last_training_size:
            return

        # Create and fit Gaussian Process model
        self.gp = self._create_gp(training_size=inputs.shape[0])
        self.gp.fit(X, y)

        self.kernel_ = self.gp.kernel_
        self.last_training_size = inputs.shape[0]

    def predictive_distribution(self, X):
        return self.gp.predict(X, return_std=True)

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
