""" Acquisition functions which can be used in Bayesian optimization."""
# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>

import numpy as np
from copy import deepcopy
from functools import partial
from scipy.stats import norm
from scipy.special import erf
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcess


class ProbabilityOfImprovement(object):
    """The "Probability of Improvement" (PI) acquisition function.

    Parameters
    ----------
    model: gaussian-process object
        Gaussian process model, which models the return landscape
    kappa: optional, float >= 0.0
        Parameter controlling the exploration-exploitation trade-off. Larger
        values correspond to increased exploration. Defaults to 0
        which corresponds to pure exploitation.
    """
    def __init__(self, model, kappa=0.0):
        self.model = model
        self.kappa = 0  # TODO: kappa

    def __call__(self, x, baseline_value, compute_gradient=False, *args,
                 **kwargs):
        """Returns the probability of improvement at query point x.

        Parameters
        ----------
        x: array-like
            The position at which the probability of improvement will be
            evaluated.
        baseline_value: float
            Baseline value, typically the maximum (actual) return observed
            so far during learning.
        compute_gradient: bool
            Whether the gradient at position x will be evaluated and returned.

        Returns
        -------
        pi: float
            The probability that query point x yields a better return than
            the best seen so far (i.e., baseline_value)
        gradient: array-like, optional (if compute_gradient is True)
            The gradient of the probability of improvement at position x.
        """
        # Let GP predict mean and variance
        if compute_gradient:
            # Let GP model predict mean and variance and their gradients
            mu_x, sigma_x, dx_mu_x, dx_sigma_x = \
                self.model.predict(x, return_std=True, eval_gradient=True)
        else:
            # Let GP model predict mean and variance
            mu_x, sigma_x = self.model.predict(x, return_std=True)

        gamma_x = (mu_x - (baseline_value - self.kappa)) / sigma_x
        pi = norm.cdf(gamma_x)

        if not compute_gradient:
            return pi
        else:
            # Compute gradient of PI with respect to x
            dx_gamma_x = dx_mu_x / sigma_x - gamma_x / sigma_x * dx_sigma_x
            dx_pi = dx_gamma_x / np.sqrt(2*np.pi) * np.exp(-gamma_x**2/2)
            return pi, dx_pi


class ExpectedImprovement(object):
    """ The "Expected Improvement" (EI) acquisition function.

    Parameters
    ----------
    model: gaussian-process object
        Gaussian process model, which models the return landscape
    kappa: optional, float >= 0.0
        Parameter controlling the exploration-exploitation trade-off. Larger
        values correspond to increased exploration. Defaults to 0
        which corresponds to pure exploitation.
    """
    def __init__(self, model, kappa=0.0):
        self.model = model
        self.kappa = 0  # TODO: kappa

    def __call__(self, x, baseline_value, compute_gradient=False, *args,
                 **kwargs):
        """Returns the expected improvement at query point x.

        Parameters
        ----------
        x: array-like
            The position at which the expected improvement will be evaluated.
        baseline_value: float
            Baseline value, typically the maximum (actual) return observed
            so far during learning.
        compute_gradient: bool
            Whether the gradient at position x will be evaluated and returned.

        Returns
        -------
        ei: float
            The expected improvement at query point x over the current best
            (i.e, baseline_value)
        gradient: array-like, optional (if compute_gradient is True)
            The gradient of the expected improvement at position x.
        """
        # Let GP predict mean and variance
        if compute_gradient:
            # Let GP model predict mean and variance and their gradients
            mu_x, sigma_x, dx_mu_x, dx_sigma_x = \
                self.model.predict(x, return_std=True, eval_gradient=True)
        else:
            # Let GP model predict mean and variance
            mu_x, sigma_x = self.model.predict(x, return_std=True)

        gamma_x = (mu_x - (baseline_value - self.kappa)) / sigma_x
        # Compute EI based on some temporary variables that can be reused in
        # gradient computation
        tmp_erf = erf(gamma_x / np.sqrt(2))
        tmp_ei_no_std = 0.5*gamma_x * (1 + tmp_erf) \
            + np.exp(-gamma_x**2/2)/np.sqrt(2*np.pi)
        ei = sigma_x * tmp_ei_no_std

        if not compute_gradient:
            return ei
        else:
            # Compute gradient of EI with respect to x
            dx_gamma_x = (dx_mu_x - gamma_x * dx_sigma_x) / sigma_x

            dx_ei = dx_sigma_x * tmp_ei_no_std \
                + 0.5 * sigma_x * dx_gamma_x * (1 + tmp_erf)

            return ei, dx_ei


class UpperConfidenceBound(object):
    """ The "Upper Confidence Bound" (UCB) acquisition function.

    Parameters
    ----------
    model: gaussian-process object
        Gaussian process model, which models the return landscape
    kappa: optional, float >= 0.0
        Parameter controlling the exploration-exploitation trade-off. Larger
        values correspond to increased exploration. Defaults to 0
        which corresponds to pure exploitation.
    """
    def __init__(self, model, kappa=0.0):
        self.model = model
        self.kappa = kappa

    def __call__(self, x, baseline_value=0, compute_gradient=False,
                 *args, **kwargs):
        """ Returns the upper confidence bound at query point x.

        Parameters
        ----------
        x: array-like
            The position at which the upper confidence bound will be evaluated.
        baseline_value: float
            Baseline value, typically the maximum (actual) return observed
            so far during learning. Defaults to 0.
        compute_gradient: bool
            Whether the gradient at position x will be evaluated and returned.

        Returns
        -------
        ucb: float
            the upper confidence point of performance at query point x.
        gradient: array-like, optional (if compute_gradient is True)
            The gradient of the upper confidence bound at position x.
        """
        # Let GP predict mean and variance
        if compute_gradient:
            # Let GP model predict mean and variance and their gradients
            mu_x, sigma_x, dx_mu_x, dx_sigma_x = \
                self.model.predict(x, return_std=True, eval_gradient=True)
        else:
            # Let GP model predict mean and variance
            mu_x, sigma_x = self.model.predict(x, return_std=True)

        ucb = (mu_x - baseline_value) + self.kappa * sigma_x

        if not compute_gradient:
            return ucb
        else:
            # Compute additionally gradient of UCB with respect to x
            dx_ucb = dx_mu_x + self.kappa * dx_sigma_x
            return ucb, dx_ucb


class ZeroBaseline(object):
    """ Context-independent baseline wrapper for acquisition functions.

    Parameters
    ----------
    acquisition_function: acquisition_function object
        The actual acquisition function to be wrapped
    """

    def __init__(self, acquisition_function, *args, **kwargs):
        self.acquisition_function = acquisition_function

    def __call__(self, x, compute_gradient=False, *args, **kwargs):
        """ Returns the acquisition value at query point x.

        Note: In contrast to the actual acquisition functions, no baseline
              value needs to be provided. This wrapper sets the baseline to 0
              for all contexts.

        Parameters
        ----------
        x: array-like
            The position at which the acquisition value will be evaluated.
        compute_gradient: bool
            Whether the gradient at position x will be evaluated and returned.

        Returns
        -------
        acquisition_vale: float
            the acquisition value at query point x.
        gradient: array-like, optional (if compute_gradient is True)
            The gradient of the upper confidence bound at position x.
        """
        return self.acquisition_function(x, 0,
                                         compute_gradient=compute_gradient,
                                         *args, **kwargs)


class MaxLCBBaseline(object):
    """ Context-independent baseline wrapper for acquisition functions.

    This baseline uses the maximum lower confidence bound (LCB) predicted by a
    model of the return landscape as baseline. This is a non-parametric
    baseline which requires a separate optimization over the parameter space in
    order to find the parameters for which the LCB becomes maximal. It is thus
    computationally expensive.

    Parameters
    ----------
    acquisition_function: acquisition_function object
        The actual acquisition function to be wrapped
    model: gaussian-process object
        Gaussian process model, which models the return landscape
    optimize_callback: function
        Callback function which returns the maximum of the function that is
        passed as argument
    n_context_dims: int
        The number of context dimensions.
    kappa: optional, float >= 0.0
        Parameter controlling the how conservative the LCB is. Larger values
        correspond to increased conservativeness.
    """
    def __init__(self, acquisition_function, model, optimize_callback,
                 n_context_dims, kappa=0.0, *args, **kwargs):
        assert "context_fixed" not in kwargs  # For detecting old usage
        self.acquisition_function = acquisition_function
        self.model = model
        self.optimize_callback = optimize_callback
        self.n_context_dims = n_context_dims
        self.kappa = kappa
        # remember max_lcb and corresponding parameters for contexts
        self.context_cache = [[], [], []]

    def __call__(self, x, compute_gradient=False, *args, **kwargs):
        """ Returns the acquisition value at query point x.

        Note: In contrast to the actual acquisition functions, no baseline
              value needs to be provided. This wrapper chooses the baseline
              intelligently as the lower confidence bound (LCB) in the
              respective context.

        Parameters
        ----------
        x: array-like (n_samples, n_dimensions)
            The positions at which the acquisition value will be evaluated.
            Note that the first self.n_context_dims dimensions contain the
            respective context.
        compute_gradient: bool
            Whether the gradient at position x will be evaluated and returned.

        Returns
        -------
        acquisition_values: array-like (n_samples)
            the acquisition value at query point x.
        gradient: array-like, optional (if compute_gradient is True)
            The gradient of the upper confidence bound at position x.
        """
        x = np.atleast_2d(x)
        # Determine maximum LCB in contexts as baseline values
        contexts = x[:, :self.n_context_dims]
        baseline_values = self._get_max_lcb(contexts)

        # Delegate to actual acquisition function
        return self.acquisition_function(x, baseline_values,
                                         compute_gradient=compute_gradient,
                                         *args, **kwargs)

    def _get_max_lcb(self, contexts, return_params=False):
        """ Get maximal lower confidence bound in context.

        Either from cache or by performing a (costly) optimization.
        """
        # Deal with duplicate contexts
        assert contexts.shape[1] == 1, \
            "Multi-dimensional contexts not yet supported"
        contexts, inverse_idx = \
            np.unique([tuple(row) for row in contexts], return_inverse=True)
        contexts = contexts[:, None]

        # The baseline values for the unique contexts will be stored here
        baseline_values = np.empty(contexts.shape[0])

        # Since maximizing the LCB ist time-intensive, we cache its results
        if len(self.context_cache[0]) > 0:
            dist, index = self.nn.kneighbors(contexts, return_distance=True)
            dist = dist[:, 0]
            index = index[:, 0]
        else:
            dist = np.full_like(contexts, np.inf)  # first call without cache

        update_nn = False  # whether we need to refit NearestNeighbors
        for i in range(contexts.shape[0]):  # TODO: vectorize?
            if dist[i] == 0:  # cache hit
                baseline_values[i] = self.context_cache[1][index[i]]
            elif dist[i] <= 1e-8:
                # Perform LCB optimization from optimal parameters of similar
                # context
                _, baseline_values[i] = \
                    self._find_max_lcb(contexts[i],
                                       self.context_cache[2][index[i]])
            else:
                # Tabula rasa LCB optimization
                lcb_params, baseline_values[i] = \
                    self._find_max_lcb(contexts[i])

                # Update cache
                self.context_cache[0].append(contexts[i])
                self.context_cache[1].append(baseline_values[i])
                self.context_cache[2].append(lcb_params)
                update_nn = True

        if update_nn:
            self.nn = \
                NearestNeighbors(n_neighbors=1).fit(self.context_cache[0])

        return baseline_values[inverse_idx]

    def _find_max_lcb(self, context, x0=None):
        lcb_params = \
            self.optimize_callback(partial(self._lcb, context=context),
                                   x0=x0)
        baseline_value = self._lcb(lcb_params, context)

        return lcb_params, baseline_value

    def _lcb(self, parameter, context, *args, **kargs):
        """ Lower confidence bound on parameter in given context. """
        cx = np.hstack((context, parameter)).T
        mean, sigma = self.model.predict(cx, return_std=True)
        return mean[0] - self.kappa * sigma


class ContextualExpectedImprovement(object):
    """
    """
    def __init__(self, model, optimize_callback, n_context_dims, kappa=0.0,
                 boundaries=None, n_samples=250, n_different_contexts=50,
                 r=None, *args, **kwargs):
        self.model = model
        self.optimize_callback = optimize_callback
        self.n_context_dims = n_context_dims
        self.kappa = kappa
        self.boundaries = boundaries
        self.r = r

        self.ei = ExpectedImprovement(self.model)
        self.maxlcb = MaxLCBBaseline(self.ei, self.model, optimize_callback,
                                     n_context_dims=n_context_dims,
                                     kappa=kappa)

        self.corr = deepcopy(self.model.corr)

        self.samples = np.zeros((n_samples, self.model.X.shape[1]))
        self.samples[:, :n_context_dims] = \
            np.random.choice(np.random.uniform(2, 10, n_different_contexts),
                             n_samples)[:, None]  # TODO: Remove fixed bounds

        self.samples[:, n_context_dims:] = \
            np.random.uniform(0.0, 1.0,
                              (n_samples,
                               self.model.X.shape[1] - n_context_dims))

        self.samples[:, n_context_dims:] *= \
            self.boundaries[:, 1] - self.boundaries[:, 0]

    def __call__(self, x, baseline_value=0, compute_gradient=False,
                 *args, **kwargs):
        """ Returns expected improvement in skill performance at query point x.

        Note: This acquisition function does not require a baseline value. It
              thus defaults to 0.

        Parameters
        ----------
        x: array-like
            The position at which the acquisition value will be evaluated.
        baseline_value: float
            Baseline value, typically the maximum (actual) return observed
            so far during learning. Defaults to 0. Not required by this
            acquisition function.
        compute_gradient: bool
            Whether the gradient at position x will be evaluated and returned.

        Returns
        -------
        contextual_ei: float
            the contextual expected improvement at query point x.
        gradient: array-like, optional (if compute_gradient is True)
            The gradient of the upper confidence bound at position x.
        """
        x = np.asarray(x)

        # Compute weights
        self.corr.X = np.array(self.samples)
        self.corr.n_samples = self.corr.X.shape[0]
        dist = 1 - self.corr(self.model.theta_, np.atleast_2d(x))[0]
        weights = np.exp(-(self.r*dist)**2)
        weights_larger_threshold = weights > 0.01  # TODO: configurable thresh.

        # Compute Contextual Expected Improvement for sampling points with
        # weight larger than threshold (for reducing computation time)
        contextual_ei = np.zeros_like(weights)
        if not np.all(np.logical_not(weights_larger_threshold)):
            contextual_ei[weights_larger_threshold] = \
                self.maxlcb(self.samples[weights_larger_threshold])
            return np.sum(contextual_ei * weights) \
                / weights[weights_larger_threshold].sum()
        else:
            return 0.0


class EISP(object):
    """ Expected Improvement in Skill Performance" (EISP) acquisition function.

    Selects the query point which (under optimistic assumptions) increases the
    expected skill performance over the context space. For this, the current
    surrogate model is updated by a UCB estimate of the performance at the
    query point and it is (roughly) estimated how the expected return over the
    context space would increase if this UCB estimate would be real. The rough
    estimate is obtained by not checking all possible parameter values but only
    those that are optimal under the UCB criteria (for varying kappa values)
    in the old model. Furthermore, the improvement is only estimated based on a
    finite set of test contexts.

    This contextual acquisition function is motivated by the da Silva et al.,
    who used it for active context selection (by effectively hiding the
    influence of the policy parameters by means of a non-stationary covariance
    function).

    .. seealso::
        Bruno Castro da Silva, George Konidaris, Andrew Barto, "Active Learning
        of Parameterized Skills", ICML 2014

    Parameters
    ----------
    model: gaussian-process object
        Gaussian process model, which models the return landscape
    optimize_callback: function
        Callback function which returns the maximum of the function that is
        passed as argument
    n_context_dims: int
        The number of context dimensions.
    kappa: optional, float >= 0.0, default=0
        Parameter controlling the how optimistic the UCB is. Larger values
        correspond to increased optimism.
    kappa: optional, list of floats >= 0.0, default=[0.0, 0.5, 1.0, 2.5, 5.0]
        The UCB-values used for selecting the parameters which are checked
        later on for improvement in the skill performance
    n_test_contexts: optional, int>0, default=25
        The number of test contexts used for approximating the EISP
    """
    def __init__(self, model, optimize_callback, n_context_dims, kappa=0.0,
                 kappa_values=[0.0, 0.5, 1.0, 2.5, 5.0], n_test_contexts=25,
                 *args, **kwargs):
        self.model = model
        self.optimize_callback = optimize_callback
        self.n_context_dims = n_context_dims
        self.kappa = kappa
        self.kappa_values = kappa_values
        self.n_test_contexts = n_test_contexts

        # Fetch training data from model and sample test contexts
        self.X = self.model.X * self.model.X_std + self.model.X_mean
        self.y = self.model.y * self.model.y_std + self.model.y_mean

        # Sample the test contexts
        self.c_test = \
            [np.random.uniform(self.X[:, :n_context_dims].min(),
                               self.X[:, :n_context_dims].max())
             for _ in range(self.n_test_contexts)]

        # Determine optimal parameters (for different values of kappa) and
        # corresponding returns in test contexts based on model
        self.x_test = []
        self.r_test = []
        for c in self.c_test:
            for kappa_ in self.kappa_values:
                x_opt = self.optimize_callback(
                    lambda x, compute_gradient:
                        self._estimate_performance(self.model, c, x,
                                                   kappa_, compute_gradient))
                self.x_test += [x_opt]
                if kappa_ == 0.0:  # This is the baseline we compare to later
                    self.r_test += \
                        [self._estimate_performance(self.model, c, x_opt)]
        c_test = np.repeat(self.c_test, len(self.kappa_values))
        self.cx_test = np.hstack([np.atleast_2d(c_test).T,
                                  np.asarray(self.x_test)])

    def _estimate_performance(self, model, c, x, kappa=0.0,
                              compute_gradient=False):
        assert compute_gradient is False
        cx = np.hstack((c, x)).T
        pred, sigma = model.predict(cx, return_std=True)
        return pred + kappa * sigma

    def __call__(self, x, baseline_value=0, compute_gradient=False,
                 *args, **kwargs):
        """ Returns expected improvement in skill performance at query point x.

        Note: This acquisition function does not require a baseline value. It
              thus defaults to 0.

        Parameters
        ----------
        x: array-like
            The position at which the acquisition value will be evaluated.
        baseline_value: float
            Baseline value, typically the maximum (actual) return observed
            so far during learning. Defaults to 0. Not required by this
            acquisition function.
        compute_gradient: bool
            Whether the gradient at position x will be evaluated and returned.

        Returns
        -------
        eisp: float
            the expected improvement in skill performance at query point x.
        gradient: array-like, optional (if compute_gradient is True)
            The gradient of the upper confidence bound at position x.
        """
        x = np.asarray(x)
        # Let GP model predict mean and variance
        mu_x, sigma_x = self.model.predict(x, return_std=True)
        ucb_x = mu_x + self.kappa * sigma_x

        # Generate extended training set
        X_ = np.vstack((self.X, x[None, :]))
        y_ = np.append(self.y, ucb_x)

        # Normalize according to old GP model
        X_ = (X_ - self.model.X_mean) / self.model.X_std
        y_ = (y_ - self.model.y_mean) / self.model.y_std

        # update GP (assuming its optimistic prediction in x would be real)
        model_new = GaussianProcess(corr=deepcopy(self.model.corr),
                                    regr=self.model.regr,
                                    theta0=np.array(self.model.theta_),
                                    normalize=False,  # prenormalized
                                    nugget=self.model.nugget,
                                    thetaL=None, thetaU=None)
        for nugget_scale in range(10):
            try:
                model_new.fit(X_, y_)
                break
            except Exception, e:
                model_new.nugget *= 10
        if nugget_scale == 9:
            print e
            return 0  # TODO

        # Correct model to take actual normalization into account
        model_new.X_mean = self.model.X_mean
        model_new.X_std = self.model.X_std
        model_new.y_mean = self.model.y_mean
        model_new.y_std = self.model.y_std
        model_new.sigma2 *= self.model.y_std ** 2.

        # Compute expected return based on updated model for test contexts
        # and their respective parameter samples (where samples
        # have been chosen s.t. expected return for different kappas is
        # maximized)
        r_test_new = model_new.predict(self.cx_test)
        # Use in each context the maximal expected return in any of the
        # parameter samples
        r_test_new = r_test_new.reshape(-1, len(self.kappa_values)).T.max(0)
        # calculate improvement in test contexts
        return np.mean(r_test_new - np.array(self.r_test)[:, 0])


ACQUISITION_FUNCTIONS = {"ucb": UpperConfidenceBound,
                         "pi": ProbabilityOfImprovement,
                         "ei": ExpectedImprovement,
                         "random": UpperConfidenceBound}

CONTEXTUAL_ACQUISITION_FUNCTIONS = \
    {"eisp": EISP,
     "contextual_ei": ContextualExpectedImprovement}
CONTEXTUAL_ACQUISITION_FUNCTIONS.update(ACQUISITION_FUNCTIONS)
