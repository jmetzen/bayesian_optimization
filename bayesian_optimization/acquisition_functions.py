""" Acquisition functions which can be used in Bayesian optimization."""
# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>

import numpy as np
from scipy.stats import norm
from scipy.special import erf


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
        self.kappa = kappa

    def __call__(self, x, incumbent, *args, **kwargs):
        """Returns the probability of improvement at query point x.

        Parameters
        ----------
        x: array-like
            The position at which the probability of improvement will be
            evaluated.
        incumbent: float
            Baseline value, typically the maximum (actual) return observed
            so far during learning.

        Returns
        -------
        pi: float
            The probability that query point x yields a better return than
            the best seen so far (i.e., incumbent)
        """
        # Determine model's predictive distribution (mean and
        # standard-deviation)
        mu_x, sigma_x = self.model.predictive_distribution(x)

        gamma_x = (mu_x - (incumbent - self.kappa)) / sigma_x
        pi = norm.cdf(gamma_x)

        return pi


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
        self.kappa = kappa

    def __call__(self, x, incumbent, *args, **kwargs):
        """Returns the expected improvement at query point x.

        Parameters
        ----------
        x: array-like
            The position at which the expected improvement will be evaluated.
        incumbent: float
            Baseline value, typically the maximum (actual) return observed
            so far during learning.

        Returns
        -------
        ei: float
            The expected improvement at query point x over the current best
            (i.e, incumbent)
        """
        # Determine model's predictive distribution (mean and
        # standard-deviation)
        mu_x, sigma_x = self.model.predictive_distribution(x)

        gamma_x = (mu_x - (incumbent - self.kappa)) / sigma_x
        # Compute EI based on some temporary variables that can be reused in
        # gradient computation
        tmp_erf = erf(gamma_x / np.sqrt(2))
        tmp_ei_no_std = 0.5*gamma_x * (1 + tmp_erf) \
            + np.exp(-gamma_x**2/2)/np.sqrt(2*np.pi)
        ei = sigma_x * tmp_ei_no_std

        return ei


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

    def __call__(self, x, incumbent=0, *args, **kwargs):
        """ Returns the upper confidence bound at query point x.

        Parameters
        ----------
        x: array-like
            The position at which the upper confidence bound will be evaluated.
        incumbent: float
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
        # Determine model's predictive distribution (mean and
        # standard-deviation)
        mu_x, sigma_x = self.model.predictive_distribution(x)

        ucb = (mu_x - incumbent) + self.kappa * sigma_x

        return ucb
