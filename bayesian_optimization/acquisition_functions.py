""" Acquisition functions which can be used in Bayesian optimization."""
# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>

import numpy as np
from scipy.stats import norm, entropy
from scipy.special import erf


class AcquisitionFunction(object):

    def set_boundaries(self, boundaries):
        pass


class ProbabilityOfImprovement(AcquisitionFunction):
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
        mu_x, sigma_x = self.model.predictive_distribution(np.atleast_2d(x))

        gamma_x = (mu_x - (incumbent + self.kappa)) / sigma_x
        pi = norm.cdf(gamma_x)

        return pi


class ExpectedImprovement(AcquisitionFunction):
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
        mu_x, sigma_x = self.model.predictive_distribution(np.atleast_2d(x))

        gamma_x = (mu_x - (incumbent + self.kappa)) / sigma_x
        # Compute EI based on some temporary variables that can be reused in
        # gradient computation
        tmp_erf = erf(gamma_x / np.sqrt(2))
        tmp_ei_no_std = 0.5*gamma_x * (1 + tmp_erf) \
            + np.exp(-gamma_x**2/2)/np.sqrt(2*np.pi)
        ei = sigma_x * tmp_ei_no_std

        return ei


class UpperConfidenceBound(AcquisitionFunction):
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

        Returns
        -------
        ucb: float
            the upper confidence point of performance at query point x.
        """
        # Determine model's predictive distribution (mean and
        # standard-deviation)
        mu_x, sigma_x = self.model.predictive_distribution(np.atleast_2d(x))

        ucb = (mu_x - incumbent) + self.kappa * sigma_x

        return ucb


class Greedy(UpperConfidenceBound):
    """ The greedy acquisition function

    This acquisition function always selects the query point with the maximal
    predicted value while ignoring uncertainty altogether.

    Parameters
    ----------
    model: gaussian-process object
        Gaussian process model, which models the return landscape
    """
    def __init__(self, model, **kwargs):
        super(Greedy, self).__init__(model, kappa=0)


class Random(AcquisitionFunction):
    """ The random acquisition function. """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x, incumbent=0, *args, **kwargs):
        """ Returns a random acquisition value independent of query point x.

        Parameters
        ----------
        x: array-like
            The position at which the upper confidence bound will be evaluated.
        incumbent: float
            Baseline value, typically the maximum (actual) return observed
            so far during learning. Defaults to 0.

        Returns
        -------
        random: float
            a random value independent of actual query point x.
        """
        return np.random.random()


class EntropySearch(AcquisitionFunction):
    """ Entropy search acquisition function

    This acquisition function samples at the position which reveals the maximal
    amount of information about the true position of the maximum. For this
    *n_candidates* data points for the position of the true maximum (p_max) are
    selected. From the GP model, *n_gp_samples* samples from the posterior are
    drawn and their entropy is computed. For each query point, the GP model is
    updated assuming *n_samples_y* outcomes (according to the current GP model).
    The change of entropy resulting from this assumed outcomes is computed and
    the query point which minimizes the entropy of p_max is selected.
    """
    def __init__(self, model, n_candidates=20, n_gp_samples=500,
                 n_samples_y=10, n_trial_points=500):
        self.model = model
        self.n_candidates = n_candidates
        self.n_gp_samples = n_gp_samples
        self.n_samples_y =  n_samples_y
        self.n_trial_points = n_trial_points

        equidistant_grid = np.linspace(0.0, 1.0, 2 * self.n_samples_y +1)[1::2]
        self.percent_points = norm.ppf(equidistant_grid)

    def __call__(self, x, incumbent=0, *args, **kwargs):
        """ Returns the change in entropy of p_max when sampling at x.

        Parameters
        ----------
        x: array-like
            The position at which the upper confidence bound will be evaluated.
        incumbent: float
            Baseline value, typically the maximum (actual) return observed
            so far during learning. Defaults to 0. [Not used by this acquisition
            function]

        Returns
        -------
        entropy_change: float
            the change in entropy of p_max when sampling at x.
        """
        x = np.atleast_2d(x)

        a_ES = np.empty((x.shape[0], self.n_samples_y))

        f_mean_all, f_cov_all = \
            self.model.gp.predict(np.vstack((self.X_candidate, x)),
                                  return_cov=True)
        f_mean = f_mean_all[:self.n_candidates]
        f_cov = f_cov_all[:self.n_candidates, :self.n_candidates]

        # XXX: Vectorize
        for i in range(self.n_candidates, self.n_candidates+x.shape[0]):
            f_cov_query = f_cov_all[[i]][:, [i]]
            f_cov_cross = f_cov_all[:self.n_candidates, [i]]
            f_cov_query_inv = np.linalg.inv(f_cov_query)
            f_cov_delta = -np.dot(np.dot(f_cov_cross, f_cov_query_inv),
                                  f_cov_cross.T)

            # precompute samples for non-modified mean
            f_cov_mod = f_cov + f_cov_delta
            f_samples = np.random.multivariate_normal(
                f_mean, f_cov + f_cov_delta, self.n_gp_samples).T

            # adapt for different means
            for j in range(self.n_samples_y):  # sample outcomes
                y_delta = np.sqrt(f_cov_query + self.model.gp.alpha)[:, 0] \
                    * self.percent_points[j]
                f_mean_delta = f_cov_cross.dot(f_cov_query_inv).dot(y_delta)

                f_samples_j = f_samples + f_mean_delta[:, np.newaxis]
                p_max = np.bincount(np.argmax(f_samples_j, 0),
                                    minlength=f_mean.shape[0]) \
                    / float(self.n_gp_samples)
                a_ES[i - self.n_candidates, j] = \
                    self.base_entropy - entropy(p_max)

        return a_ES.mean(1)

    def set_boundaries(self, boundaries, X_candidate=None):
        """Sets boundaries of search space.

        Parameters
        ----------
        boundaries: ndarray-like, shape=(n_params_dims, 2)
            Box constraint on search space. boundaries[:, 0] defines the lower
            bounds on the dimensions, boundaries[:, 1] defines the upper
            bounds.
        """
        self.X_candidate = X_candidate
        if self.X_candidate is None:
            # Sample n_candidates data points, which are checked for
            # being the location of p_max
            self.X_candidate = \
                np.empty((self.n_candidates, boundaries.shape[0]))
            for i in range(self.n_candidates):
                # Select n_trial_points data points uniform randomly
                candidates = np.random.uniform(
                    boundaries[:, 0], boundaries[:, 1],
                    (self.n_trial_points, boundaries.shape[0]))
                # Sample function from GP posterior and select the trial points
                # which maximizes the posterior sample as candidate
                y_samples = self.model.gp.sample_y(candidates)
                self.X_candidate[i] = candidates[np.argmax(y_samples)]

        # determine base entropy
        f_mean, f_cov = \
            self.model.gp.predict(self.X_candidate, return_cov=True)

        f_samples = np.random.multivariate_normal(f_mean, f_cov,
                                                  self.n_gp_samples).T
        p_max = np.bincount(np.argmax(f_samples, 0), minlength=f_mean.shape[0]) \
            / float(self.n_gp_samples)
        self.base_entropy = entropy(p_max)


ACQUISITION_FUNCTIONS = {
    "PI": ProbabilityOfImprovement,
    "EI": ExpectedImprovement,
    "UCB": UpperConfidenceBound,
    "GREEDY": Greedy,
    "RANDOM": Random,
    "EntropySearch": EntropySearch}


def create_acquisition_function(name, model, **kwargs):
    return ACQUISITION_FUNCTIONS[name](model, **kwargs)
