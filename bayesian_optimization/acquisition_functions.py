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
        mu_x, sigma_x = self.model.predictive_distribution(x)

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
        mu_x, sigma_x = self.model.predictive_distribution(x)

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
        mu_x, sigma_x = self.model.predictive_distribution(x)

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
    """
    """
    def __init__(self, model, n_sample_points=20, n_gp_samples=500,
                 n_samples_y=10):
        self.model = model
        self.n_sample_points = n_sample_points
        self.n_gp_samples = n_gp_samples
        self.n_samples_y =  n_samples_y

        equidistant_grid = np.linspace(0.0, 1.0, 2 * self.n_samples_y +1)[1::2]
        self.percent_points = norm.ppf(equidistant_grid)

    def __call__(self, x, incumbent=0, *args, **kwargs):
        """ Returns the upper confidence bound at query point x.

        Parameters
        ----------
        x: array-like
            The position at which the upper confidence bound will be evaluated.
        incumbent: float
            Baseline value, typically the maximum (actual) return observed
            so far during learning. Defaults to 0.
        """
        x = np.atleast_2d(x)

        a_ES = np.empty((x.shape[0], self.n_samples_y))

        X_joint = np.vstack((self.X_samples, x))

        f_mean_all, f_cov_all = \
            self.model.gp.predict(np.atleast_2d(X_joint), return_cov=True)
        f_mean = f_mean_all[:self.n_sample_points]
        f_cov = f_cov_all[:self.n_sample_points, :self.n_sample_points]

        # base entropy
        f_samples = np.random.multivariate_normal(f_mean, f_cov,
                                                  self.n_gp_samples).T
        p_max = np.bincount(np.argmax(f_samples, 0), minlength=f_mean.shape[0]) \
            / float(self.n_gp_samples)
        base_entropy = entropy(p_max)

        # XXX: Vectorize
        for i in range(self.n_sample_points, self.n_sample_points+x.shape[0]):
            f_cov_query = f_cov_all[[i]][:, [i]]
            f_cov_query_inv = np.linalg.inv(f_cov_query)
            f_cov_delta = -f_cov_all[:self.n_sample_points, [i]].dot(f_cov_query_inv).dot(f_cov_all[[i], :self.n_sample_points])

            # precompute samples for non-modified mean
            f_cov_mod = f_cov + f_cov_delta
            f_samples = np.random.multivariate_normal(f_mean, f_cov_mod,
                                                      self.n_gp_samples).T

            for j in range(self.n_samples_y):  # sample outcomes
                y_delta = np.sqrt(f_cov_query + self.model.gp.alpha)[:, 0] \
                    * self.percent_points[j]
                f_mean_delta = f_cov_all[:self.n_sample_points, [i]].dot(f_cov_query_inv).dot(y_delta)

                f_samples_j = f_samples + f_mean_delta[:, np.newaxis]
                p_max = np.bincount(np.argmax(f_samples_j, 0), minlength=f_mean.shape[0]) \
                    / float(self.n_gp_samples)
                a_ES[i - self.n_sample_points, j] = base_entropy - entropy(p_max)

        return a_ES.mean(1)

    def set_boundaries(self, boundaries):
        self.X_samples = \
            np.empty((self.n_sample_points, boundaries.shape[0]))
        for i in range(self.n_sample_points):
            # XXX: n_candidates
            candidates = np.random.uniform(boundaries[:, 0],
                                           boundaries[:, 1],
                                           (500, boundaries.shape[0]))
            y_samples = self.model.gp.sample_y(candidates)
            self.X_samples[i] = candidates[np.argmax(y_samples)]


ACQUISITION_FUNCTIONS = {
    "PI": ProbabilityOfImprovement,
    "EI": ExpectedImprovement,
    "UCB": UpperConfidenceBound,
    "GREEDY": Greedy,
    "RANDOM": Random,
    "EntropySearch": EntropySearch}


def create_acquisition_function(name, model, **kwargs):
    return ACQUISITION_FUNCTIONS[name](model, **kwargs)
