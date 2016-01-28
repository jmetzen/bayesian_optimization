""" Acquisition functions which can be used in Bayesian optimization."""
# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>

import numpy as np
from scipy.stats import norm, entropy
from scipy.special import erf

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class AcquisitionFunction(object):
    """ Abstract base class for acquisition functions."""

    def set_boundaries(self, boundaries):
        """Sets boundaries of search space.

        Parameters
        ----------
        boundaries: ndarray-like, shape=(n_params_dims, 2)
            Box constraint on search space. boundaries[:, 0] defines the lower
            bounds on the dimensions, boundaries[:, 1] defines the upper
            bounds.
        """
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


class GPUpperConfidenceBound(AcquisitionFunction):
    """ The "GP Upper Confidence Bound" (UCB) acquisition function.

    Parameters
    ----------
    model: gaussian-process object
        Gaussian process model, which models the return landscape
    """
    def __init__(self, model, const):
        self.model = model
        self.const = const

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
        T = self.model.gp.X_train_.shape[0]
        D = self.model.gp.X_train_.shape[1]

        kappa = np.sqrt(4*(D + 1)*np.log(T) + self.const)

        # Determine model's predictive distribution (mean and
        # standard-deviation)
        mu_x, sigma_x = self.model.predictive_distribution(np.atleast_2d(x))

        ucb = (mu_x - incumbent) + kappa * sigma_x

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
                 n_samples_y=10, n_trial_points=500, rng_seed=0):
        self.model = model
        self.n_candidates = n_candidates
        self.n_gp_samples = n_gp_samples
        self.n_samples_y =  n_samples_y
        self.n_trial_points = n_trial_points
        self.rng_seed = rng_seed

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
            f_samples = np.random.RandomState(self.rng_seed).multivariate_normal(
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
                try:
                    y_samples = self.model.gp.sample_y(candidates)
                    self.X_candidate[i] = candidates[np.argmax(y_samples)]
                except np.linalg.LinAlgError:
                    self.X_candidate[i] = candidates[0]
        else:
            self.n_candidates = self.X_candidate.shape[0]

        # determine base entropy
        f_mean, f_cov = \
            self.model.gp.predict(self.X_candidate, return_cov=True)

        f_samples = np.random.RandomState(self.rng_seed).multivariate_normal(
            f_mean, f_cov, self.n_gp_samples).T
        p_max = np.bincount(np.argmax(f_samples, 0), minlength=f_mean.shape[0]) \
            / float(self.n_gp_samples)
        self.base_entropy = entropy(p_max)


class MinimalRegretSearch(AcquisitionFunction):
    """
    """
    def __init__(self, model, n_candidates=20, n_gp_samples=500,
                 n_samples_y=10, n_trial_points=500, point=False, rng_seed=0):
        self.model = model
        self.n_candidates = n_candidates
        self.n_gp_samples = n_gp_samples
        self.n_samples_y =  n_samples_y
        self.n_trial_points = n_trial_points
        self.point = point
        self.rng_seed = rng_seed

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
        regret_change: float
            the change in regret when sampling at x.
        """
        x = np.atleast_2d(x)

        a_MRS = np.empty((x.shape[0], self.n_samples_y))

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
            f_samples = np.random.RandomState(self.rng_seed).multivariate_normal(
                f_mean, f_cov + f_cov_delta, self.n_gp_samples).T

            # adapt for different means
            for j in range(self.n_samples_y):  # sample outcomes
                y_delta = np.sqrt(f_cov_query + self.model.gp.alpha)[:, 0] \
                    * self.percent_points[j]
                f_mean_delta = f_cov_cross.dot(f_cov_query_inv).dot(y_delta)

                f_mean_j = f_mean + f_mean_delta
                f_samples_j = f_samples + f_mean_delta[:, np.newaxis]

                if self.point:
                    opt_ind = f_mean_j.argmax()
                    regrets = f_samples_j.max(0) - f_samples_j[opt_ind, :]
                    a_MRS[i - self.n_candidates, j] = \
                        np.mean(self.base_regrets - regrets)
                else:
                    bincount = np.bincount(np.argmax(f_samples_j, 0),
                                           minlength=f_mean.shape[0])
                    p_max =  bincount / float(self.n_gp_samples)

                    regrets = f_samples_j.max(0) - f_samples_j
                    a_MRS[i - self.n_candidates, j] = \
                        (np.mean(self.base_regrets - regrets, 1) * p_max).sum()

        return a_MRS.mean(1)  # Mean over y_delta

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
                try:
                    y_samples = self.model.gp.sample_y(candidates)
                    self.X_candidate[i] = candidates[np.argmax(y_samples)]
                except np.linalg.LinAlgError:
                    self.X_candidate[i] = candidates[0]
        else:
            self.n_candidates = self.X_candidate.shape[0]

        # determine base regret
        f_mean, f_cov = \
            self.model.gp.predict(self.X_candidate, return_cov=True)
        f_samples = np.random.RandomState(self.rng_seed).multivariate_normal(
            f_mean, f_cov, self.n_gp_samples).T

        if self.point:
            opt_ind = f_mean.argmax()
            self.base_regrets  = f_samples.max(0) - f_samples[opt_ind, :]
        else:
            self.base_regrets = f_samples.max(0) - f_samples


class ContextualEntropySearchLocal(AcquisitionFunction):
    def __init__(self, model, n_context_dims,
                 n_candidates=20, n_gp_samples=500,
                 n_samples_y=10, n_trial_points=100):
        self.model = model
        self.n_context_dims = n_context_dims

        self.n_candidates = n_candidates
        self.n_gp_samples = n_gp_samples
        self.n_samples_y =  n_samples_y
        self.n_trial_points = n_trial_points

    def __call__(self, x, incumbent=0, *args, **kwargs):
        boundaries_i = np.copy(self.boundaries)
        boundaries_i[:self.n_context_dims] = \
            x[:self.n_context_dims, np.newaxis]
        entropy_search_fixed_context = \
            EntropySearch(self.model, self.n_candidates, self.n_gp_samples,
                          self.n_samples_y, self.n_trial_points)
        entropy_search_fixed_context.set_boundaries(boundaries_i)
        return entropy_search_fixed_context(x)[0]

    def set_boundaries(self, boundaries):
        self.boundaries = boundaries


class ContextualEntropySearch(AcquisitionFunction):
    """
    n_context_samples: int, default: 20
        The number of context sampled from context space on which sample
        policies are evaluated
    """
    def __init__(self, model, n_context_dims, n_context_samples,
                 n_candidates=20, n_gp_samples=500,
                 n_samples_y=10, n_trial_points=100, n_neighbors=20):
        self.model = model
        self.n_context_dims = n_context_dims
        self.n_context_samples = n_context_samples

        self.n_candidates = n_candidates
        self.n_gp_samples = n_gp_samples
        self.n_samples_y =  n_samples_y
        self.n_trial_points = n_trial_points
        self.n_neighbors = n_neighbors

    def __call__(self, x, incumbent=0, *args, **kwargs):
        ind = list(self.nbrs.kneighbors(x[np.newaxis, :self.n_context_dims],
                                        return_distance=False)[0])

        entropy_reductions = \
            [self.entropy_search_ensemble[i](x)[0] for i in ind]

        return np.mean(entropy_reductions)

    def set_boundaries(self, boundaries):
        self._sample_contexts(boundaries[:self.n_context_dims])

        # XXX: do that lazily
        self.entropy_search_ensemble = []
        for i in range(self.n_context_samples):
            boundaries_i = np.copy(boundaries)
            boundaries_i[:self.n_context_dims] = \
                self.context_samples[i][:, np.newaxis]
            entropy_search_fixed_context = \
                EntropySearch(self.model, self.n_candidates, self.n_gp_samples,
                              self.n_samples_y, self.n_trial_points)
            entropy_search_fixed_context.set_boundaries(boundaries_i)

            self.entropy_search_ensemble.append(entropy_search_fixed_context)

    def _sample_contexts(self, context_boundaries):
        # Determine samples at which CES will be evaluated by
        # 1. uniform random sampling
        self.context_samples = \
            np.random.uniform(context_boundaries[:, 0],
                              context_boundaries[:, 1],
                              (self.n_context_samples*25, self.n_context_dims))
        # 2. subsampling via k-means clustering
        kmeans = KMeans(n_clusters=self.n_context_samples, n_jobs=1)
        self.context_samples = \
            kmeans.fit(self.context_samples).cluster_centers_

        # Initialize nearest neighbors query structure which takes GP
        # length-scales into account
        # XXX: Kernel structure is hard-coded
        length_scales = \
            self.model.gp.kernel_.k1.k2.length_scale[:self.n_context_dims]
        self.nbrs = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm='ball_tree', metric="mahalanobis",
            metric_params={"VI": np.linalg.inv(np.diag(length_scales))})
        self.nbrs.fit(self.context_samples)


ACQUISITION_FUNCTIONS = {
    "PI": ProbabilityOfImprovement,
    "EI": ExpectedImprovement,
    "UCB": UpperConfidenceBound,
    "GREEDY": Greedy,
    "RANDOM": Random,
    "EntropySearch": EntropySearch,
    "MinimalRegretSearch": MinimalRegretSearch,
    "ContextualEntropySearch": ContextualEntropySearch,
    "ContextualEntropySearchLocal": ContextualEntropySearchLocal}


def create_acquisition_function(name, model, **kwargs):
    return ACQUISITION_FUNCTIONS[name](model, **kwargs)
