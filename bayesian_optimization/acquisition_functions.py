""" Acquisition functions which can be used in Bayesian optimization."""
# Author: Jan Hendrik Metzen <janmetzen@mailbox.org>

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
    *n_candidates* data points (representers) for the position of the true
    maximum (p_max) are selected. 
    From the GP model, *n_gp_samples* samples from the posterior are
    drawn and their entropy is computed. For each query point, the GP model is
    updated assuming *n_samples_y* outcomes (according to the current GP model).
    The change of entropy resulting from this assumed outcomes is computed and
    the query point which minimizes the entropy of p_max is selected.

    See also:
        Hennig, Philipp and Schuler, Christian J. 
        Entropy Search for Information-Efficient Global Optimization. 
        JMLR, 13:1809–1837, 2012.
    """
    def __init__(self, model, n_candidates=20, n_gp_samples=500,
                 n_samples_y=10, n_trial_points=500, rng_seed=0):
        self.model = model
        self.n_candidates = n_candidates
        self.n_gp_samples = n_gp_samples
        self.n_samples_y =  n_samples_y
        self.n_trial_points = n_trial_points
        self.rng_seed = rng_seed

        # We use an equidistant grid instead of sampling from the 1d normal
        # distribution over y
        equidistant_grid = np.linspace(0.0, 1.0, 2 * self.n_samples_y +1)[1::2]
        self.percent_points = norm.ppf(equidistant_grid)

    def __call__(self, x, incumbent=0, *args, **kwargs):
        """ Returns the change in entropy of p_max when sampling at x.

        Parameters
        ----------
        x: array-like
            The position(s) at which the upper confidence bound will be evaluated.
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

        # Evaluate mean and covariance of GP at all representer points and
        # points x where MRS will be evaluated
        f_mean_all, f_cov_all = \
            self.model.gp.predict(np.vstack((self.X_candidate, x)),
                                  return_cov=True)
        f_mean = f_mean_all[:self.n_candidates]
        f_cov = f_cov_all[:self.n_candidates, :self.n_candidates]

        # Iterate over all x[i] at which we will evaluate the acquisition 
        # function (often x.shape[0]=1)
        for i in range(self.n_candidates, self.n_candidates+x.shape[0]):
            # Simulate change of covariance (f_cov_delta) for a sample at x[i], 
            # which actually would not depend on the observed value y[i]
            f_cov_query = f_cov_all[[i]][:, [i]]
            f_cov_cross = f_cov_all[:self.n_candidates, [i]]
            f_cov_query_inv = np.linalg.inv(f_cov_query)
            f_cov_delta = -np.dot(np.dot(f_cov_cross, f_cov_query_inv),
                                  f_cov_cross.T)

            # precompute samples from GP posterior for non-modified mean
            f_samples = np.random.RandomState(self.rng_seed).multivariate_normal(
                f_mean, f_cov + f_cov_delta, self.n_gp_samples).T

            # adapt for different outcomes y_i[j] of the query at x[i]
            for j in range(self.n_samples_y):  
                # "sample" outcomes y_i[j] (more specifically where on the 
                # Gaussian distribution over y_i[j] we would end up)
                y_delta = np.sqrt(f_cov_query + self.model.gp.alpha)[:, 0] \
                    * self.percent_points[j]
                # Compute change in GP mean at representer points
                f_mean_delta = f_cov_cross.dot(f_cov_query_inv).dot(y_delta)

                # Adapt samples to changes in GP posterior mean
                f_samples_j = f_samples + f_mean_delta[:, np.newaxis]
                # Count frequency of the candidates being the optima in the samples
                p_max = np.bincount(np.argmax(f_samples_j, 0),
                                    minlength=f_mean.shape[0]) \
                    / float(self.n_gp_samples)
                # Determing entropy of distr. p_max and compare to base entropy
                a_ES[i - self.n_candidates, j] = \
                    self.base_entropy - entropy(p_max)

         # Average entropy change over the different  assumed outcomes y_i[j]
        return a_ES.mean(1) 

    def set_boundaries(self, boundaries, X_candidate=None):
        """Sets boundaries of search space.

        This method is assumed to be called once before running the
        optimization of the acquisition function.

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
            # being selected as representer points using (discretized) Thompson
            # sampling
            self.X_candidate = \
                np.empty((self.n_candidates, boundaries.shape[0]))
            for i in range(self.n_candidates):
                # Select n_trial_points data points uniform randomly
                candidates = np.random.uniform(
                    boundaries[:, 0], boundaries[:, 1],
                    (self.n_trial_points, boundaries.shape[0]))
                # Sample function from GP posterior and select the trial points
                # which maximizes the posterior sample as representer points 
                try:
                    y_samples = self.model.gp.sample_y(candidates)
                    self.X_candidate[i] = candidates[np.argmax(y_samples)]
                except np.linalg.LinAlgError:  # This should happen very infrequently
                    self.X_candidate[i] = candidates[0]
        else:
            self.n_candidates = self.X_candidate.shape[0]

        ### Determine base entropy
        # Draw n_gp_samples functions from GP posterior
        f_mean, f_cov = \
            self.model.gp.predict(self.X_candidate, return_cov=True)
        f_samples = np.random.RandomState(self.rng_seed).multivariate_normal(
            f_mean, f_cov, self.n_gp_samples).T
        # Count frequency of the candidates being the optima in the samples
        p_max = np.bincount(np.argmax(f_samples, 0), minlength=f_mean.shape[0]) \
            / float(self.n_gp_samples)
        # Determing entropy of distr. p_max
        self.base_entropy = entropy(p_max)


class MinimalRegretSearch(AcquisitionFunction):
    """ Minimum regret search acquisition function

    This acquisition function samples at the position which reduces the expected
    simple regret of the recommended point (maximum of GP mean) the most.  For
    this *n_candidates* data points (representers) for the position of  the true
    maximum (p_max) are selected.  From the GP model, *n_gp_samples* samples
    from the posterior are drawn and their expected simple regret is computed.
    For each query point, the GP model is updated assuming *n_samples_y*
    outcomes (according to the current GP model). The change of expected simple
    regret resulting from this assumed outcomes is computed and the query point
    which minimizes the expected simple regret is selected.

    If *point* is True, MRS_point is used, which is faster but slightly less 
    performant, while otherwise, the full MRS is used.

    See also:
        Metzen, Jan Hendrik
        Minimum Regret Search for Single- and Multi-Task Optimization
        ICML, 2016
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

        # Evaluate mean and covariance of GP at all representer points and
        # points x where MRS will be evaluated
        f_mean_all, f_cov_all = \
            self.model.gp.predict(np.vstack((self.X_candidate, x)),
                                  return_cov=True)
        f_mean = f_mean_all[:self.n_candidates]
        f_cov = f_cov_all[:self.n_candidates, :self.n_candidates]

        # Iterate over all x[i] at which we will evaluate the acquisition 
        # function (often x.shape[0]=1)
        for i in range(self.n_candidates, self.n_candidates+x.shape[0]):
            # Simulate change of covariance (f_cov_delta) for a sample at x[i], 
            # which actually would not depend on the observed value y_i
            f_cov_query = f_cov_all[[i]][:, [i]]
            f_cov_cross = f_cov_all[:self.n_candidates, [i]]
            f_cov_query_inv = np.linalg.inv(f_cov_query)
            f_cov_delta = -np.dot(np.dot(f_cov_cross, f_cov_query_inv),
                                  f_cov_cross.T)

            # precompute samples from GP posterior for non-modified mean
            f_samples = np.random.RandomState(self.rng_seed).multivariate_normal(
                f_mean, f_cov + f_cov_delta, self.n_gp_samples).T

            # adapt for different outcomes y_i[j] of the query at x[i]
            for j in range(self.n_samples_y):
                # "sample" outcomes y_i[j] (more specifically where on the 
                # Gaussian distribution over y_i[j] we would end up)
                y_delta = np.sqrt(f_cov_query + self.model.gp.alpha)[:, 0] \
                    * self.percent_points[j]
                # Compute change in GP mean at representer points
                f_mean_delta = f_cov_cross.dot(f_cov_query_inv).dot(y_delta)

                # Adapt GP posterior mean and samples for modified mean
                f_mean_j = f_mean + f_mean_delta
                f_samples_j = f_samples + f_mean_delta[:, np.newaxis]

                if self.point:
                    # MRS point:
                    # Select representer point for the j-th assumed outcome 
                    # y_i[j] as the maximum of the GP mean
                    opt_ind = f_mean_j.argmax()  # selected representer point
                    # Compute regret of selected representer point compared
                    # to optimal representer point in the respective GP samples
                    regrets = f_samples_j.max(0) - f_samples_j[opt_ind, :]
                    # Store mean of regret change (to base regret) 
                    # over all GP samples as MRS_point
                    a_MRS[i - self.n_candidates, j] = \
                        np.mean(self.base_regrets - regrets)
                else:
                    # MRS:
                    # Determine frequency of how often each representer point 
                    # is the optimum in the GP samples
                    bincount = np.bincount(np.argmax(f_samples_j, 0),
                                           minlength=f_mean.shape[0])
                    p_max =  bincount / float(self.n_gp_samples)
                    # Compute the incurred regrets for ALL representer points 
                    # relative to the respective optima of the GP samples
                    regrets = f_samples_j.max(0) - f_samples_j
                    # Compute mean of regret change (to base regret) 
                    # over all GP samples
                    mean_regrets = np.mean(self.base_regrets - regrets, 1)
                    # Compute weighted mean over all representer points where
                    # the probability of being the optimum (p_max) of a 
                    # representer point is used as weight.
                    a_MRS[i - self.n_candidates, j] = \
                        (mean_regrets * p_max).sum()

        # Average MRS over the different assumed outcomes y_i[j]
        return a_MRS.mean(1) 

    def set_boundaries(self, boundaries, X_candidate=None):
        """Sets boundaries of search space.

        This method is assumed to be called once before running the
        optimization of the acquisition function.

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
            # being selected as representer points using (discretized) Thompson
            # sampling
            self.X_candidate = \
                np.empty((self.n_candidates, boundaries.shape[0]))
            for i in range(self.n_candidates):
                # Select n_trial_points data points uniform randomly
                candidates = np.random.uniform(
                    boundaries[:, 0], boundaries[:, 1],
                    (self.n_trial_points, boundaries.shape[0]))
                # Sample function from GP posterior and select the trial points
                # which maximizes the posterior sample as representer points 
                try:
                    y_samples = self.model.gp.sample_y(candidates)
                    self.X_candidate[i] = candidates[np.argmax(y_samples)]
                except np.linalg.LinAlgError:  # This should happen very infrequently
                    self.X_candidate[i] = candidates[0]
        else:
            self.n_candidates = self.X_candidate.shape[0]

        ### Determine base regret
        # Draw n_gp_samples functions from GP posterior
        f_mean, f_cov = \
            self.model.gp.predict(self.X_candidate, return_cov=True)
        f_samples = np.random.RandomState(self.rng_seed).multivariate_normal(
            f_mean, f_cov, self.n_gp_samples).T

        if self.point:
            # MRS point:
            # Compute the incurred regret of the representer points that 
            # maximizes the GP mean relative to the respective optima of
            # the GP samples
            opt_ind = f_mean.argmax()  # selected representer point
            self.base_regrets  = f_samples.max(0) - f_samples[opt_ind, :]
        else:
            # MRS:
            # Compute the incurred regrets for ALL representer points 
            # relative to the respective optima of the GP samples
            self.base_regrets = f_samples.max(0) - f_samples


ACQUISITION_FUNCTIONS = {
    "PI": ProbabilityOfImprovement,
    "EI": ExpectedImprovement,
    "UCB": UpperConfidenceBound,
    "GREEDY": Greedy,
    "RANDOM": Random,
    "EntropySearch": EntropySearch,
    "MinimalRegretSearch": MinimalRegretSearch}


def create_acquisition_function(name, model, **kwargs):
    return ACQUISITION_FUNCTIONS[name](model, **kwargs)
