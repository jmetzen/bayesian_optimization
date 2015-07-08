# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial.distance import cdist
from bolero.optimizer import Optimizer
from bolero.optimizer import fmin
from bolero.utils.log import HideExtern
from bolero.utils.validation import check_random_state, check_feedback
from .model import GaussianProcessModel
from .acquisition_functions import ACQUISITION_FUNCTIONS


class BayesianOptimizer(Optimizer):
    """Bayesian Optimization.

    This optimizer models the landscape of the function to be optimized
    internally by a Gaussian process (GP) and evaluates always those parameters
    which are considered as global optimum of an acquisition function defined
    over this GP. Different acquisition functions and optimizers can be used
    internally.

    Bayesian optimization aims at reducing the number of evaluations of the
    actual function, which is assumed to be costly. To achieve this, a large
    computational budget is allocated at modeling the true function and finding
    potentially optimal positions based on this model.

    .. seealso:: Brochu, Cora, de Fretas
                 "A tutorial on Bayesian optimization of expensive cost
                  functions, with application to active user modeling and
                  hierarchical reinforcement learning"

    Parameters
    ----------
    boundaries : list of pair of floats
        The boundaries of the parameter space in which the optimum is search.

    acquisition_function : optional, string
        String identifying the acquisition function to be used. Supported are
        * "ucb": Upper-Confidence Bound (default)
        * "pi": Probability of Improvement
        * "ei": Expected Improvement
        * "random": Randomly choose point to sample

    optimizer: optional, string
        The global optimizer used internally to find the global optimum of the
        respective acquisition function. Supported are:
        * "direct": Using the DIRECT optimizer
        * "lbfgs": Use L-BFGS optimizer
        * "direct+lbfgs": Using the DIRECT optimizer with subsequent L-BFGS
        * "random": Randomly search the parameter space
        * "random+lbfgs": Randomly search parameter space with subsequent
                          L-BFGS
        * "cmaes": Using CMA-ES
        * "cmaes+lbfgs": Using CMA-ES with subsequent L-BFGS

    optimizer_kwargs: optional, dict
        Optional keyword arguments passed to optimize function. Currently
        supported is maxf (an integer, default=100), which determines the
        maximum number of function evaluations

    acq_fct_kwargs: option, dict
        Optional keyword arguments passed to acquisition function. Currently
        supported is kappa (a float >= 0.0, default=0.0), which handles
        the exploration-exploitation trade-off in the acquisition function.

    gp_kwargs: optional, dict
        Optional configuration parameters passed to Gaussian process model.
        See documentation of GaussianProcessModel for details.

    value_transform : optional, function
        Function mapping actual values to values internally by GP modelled.
        For instance, in some situations, a log-transform might be useful.
        Should be a monotonic increasing function. Defaults to identity.

    approx_grad : optional, bool
        Whether the gradient will be approximated numerically during
        optimization or computed analytically. Defaults to False.

    random_state : optional, int
        Seed for the random number generator.
    """
    def __init__(self, boundaries, acquisition_function="ucb",
                 optimizer="direct+lbfgs", optimizer_kwargs={},
                 acq_fct_kwargs={}, gp_kwargs={},
                 value_transform=lambda x: x,
                 approx_grad=True, random_state=None, **kwargs):
        assert acquisition_function in ACQUISITION_FUNCTIONS
        assert isinstance(boundaries, list), \
            "Boundaries must be passed as a list of tuples (pairs)."

        self.boundaries = boundaries
        self.value_transform = value_transform
        if isinstance(self.value_transform, basestring):
            self.value_transform = eval(self.value_transform)

        self.acquisition_function = acquisition_function
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.approx_grad = approx_grad

        self.kappa = acq_fct_kwargs.pop('kappa', kwargs.get('kappa', 0.0))
        self.acq_fct_kwargs = acq_fct_kwargs

        self.gp_kwargs = gp_kwargs

        self.random = check_random_state(random_state)

        self.model = \
            GaussianProcessModel(random_state=self.random, **self.gp_kwargs)

    def init(self, dimension):
        self.dimension = dimension

        if len(self.boundaries) == 1:
            self.boundaries = np.array(self.boundaries * self.dimension)
        elif len(self.boundaries) == self.dimension:
            self.boundaries = np.array(self.boundaries)
        else:
            raise Exception("Boundaries not specified for all dimensions")

        self.parameters = []
        self.returns = []

        self.opt_value = -np.inf  # maximal value obtained so far
        self.opt_params = None  # best parameters found so far

    def get_next_parameters(self, params, explore=True):
        """Return parameter vector that shall be evaluated next.

        Parameters
        ----------

        params : array-like
            The selected parameters will be written into this as a side-effect.
        explore : bool
            Whether exploration in parameter selection is enabled
        """
        params_ = (self.random.uniform(size=self.dimension) *
                   (self.boundaries[:, 1] - self.boundaries[:, 0]) +
                   self.boundaries[:, 0])
        # Find query point where the acquisition function becomes maximal
        if len(self.parameters) > self.dimension:
            params_ = self._determine_next_query_point(params_, explore)

        self.parameters.append(params_)
        params[:] = params_

    def set_evaluation_feedback(self, feedbacks):
        """Inform optimizer of outcome of a rollout with current weights."""
        return_ = check_feedback(feedbacks, compute_sum=True)
        # Transform reward (e.g. to a log-scale)
        return_ = self.value_transform(return_)
        if return_ > self.opt_value:
            self.opt_value = return_
            self.opt_params = self.parameters[-1]

        self.returns.append(return_)

    def get_best_parameters(self):
        return self.opt_params

    def is_behavior_learning_done(self):
        # TODO
        return False

    def _determine_next_query_point(self, start_point, explore):
        # Train model approximating the return landscape
        self.model.train(np.vstack(self.parameters), self.returns)

        # Create acquisition function
        kappa = self.kappa if explore else 0.0
        acquisition_function = ACQUISITION_FUNCTIONS[self.acquisition_function](
            self.model.gp, kappa)

        def target_function(x, compute_gradient=False):
            # Check boundaries
            if not np.all(np.logical_and(x >= self.boundaries[:, 0],
                                         x <= self.boundaries[:, 1])):
                return -np.inf

            return acquisition_function(x, baseline_value=self.opt_value,
                                        compute_gradient=compute_gradient)

        # Perform optimization
        if self.acquisition_function != "random" or not explore:
            opt = optimize(target_function, boundaries=self.boundaries,
                           optimizer=self.optimizer,
                           maxf=self.optimizer_kwargs.get("maxf", 100),
                           approx_grad=self.approx_grad,
                           random=self.random)
        else:  # the start point is already randomly chosen, so we keep it
            opt = start_point

        # Check if we have tried a very similar parameter vector before
        if (cdist(self.parameters, [opt]).min() / 1e-8 <
                np.linalg.norm(self.boundaries[:, 1] - self.boundaries[:, 0])):
            # Choose a random parameter vector
            opt = self.random.uniform(size=self.dimension) \
                * (self.boundaries[:, 1] - self.boundaries[:, 0]) \
                + self.boundaries[:, 0]

        # Clip to hard boundaries
        return np.maximum(self.boundaries[:, 0],
                          np.minimum(opt, self.boundaries[:, 1]))

    def __getstate__(self):
        """Return a pickable state for this object """
        odict = self.__dict__.copy()  # copy the dict since we change it
        if "value_transform" in odict:
            odict.pop("value_transform")
        return odict


def optimize(objective_function, boundaries, optimizer, maxf, x0=None,
             approx_grad=True, random=np.random, *args, **kwargs):
    """Minimize objective_function within given boundaries.

    This function optimizes an objective function in a search space with the
    given boundaries. The optimizer may use up to maxf evaluations of the
    objective function. The optimizer is specified by a string which may be
    any of "direct", "direct+lbfgs", "random", "random+lbfgs", "cmaes", or
    "cmaes+lbfgs".
    """
    if optimizer in ["direct", "direct+lbfgs"]:
        # Use DIRECT to perform approximate global optimization of
        # objective_function
        try:
            import nlopt
        except ImportError:
            raise Exception("'direct' optimizer requires the package nlopt."
                            "You may install it using "
                            "'sudo apt-get install python-nlopt'")

        opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND, boundaries.shape[0])
        opt.set_lower_bounds(boundaries[:, 0])
        opt.set_upper_bounds(boundaries[:, 1])
        opt.set_maxeval(maxf)

        def prox_func(params, grad):
            # Note: nlopt minimizes function, hence the minus
            func_value = -objective_function(params)
            if np.iterable(func_value):
                return func_value[0]
            else:
                return func_value
        opt.set_min_objective(prox_func)
        x0 = opt.optimize(boundaries.mean(1))
    elif optimizer in ["random", "random+lbfgs"]:
        # Sample maxf points uniform randomly from the search space and
        # remember the one with maximal objective value
        if x0 is not None:
            f_opt = objective_function(x0)
        else:
            f_opt = -np.inf
        for _ in range(maxf):
            x0_trial = \
                random.uniform(size=boundaries.shape[0]) \
                * (boundaries[:, 1] - boundaries[:, 0]) \
                + boundaries[:, 0]
            f_trial = objective_function(x0_trial)
            if f_trial > f_opt:
                f_opt = f_trial
                x0 = x0_trial
    elif optimizer in ["cmaes", "cmaes+lbfgs"]:
        # Use CMAES to perform approximate global optimization of
        # objective_function
        if x0 is None:
            x0 = boundaries.mean(1)
        x0 = fmin_cma(lambda x, compute_gradient=False: -objective_function(x),
                      x0=x0, xL=boundaries[:, 0], xU=boundaries[:, 1],
                      sigma0=kwargs.get("sigma0", 0.01), maxfun=maxf)
    elif x0 is None:
        raise Exception("Unknown optimizer %s and x0 is None."
                        % optimizer)

    if optimizer in ["direct", "random", "cmaes"]:
        # return DIRECT/Random/CMAES solution without refinement
        return x0
    elif optimizer in ["lbfgs", "direct+lbfgs", "random+lbfgs", "cmaes+lbfgs"]:
        # refine solution with L-BFGS
        def proxy_function(x):
            if approx_grad:
                return -objective_function(x)
            else:
                f, f_grad = objective_function(x, compute_gradient=True)
                return -f, -f_grad[0]
        res = fmin_l_bfgs_b(proxy_function, x0,
                            approx_grad=approx_grad,
                            bounds=boundaries, disp=0)
        return res[0]
    else:
        raise Exception("Unknown optimizer %s" % optimizer)


def fmin_cma(objective_function, x0, xL, xU, sigma0=0.01, maxfun=1000):
    """ Minimize objective function in hypercube using CMA-ES.

    This function optimizes an objective function in a search space bounded by
    a hypercube. One corner of the hypercube is given by xL and the opposite by
    xU. The initial mean of the search distribution is given by x0. The search
    space is scaled internally to the unit hypercube to accommodate CMA-ES.

    Parameters
    ----------
    objective_function : callable
        The objective function to be minimized. Must return a scalar value

    x0 : array-like
        Initial mean of the search distribution

    xL: array-like
        Lower, left corner of the bounding hypercube

    xU: array-like
        Upper, right corner of the bounding hypercube

    sigma0: float, default=0.01
        Initial variance of search distribution of CMA-ES

    maxfun: int, default=1000
        Maximum number of evaluations of the objective function after which the
        optimization is stopped.

    Returns
    ----------
    x_opt : array-like
        The minimum of objective function identified by CMA-ES
    """
    x0 = np.asarray(x0)
    xL = np.asarray(xL)
    xU = np.asarray(xU)
    # Scale parameters such that search space is a unit-hypercube
    x0 = (x0 - xL) / (xU - xL)
    bounds = np.array([np.zeros_like(x0), np.ones_like(x0)]).T

    # Rescale in objective function
    def scaled_objective_function(x):
        return objective_function(x * (xU - xL) + xL)

    # Minimize objective function using CMA-ES. Restart if no valid solution is
    # found
    res = (None, np.inf)
    while not np.isfinite(res[1]):
        res = fmin(scaled_objective_function, x0=x0, variance=sigma0,
                   bounds=bounds, maxfun=maxfun)
        x_opt = res[0]
    # Return rescaled solution
    return x_opt * (xU - xL) + xL
