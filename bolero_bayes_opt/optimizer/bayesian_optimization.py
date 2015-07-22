# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>

import numpy as np
from scipy.spatial.distance import cdist

from bolero.optimizer import Optimizer
from bolero.utils.validation import check_random_state, check_feedback

from ..model import GaussianProcessModel


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

