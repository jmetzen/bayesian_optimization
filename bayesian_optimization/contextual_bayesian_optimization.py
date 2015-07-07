# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>

import numpy as np
from bolero.optimizer import ContextualOptimizer
from bolero.representation.ul_policies import BoundedScalingPolicy
from bolero.utils.validation import check_feedback, check_random_state
from .bayesian_optimization import optimize
from .acquisition_functions import (CONTEXTUAL_ACQUISITION_FUNCTIONS,
                                    ZeroBaseline, MaxLCBBaseline)
from .model import GaussianProcessModel
from .model import model_free_policy_training, model_based_policy_training


class ContextualBayesianOptimizer(ContextualOptimizer):
    """Contextual Bayesian optimization.

    Parameters
    ----------
    boundaries : list of pair of floats
        The boundaries of the parameter space in which the optimum is searched.

    acquisition_function : string, optional (default: 'ucb')
        String identifying the acquisition function to be used. Supported are
        * "ucb": Upper-Confidence Bound (default)
        * "pi": Probability of Improvement
        * "ei": Expected Improvement
        * "eisp": Expected Improvement in Skill Performance
        * "random": Randomly choose point to sample

    policy : UpperLevelPolicy (default: None)
        If not None, the given upper-level-policy object is optimized
        such that the average reward in the internal GP-model is maximized.
        Otherwise, an on-demand policy is returned, which runs a separate
        optimization for every query point.

    optimizer: string, optional (default: 'direct+lbfgs')
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

    optimizer_kwargs: dict, optional (default: {})
        Optional keyword arguments passed to optimize function. Currently
        supported is maxf (an integer, default=100), which determines the
        maximum number of function evaluations

    acq_fct_kwargs: dict, optional (default: {})
        Optional keyword arguments passed to acquisition function. Currently
        supported is kappa (a float >= 0.0, default=0.0), which handles
        the exploration-exploitation trade-off in the acquisition function.
        Furthermore, additional keyword arguments can be passed to the
        contextual_ei acqusition function

    gp_kwargs: dict, optional (default: {})
        Optional configuration parameters passed to Gaussian process model.
        See documentation of GaussianProcessModel for details.

    value_transform : function, optional (default: lambda x: x)
        Function mapping actual values to values internally by GP modelled.
        For instance, in some situations, a log-transform might be useful.
        Should be a monotonic increasing function. Defaults to identity.

    random_state : RandomState or int (default: None)
        Seed for the random number generator.
    """
    def __init__(self, boundaries, acquisition_function="ucb",
                 policy=None, optimizer="direct+lbfgs", optimizer_kwargs={},
                 acq_fct_kwargs={}, gp_kwargs={},
                 value_transform=lambda x: x,
                 approx_grad=True, random_state=None,
                 *args, **kwargs):
        assert acquisition_function in CONTEXTUAL_ACQUISITION_FUNCTIONS
        assert isinstance(boundaries, list), \
            "Boundaries must be passed as a list of tuples (pairs)."

        self.boundaries = boundaries
        self.value_transform = value_transform
        if isinstance(self.value_transform, basestring):
            self.value_transform = eval(self.value_transform)

        self.acquisition_function = acquisition_function

        self.policy = policy
        if self.policy is not None:  # Bound policy to allowed parameter space
            self.policy = \
                BoundedScalingPolicy(self.policy,
                                     scaling="auto",
                                     bounds=np.array(self.boundaries))
            self.policy_fitted = False

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.approx_grad = approx_grad

        self.kappa = acq_fct_kwargs.pop('kappa', kwargs.get('kappa', 0.0))
        self.acq_fct_kwargs = acq_fct_kwargs

        self.gp_kwargs = gp_kwargs

        self.random = check_random_state(random_state)

        self.model = \
            GaussianProcessModel(random_state=self.random, **self.gp_kwargs)

    def init(self, n_params, n_context_dims):
        self.dimension = n_params
        self.context_dims = n_context_dims

        if len(self.boundaries) == 1:
            self.boundaries = np.array(self.boundaries * self.dimension)
        elif len(self.boundaries) == self.dimension:
            self.boundaries = np.array(self.boundaries)
        else:
            raise Exception("Boundaries not specified for all dimensions")

        self.contexts = []
        self.parameters = []
        self.returns = []

    def set_context(self, context):
        """ Set context of next evaluation"""
        super(ContextualBayesianOptimizer, self).set_context(context)
        self.contexts.append(context)

    def get_next_parameters(self, params, explore=True):
        """Return parameter vector that shall be evaluated next.

        Parameters
        ----------
        params : array-like
            The selected parameters will be written into this as a side-effect.

        explore : bool
            Whether exploration in parameter selection is enabled
        """
        params_ = \
            self.random.uniform(size=self.dimension) \
            * (self.boundaries[:, 1] - self.boundaries[:, 0]) \
            + self.boundaries[:, 0]
        # Find query point where the acquisition function becomes maximal
        if len(self.parameters) > self.dimension + self.context_dims:
            params_ = \
                self._determine_next_query_point(self.contexts[-1], params_,
                                                 explore)

        assert np.issubdtype(params_.dtype, float)

        self.parameters.append(params_)
        params[:] = params_

    def set_evaluation_feedback(self, feedbacks):
        """Inform optimizer of outcome of a rollout with current weights."""
        return_ = check_feedback(feedbacks, compute_sum=True)
        # Transform reward (e.g., to a log-scale)
        return_ = self.value_transform(return_)
        self.returns.append(return_)

        if self.policy is not None:
            # Policy derived from internal model is no longer valid as the data
            # has changed
            self.policy_fitted = False

    def best_policy(self, maxfun=50000, variance=0.01,
                    training=["model-free", "model-based"]):
        """ Returns the best (greedy) policy learned so far.

        Parameters
        ----------
        maxfun : int (default: 50000)
            How many function evaluations are used for model-based policy
            training. Only relevant if policy is not None.

        variance: float, optional (default: 0.01)
            The initial exploration variance of CMA-ES in the model-based
            policy training. Only relevant if policy is not None

        training : list (default: ["model-free", "model-based"])
            How the policy is trained from data. If "model-free" is in the list
            a CREPS-based training is performed. If "model-based" is in the
            list, a model-based training is performed in the model by
            simulating rollouts. If both are in the list, first model-free
            training and then model-based fine-tuning is performed
        """
        if self.policy is not None and training != []:
            if self.policy_fitted:  # return already learned policy
                return self.policy
            assert "model-free" in training or "model-based" in training, \
                "training must contain either 'model-free' or 'model-based'"
            # Perform training
            if "model-free" in training:
                self.policy = model_free_policy_training(
                    self.policy, np.array(self.contexts[:len(self.parameters)]),
                    self.parameters, self.returns, epsilon=1.0, min_eta=1e-6)
            if "model-based" in training:
                self.policy = model_based_policy_training(
                    self.policy, self.contexts[:len(self.parameters)],
                    self.parameters, self.returns, boundaries=self.boundaries,
                    policy_initialized="model-free" in training,
                    maxfun=maxfun, variance=variance,
                    model_conf=self.gp_kwargs)
            self.policy_fitted = True
            return self.policy
        else:
            # TODO return UpperLevelPolicy object
            def non_parametric_policy(c, explore):
                return self._determine_next_query_point(c, None, explore)
            return non_parametric_policy

    def is_behavior_learning_done(self):
        # TODO
        return False

    def _determine_next_query_point(self, context, start_point, explore):
        # Concatenate contexts and params
        inputs = np.hstack([np.vstack(self.contexts[:len(self.parameters)]),
                            np.vstack(self.parameters)])
        # Train model approximating the context x parameter -> return
        # landscape
        self.model.train(inputs, self.returns)

        if explore:
            # Callback to invoke optimizer which optimizes a certain
            # function over parameter space
            param_selection_callback = lambda f, x0=None: \
                optimize(f, boundaries=self.boundaries,
                         optimizer="random+lbfgs", x0=x0,
                         maxf=self.optimizer_kwargs.get("maxf", 100),
                         approx_grad=self.approx_grad,
                         random=self.random)

            if self.acquisition_function in ["pi", "ei"]:
                acquisition_function = \
                    CONTEXTUAL_ACQUISITION_FUNCTIONS[self.acquisition_function](
                        self.model.gp, self.kappa)
                cx_acquisition_fct = \
                    MaxLCBBaseline(acquisition_function,
                                   self.model.gp, param_selection_callback,
                                   self.context_dims, kappa=self.kappa)
            elif self.acquisition_function == "contextual_ei":
                cx_acquisition_fct = \
                    CONTEXTUAL_ACQUISITION_FUNCTIONS["contextual_ei"](
                        self.model.gp, param_selection_callback,
                        self.context_dims, kappa=self.kappa,
                        boundaries=self.boundaries, **self.acq_fct_kwargs)
            elif self.acquisition_function == "eisp":
                cx_acquisition_fct = \
                    CONTEXTUAL_ACQUISITION_FUNCTIONS["eisp"](
                        self.model.gp, param_selection_callback,
                        self.context_dims, kappa=self.kappa)
            else:  # UCB and random do not require a baseline-value
                acquisition_function = \
                    CONTEXTUAL_ACQUISITION_FUNCTIONS[self.acquisition_function](
                        self.model.gp, self.kappa)
                cx_acquisition_fct = \
                    ZeroBaseline(acquisition_function)

        def target_function(x, compute_gradient=False):
            # Concatenate contexts and parameter x
            cx = np.hstack([context, x])

            if explore:
                # Evaluate acquisition function
                return cx_acquisition_fct(cx)
            else:
                # Return mean prediction of GP
                return self.model.gp.predict(cx, return_std=False)[0]

        # Perform optimization
        if self.acquisition_function != "random" or not explore:
            opt = optimize(target_function, boundaries=self.boundaries,
                           optimizer=self.optimizer,
                           maxf=self.optimizer_kwargs.get("maxf", 100),
                           approx_grad=self.approx_grad,
                           random=self.random)
        else:  # the start point is already randomly chosen, so we keep it
            opt = start_point

        # Clip to hard boundaries
        return np.maximum(self.boundaries[:, 0],
                          np.minimum(opt, self.boundaries[:, 1]))

    def __getstate__(self):
        """ Return a pickable state for this object """
        odict = self.__dict__.copy()  # copy the dict since we change it
        if "value_transform" in odict:
            odict.pop("value_transform")
        return odict
