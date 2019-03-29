# Author: Jan Hendrik Metzen <janmetzen@mailbox.org>

from copy import deepcopy

import numpy as np

from bolero.optimizer import ContextualOptimizer
from bolero.representation.ul_policies import BoundedScalingPolicy
from bolero.utils.validation import check_feedback, check_random_state

from bayesian_optimization import (REMBOOptimizer, InterleavedREMBOOptimizer,
    BayesianOptimizer, GaussianProcessModel, create_acquisition_function)

from ..representation.ul_policies \
    import model_free_policy_training, model_based_policy_training


class BOCPSOptimizer(ContextualOptimizer):
    """Bayesian Optimization for Contextual Policy Search (BO-CPS).

    Parameters
    ----------
    boundaries : list of pair of floats
        The boundaries of the parameter space in which the optimum is searched.

    bo_type: str, default: "bo"
        The type of Bayesian optimization performed. Can be "bo" for standard
        Bayesian optimization, "rembo" for REMBO, or "interleaved_rembo" for
        several interleaved runs of REMBO. Note that REMBO-based optimizers
        only embed the parameter dimensions and not the context dimensions

    acquisition_function : string, optional (default: 'ucb')
        String identifying the acquisition function to be used. Supported are
        * "UCB": Upper-Confidence Bound (default)
        * "PI": Probability of Improvement
        * "EI": Expected Improvement

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
    def __init__(self, boundaries, bo_type="bo", acquisition_function="UCB",
                 policy=None, optimizer="direct+lbfgs",
                 acq_fct_kwargs={}, gp_kwargs={},
                 value_transform=lambda x: x, random_state=None,
                 *args, **kwargs):
        if not isinstance(boundaries, list):
            raise ValueError("Boundaries must be passed as a list of tuples "
                             "(pairs).")

        self.boundaries = boundaries
        self.bo_type = bo_type
        self.value_transform = value_transform
        if isinstance(self.value_transform, str):
            self.value_transform = eval(self.value_transform)
        self.optimizer = optimizer

        self.acquisition_function = acquisition_function
        self.acq_fct_kwargs = acq_fct_kwargs
        self.gp_kwargs = gp_kwargs

        self.policy = policy
        self.rng = check_random_state(random_state)

        self.kwargs = kwargs

    def init(self, n_params, n_context_dims):
        self.dimension = n_params
        self.context_dims = n_context_dims

        if self.policy is not None:  # Bound policy to allowed parameter space
            self.policy = \
                BoundedScalingPolicy(self.policy, scaling="none",
                                     bounds=np.array(self.boundaries))
            self.policy_fitted = False

        # Create surrogate model, acquisition function and Bayesian optimizer
        self.model = \
            GaussianProcessModel(random_state=self.rng, **self.gp_kwargs)
        self.acquisition_function = \
            self._create_acquisition_function(
                self.acquisition_function, self.model, **self.acq_fct_kwargs)

        if len(self.boundaries) == 1:
            self.boundaries = np.array(self.boundaries * self.dimension)
        elif len(self.boundaries) == self.dimension:
            self.boundaries = np.array(self.boundaries)
        else:
            raise Exception("Boundaries not specified for all dimensions")

        if self.bo_type == "bo":
            BoClass = BayesianOptimizer
        elif self.bo_type == "rembo":
            BoClass = REMBOOptimizer
        elif self.bo_type == "interleaved_rembo":
            BoClass = InterleavedREMBOOptimizer
        else:
            raise Exception("Unknown bo_type %s" % self.bo_type)
        self.bayes_opt = BoClass(
            model=self.model, acquisition_function=self.acquisition_function,
            optimizer=self.optimizer, n_dims=self.dimension + self.context_dims,
            data_space=self.boundaries, n_keep_dims=self.context_dims,
            maxf=self.kwargs.pop("maxf", 100), random_state=self.rng,
            **self.kwargs)

    def set_context(self, context):
        """ Set context of next evaluation"""
        super(BOCPSOptimizer, self).set_context(context)
        self.context = context

    def get_next_parameters(self, params, explore=True):
        """Return parameter vector that shall be evaluated next.

        Parameters
        ----------
        params : array-like
            The selected parameters will be written into this as a side-effect.

        explore : bool
            Whether exploration in parameter selection is enabled
        """
        self.parameters = \
            self._determine_next_query_point(self.context, self.bayes_opt)
        params[:] = self.parameters

    def set_evaluation_feedback(self, feedbacks):
        """Inform optimizer of outcome of a rollout with current weights."""
        return_ = check_feedback(feedbacks, compute_sum=True)
        # Transform reward (e.g., to a log-scale)
        return_ = self.value_transform(return_)

        self.bayes_opt.update(np.hstack((self.context, self.parameters)),
                              return_)

        if self.policy is not None:
            # Policy derived from internal model is no longer valid as the data
            # has changed
            self.policy_fitted = False

    def best_policy(self, maxfun=15000, variance=0.01,
                    training=["model-free", "model_based"]):
        """Returns the best (greedy) policy learned so far.

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
            X = np.asarray(self.bayes_opt.X_)
            contexts = X[:, :self.context_dims]
            parameters = X[:, self.context_dims:]
            returns = np.asarray(self.bayes_opt.y_)
            # Perform training
            if "model-free" in training:
                self.policy = model_free_policy_training(
                    self.policy, contexts, parameters, returns,
                    epsilon=1.0, min_eta=1e-6)
            if "model-based" in training:
                self.policy = model_based_policy_training(
                    self.policy, contexts, parameters, returns,
                    boundaries=self.boundaries,
                    policy_initialized="model-free" in training,
                    maxfun=maxfun, variance=variance,
                    model_conf=self.gp_kwargs)
            self.policy_fitted = True
            return self.policy
        else:
            # TODO return UpperLevelPolicy object
            greedy_optimizer = deepcopy(self.bayes_opt)
            greedy_optimizer.acquisition_function = \
                create_acquisition_function("GREEDY", self.model)
            def non_parametric_policy(c, explore):
                return self._determine_next_query_point(c, greedy_optimizer)
            return non_parametric_policy

    def is_behavior_learning_done(self):
        # TODO
        return False

    def get_desired_context(self):
        """Chooses desired context for next evaluation.

        Returns
        -------
        context : ndarray-like, default=None
            The context in which the next rollout shall be performed. If None,
            the environment may select the next context without any preferences.
        """
        return None

    def _determine_next_query_point(self, context, optimizer):
        # Prepend fixed context to search space
        cx_boundaries = np.empty((self.context_dims + self.dimension, 2))
        cx_boundaries[:self.context_dims] = context[:, np.newaxis]
        cx_boundaries[self.context_dims:] = self.boundaries

        # Determine optimal parameters for fixed context
        cx = optimizer.select_query_point(cx_boundaries)
        return cx[self.context_dims:]

    def _create_acquisition_function(self, name, model, **kwargs):
        if not name in ["UCB", "EntropySearch", "MinimalRegretSearch",
                        "GREEDY", "RANDOM"]:
            raise ValueError("%s acquisition function not supported."
                             % name)

        return create_acquisition_function(name, model, **kwargs)

    def __getstate__(self):
        """ Return a pickable state for this object """
        odict = self.__dict__.copy()  # copy the dict since we change it
        if "value_transform" in odict:
            odict.pop("value_transform")
        return odict
