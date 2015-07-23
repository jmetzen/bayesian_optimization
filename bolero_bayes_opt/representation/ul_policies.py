import numpy as np

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge

from bolero.representation.ul_policies \
    import UpperLevelPolicy, BoundedScalingPolicy
from bolero.utils.validation import check_random_state

from bayesian_optimization import GaussianProcessModel


class KernelRegressionPolicy(UpperLevelPolicy):
    """Linear policy in approximated kernel space.

    A linear policy in kernel space is learned. In order to keep computation
    and risk of overfitting limited, a low-dimensional approximation of the
    kernel space is used, which is determined by the Nystroem approximation.
    Thus, an explicit feature map is learned based on the training data. This
    has the advantage compared to predefined feature maps that the features
    are adaptive.

    Parameters
    ----------
    weight_dims: int
        dimensionality of weight vector of lower-level policy

    context_dims: int
        dimensionality of context vector

    kernel : string or callable (default: "rbf")
        Kernel map to be approximated. A callable should accept two
        arguments and the keyword arguments passed to this object as
        kernel_params, and should return a floating point number.

    gamma : float (default: None)
        Gamma parameter for the RBF, polynomial, exponential chi2 and sigmoid
        kernels. Interpretation of the default value is left to the kernel;
        see the documentation for sklearn.metrics.pairwise. Ignored by
        other kernels.

    coef0 : float (default: 1.5)
        The coef0 parameter for the kernels. Interpretation of the value
        is left to the kernel; see the documentation for
        sklearn.metrics.pairwise. Ignored by other kernels.

    n_components: int (default: 20)
        The number of components used in the Nystroem approximation of the
        kernel

    covariance_scale: float (default: 1.0)
        the covariance is initialized to numpy.eye(weight_dims) *
        covariance_scale.

    alpha: float (default: 0.0)
        Controlling the L2-regularization in the ridge regression for
        learning of the policy's weights

    bias: bool (default: True)
        Whether a constant bias dimension is added to the approximated kernel
        space. This allows learning offsets more easily.

    normalize: bool (default: True)
        Whether the activations in the approximated kernel space are normalized.
        This should improve generalization beyond the boundaries of the
        observed context space.

    random_state : optional, int
        Seed for the random number generator.

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2014/11/20
    """

    def __init__(self, weight_dims, context_dims, kernel="rbf", gamma=None,
                 coef0=1.5, n_components=20, covariance_scale=1.0, alpha=0.0,
                 bias=True, normalize=True, random_state=None):
        self.weight_dims = weight_dims
        self.context_dims = context_dims
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.n_components = n_components
        self.alpha = alpha
        self.bias = bias
        self.normalize = normalize

        self.Sigma = np.eye(weight_dims) * covariance_scale

        self.random_state = check_random_state(random_state)

    def __call__(self, contexts, explore=True):
        """Evaluates policy for given contexts.

        Samples weight vector from distribution if explore is true, otherwise
        return the distribution's mean (which depends on the context).

        Parameters
        ----------
        contexts: array-like, [n_contexts, context_dims]
            context vector

        explore: bool
            if true, weight vector is sampled from distribution. otherwise the
            distribution's mean is returned
        """
        X = self.nystroem.transform(contexts)
        if self.bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        if self.normalize:
            X /= np.abs(X).sum(1)[:, None]

        means = np.dot(X, self.W.T)
        if not explore:
            return means
        else:
            sample_func = lambda x: self.random_state.multivariate_normal(
                x, self.Sigma, size=[1])[0]
            samples = np.apply_along_axis(sample_func, 1, means)
            return samples

    def fit(self, X, Y, weights=None, context_transform=True):
        """ Trains policy by weighted maximum likelihood.

        .. note:: This call changes this policy (self)

        Parameters
        ----------
        X: array-like, shape (n_samples, context_dims)
            Context vectors

        Y: array-like, shape (n_samples, weight_dims)
            Low-level policy parameter vectors

        weights: array-like, shape (n_samples,)
            Weights of individual samples (should depend on the obtained
            reward)
        """
        # Kernel approximation
        self.nystroem = Nystroem(kernel=self.kernel, gamma=self.gamma,
                                 coef0=self.coef0,
                                 n_components=np.minimum(X.shape[0],
                                                         self.n_components),
                                 random_state=self.random_state)
        self.X = self.nystroem.fit_transform(X)
        if self.bias:
            self.X = np.hstack((self.X, np.ones((self.X.shape[0], 1))))
        if self.normalize:
            self.X /= np.abs(self.X).sum(1)[:, None]

        # Standard ridge regression
        ridge = Ridge(alpha=self.alpha, fit_intercept=False)
        ridge.fit(self.X, Y, weights)
        self.W = ridge.coef_

        # TODO: self.Sigma needs to be adapted


def model_free_policy_training(policy, contexts, parameters, returns=None,
                               epsilon=1.0, min_eta=1e-6):
    """Train a contextual policy in a model-free way using REPS.

    Use the given (context, parameter, return) triples for training a
    contextual policy using REPS. The contexts are passed through a
    context_transformation function.

    Parameters
    ----------
    policy: UpperLevelPolicy
        A contextual upper level policy that shall be updated based on samples.

    contexts: array-like, [n_samples, context_dims]
        The contexts used for training

    parameters: array-like, [n_samples, param_dims]
        The parameters of the low-level policy used in the respective context.

    returns: array-like, [n_samples]
        The returns obtained by the low-level policy in the respective context.
        If None, then all training examples are used with equal weight;
        otherwise the returns are used to determine the weights of the samples
        by using REPS.

    epsilon: float > 0.0
        The maximum the KL divergence between training and policy distribution
        might take on. Defaults to 1.0

    min_eta: float > 0.0
        eta is the Lagrangian multiplier connected to the KL-bound constraint.
        If it becomes too small, numerical instability might occur. Because of
        this, min_eta can be set to a value larger than zero which acts as a
        lower bound for eta. Defaults to 1e-6.

    Returns
    -------
    policy : LinearGaussianPolicy
        Upper level policy that maps contexts to parameter vectors
    """
    parameters = np.asarray(parameters)
    features = np.array([policy.transform_context(c) for c in contexts])
    if returns is not None:
        returns = np.asarray(returns)
        from optimizer.python.utils.reps import solve_dual_contextual_reps
        d, _ = solve_dual_contextual_reps(features, returns, epsilon=epsilon,
                                          min_eta=min_eta)
    else:
        d = np.ones(contexts.shape[0]) / contexts.shape[0]

    policy.fit(contexts, parameters, d)

    return policy


def model_based_policy_training(policy, contexts, parameters, returns,
                                boundaries=None, policy_initialized=False,
                                model_conf={}, maxfun=5000,
                                variance=0.01, *args, **kwargs):
    """Train a contextual policy in a model-based way on GP model.

    Use the given (context, parameter, return) triples for training a
    model (based on Gaussian processes) of the function
    context x parameter > return. Optimize the parameters of a parametric
    policy such that it would maximize the accrued return in the model.

    Parameters
    ----------
    policy: UpperLevelPolicy
        A contextual upper level policy that shall be updated based on samples.

    contexts: array-like, [n_samples, context_dims]
        The contexts used for training

    parameters: array-like, [n_samples, param_dims]
        The parameters of the low-level policy used in the respective context.

    returns: array-like, [n_samples]
        The returns obtained by the low-level policy in the respective context.
        If None, then all training examples are used with equal weight;
        otherwise the returns are used to determine the weights of the samples
        by using REPS.

    boundaries : array-like or None, shape=[param_dims, 2] (default: None)
        The boundaries of the action space.

    policy_initialized : bool (default: False)
        Whether the policy has already been initialized reasonably or needs to
        be initialized first before performing model-based optimization

    model_conf: dict (default : {})
        Passed as keyword arguments to the GP model.

    maxfun: int, optional (default: 5000)
        The maximal number of policy parameters after which the optimizer
        terminates

    variance: float, optional (default: 0.01)
        The initial exploration variance of CMA-ES

    Returns
    -------
    policy : LinearGaussianPolicy
        Upper level policy that maps contexts to parameter vectors
    """
    contexts = np.asarray(contexts)
    parameters = np.asarray(parameters)

    # Train model approximating the context x parameter -> return landscape
    inputs = np.hstack([contexts, parameters])
    model = GaussianProcessModel(**model_conf)
    model.fit(inputs, returns)

    if not policy_initialized:
        # Initialize policy
        if boundaries is not None:
            # Initialize policy such that outputs are maximally far from
            # boundary
            policy.fit(contexts, [boundaries.mean(1)]*contexts.shape[0],
                       weights=np.ones(contexts.shape[0]))
        else:
            # Let policy return mean parameter vector
            policy.fit(contexts, parameters.mean(),
                       weights=np.ones(contexts.shape[0]))

    return model_based_policy_training_pretrained(
        policy, model.gp, contexts, boundaries, maxfun, variance,
        *args, **kwargs)


def model_based_policy_training_pretrained(
     policy, model, contexts, boundaries=None, maxfun=5000, variance=0.01,
     *args, **kwargs):
    """Train a contextual policy in a model-based way on GP model.

    Optimize the parameters of a parametric policy such that it would maximize
    the accrued return in a pretrained model for the given contexts.

    Parameters
    ----------
    policy: UpperLevelPolicy
        A contextual upper level policy that shall be updated based on samples.

    model: GPR-instance
        The GPR-model in which the policy is optimized

    contexts: array-like, [n_samples, context_dims]
        The contexts used for training

    boundaries : array-like or None, shape=[param_dims, 2] (default: None)
        The boundaries of the action space.

    maxfun: int, optional (default: 5000)
        The maximal number of policy parameters after which the optimizer
        terminates

    variance: float, optional (default: 0.01)
        The initial exploration variance of CMA-ES

    Returns
    -------
    policy : LinearGaussianPolicy
        Upper level policy that maps contexts to parameter vectors
    """

    def average_return(policy_params):
        """ Return predicted return of policy for given policy parameters. """
        # Instantiate policy
        policy.W = np.array(policy_params.reshape(policy.W.shape))
        policy.W *= scale_factor
        # Determine parameters selected by policy for given contexts
        params = policy(contexts, explore=False)
        if boundaries is not None:  # check boundaries
            params = np.minimum(np.maximum(params, boundaries[:, 0]),
                                boundaries[:, 1])
        # Compute mean output of GP model for contexts and selected params
        values = model.predict(np.hstack((contexts, params)))
        # Return mean over return obtained on all training contexts
        return values.mean()

    if boundaries is not None:
        # Compute scale factor per dimension such that isotropic exploration
        # is meaningful
        if isinstance(policy, BoundedScalingPolicy):
            scale_factor = \
                (policy.scaling.inv_scale(boundaries[:, 1])
                 - policy.scaling.inv_scale(boundaries[:, 0]))[:, None]
        else:
            scale_factor = (boundaries[:, 1] - boundaries[:, 0])[:, None]
    else:
        scale_factor = 1  # Don't do scaling since the scales are unknown

    # Refine policy determined in model-free way by performing L-BFGS on
    # the model.
    from bolero.optimizer.cmaes import fmin as fmin_cmaes
    policy.W /= scale_factor
    x_lbfgs, _ = \
        fmin_cmaes(average_return, x0=policy.W.flatten(), maxfun=maxfun,
                   eval_initial_x=True, variance=variance, maximize=True,
                   *args, **kwargs)

    # Set weights of linear policy
    policy.W = x_lbfgs.reshape(policy.W.shape)
    policy.W *= scale_factor

    return policy