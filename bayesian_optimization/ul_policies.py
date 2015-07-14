import numpy as np
from bolero.representation.ul_policies import UpperLevelPolicy
from bolero.utils.validation import check_random_state
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge


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
