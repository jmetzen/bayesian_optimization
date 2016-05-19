# Author: Jan Hendrik Metzen <janmetzen@mailbox.org>

import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def global_optimization(objective_function, boundaries, optimizer, maxf,
                        x0=None, approx_grad=True, random=np.random,
                        *args, **kwargs):
    """Maximize objective_function within given boundaries.

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
        nlopt.srand(0)
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
            return -objective_function(x)
        res = fmin_l_bfgs_b(proxy_function, x0,
                            approx_grad=True,
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
    try:
        from bolero.optimizer import fmin
    except ImportError:
        raise Exception("'cmaes' optimizer requires the package bolero.")

    x0 = np.asarray(x0)
    xL = np.asarray(xL)
    xU = np.asarray(xU)
    # Scale parameters such that search space is a unit-hypercube
    x0 = (x0 - xL) / (xU - xL)
    x0[~np.isfinite(x0)] = 0  # Deal with situations where xU == xL
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
