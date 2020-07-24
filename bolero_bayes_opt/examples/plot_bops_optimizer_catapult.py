
from functools import partial

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import uniform
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

from bolero.environment.catapult import Catapult
from bolero_bayes_opt import BOPSOptimizer

catapult = Catapult([(0, 0), (2.0, -0.5), (3.0, 0.5), (4, 0.25),
                     (5, 2.5), (7, 0), (10, 0.5), (15, 0)])
catapult.init()


kernel = C(100.0, (1.0, 10000.0)) \
    * Matern(length_scale=(1.0, 1.0), length_scale_bounds=[(0.1, 100), (0.1, 100)])

opt = BOPSOptimizer(
    boundaries=[(5, 10), (0, np.pi/2)], bo_type="bo",
    acquisition_function="UCB", acq_fct_kwargs=dict(kappa=2.5),
    optimizer="direct+lbfgs", maxf=100,
    gp_kwargs=dict(kernel=kernel, normalize_y=True, alpha=1e-5))


target = 4.0  # Fixed target
context = (target - 2.0) / 8.0
# Compute maximal achievable reward for this target
optimal_reward = -np.inf
for _ in range(10):
    x0 = [uniform.rvs(5.0, 5.0), uniform.rvs(0.0, np.pi/2)]
    result = fmin_l_bfgs_b(
        lambda x, context=context: -catapult._compute_reward(x, context=context),
        x0, approx_grad=True, bounds=[(5.0, 10.0), (0.0, np.pi/2)])
    if -result[1] > optimal_reward:
        optimal_reward = -result[1]
        optimal_params = result[0]
        theta_opt = result[0][1]

# Determine reward landscape
v = np.linspace(5.0, 10.0, 100)
theta = np.linspace(0.0, np.pi / 2, 100)
V, Theta = np.meshgrid(v, theta)

Z = np.array([[partial(catapult._compute_reward, context=context)
                   ([V[i, j], Theta[i, j]])
               for j in range(v.shape[0])]
               for i in range(theta.shape[0])])

# Perform actual experiment
n_rollouts = 100
opt.init(2)
params_ = np.zeros(2)
reward = np.empty(1)
samples = np.empty((n_rollouts, 2))
rewards = np.empty((n_rollouts))
for rollout in range(n_rollouts):
    opt.get_next_parameters(params_)
    samples[rollout] = (params_[0], params_[1])

    reward = catapult._compute_reward(params_, context) - optimal_reward

    rewards[rollout] = reward
    opt.set_evaluation_feedback(reward)

plt.figure(0)
plt.plot(rewards)
plt.title("Learning curve")

plt.figure(1)
plt.scatter(samples[:, 0], samples[:, 1], c='k')
plt.scatter(optimal_params[[0]], optimal_params[[1]], c='c')
plt.contourf(V, Theta, Z, 25, zorder=-1)
plt.colorbar()
plt.xlabel("velocity")
plt.ylabel("theta")
plt.title("Cost for target=%.3f" % target)
plt.show()
