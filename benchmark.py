import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from datetime import date
import time
from src.dgps import generate_data
from src.methods import compute_estimates, logistic_irls
from src.methods_old import compute_estimates_old, mle_old

# Set simulation parameters
params = {
    'run_date': date.today(), 
    'n_sim_multiplier' : 8000,
    'n_obs': [40, 100, 500, 2000],
    'n_predict' : 10, 
    'rho': 0.5,
    'sigma': 0.5,
    'tau0': 0.2,
    'tau1': 0.2,
    'covariate_dist' : 'bernoulli',
    'expected_prop' : 0.5
}

# Generate data
sample_sizes = params['n_obs']
n = params['n_predict']
rho = params['rho']
sigma = params['sigma']
tau0 = params['tau0']
tau1 = params['tau1']
covariate_dist = params['covariate_dist']
expected_prop = params['expected_prop']
effect_prob = 0.2

for N in sample_sizes:
    y, X, beta = generate_data(N, n, tau0, tau1, rho, sigma, effect_prob=effect_prob,
                            covariate_dist = covariate_dist, expected_prop=expected_prop)

    true_tau = effect_prob*(1-effect_prob)*tau1**2 + tau0**2
    
    # Compute and time estimates

    # Old approach
    t0_old = time.time()
    estimates, variances, tau2 = compute_estimates_old(y, X, true_tau)
    t1_old = time.time()
    elapsed_old = t1_old - t0_old

    # New approach
    t0_new = time.time()
    estimates, variances, tau2 = compute_estimates(y, X, true_tau)
    t1_new = time.time()
    elapsed_new = t1_new - t0_new

    # Print results
    print(f"Simulation with {N} observations and {n} features:")
    print(f"Time with old code: {elapsed_old}")
    print(f"Time with new code: {elapsed_new}")
    print(f"Speed up: {elapsed_old/elapsed_new} times faster\n")