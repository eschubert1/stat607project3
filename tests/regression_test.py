import pytest
import numpy as np
from datetime import date

from ..src.dgps import generate_data
from ..src.methods import logistic_irls

def test_regression():

    # Set simulation parameters
    params = {
        'run_date': date.today(), 
        'n_sim_multiplier' : 8000,
        'n_obs': 2000,
        'n_predict' : 100, 
        'rho': 0.5,
        'sigma': 0.5,
        'tau0': 0.2,
        'tau1': 0.2,
        'covariate_dist' : 'bernoulli',
        'expected_prop' : 0.5
    }

    # Generate data
    N = params['n_obs']
    n = params['n_predict']
    rho = params['rho']
    sigma = params['sigma']
    tau0 = params['tau0']
    tau1 = params['tau1']
    covariate_dist = params['covariate_dist']
    expected_prop = params['expected_prop']
    effect_prob = 0.2
    
    y, X, beta = generate_data(N, n, tau0, tau1, rho, sigma, effect_prob=effect_prob,
                            covariate_dist = covariate_dist, expected_prop=expected_prop)

    # New approach
    ml_hat_new, ml_cov_new = logistic_irls(y, X)

    d = np.linalg.norm(ml_hat_new-beta)**2
    # Criterion based off of Chardon, Lerasle, Mourtada (2024) equation 11
    a = 20*(n*np.log(N)+2*np.log(1000))/N
    assert d < a