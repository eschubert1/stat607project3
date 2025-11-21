import pytest
import numpy as np

from ..src.dgps import generate_data
from ..src.methods import compute_estimates, empirical_bayes, logistic_irls
from ..src.metrics import compute_metrics

def test_estimation_size():
    """
        Test that estimates have appropriate dimensions
    """
    y, X, beta = generate_data(40, 10, 0.2, 0.2, 0.5, 1)
    ests, vars, tau = compute_estimates(y, X, 0.25)

    n = np.shape(X)[1]
    assert np.shape(ests) == (6, n)
    assert np.shape(vars) == (6, n)

def test_estimation_range():
    """
        Test that estimates are in expected ranges
    """
    y, X, beta = generate_data(40, 10, 0.2, 0.2, 0.5, 1)
    ests, vars, tau = compute_estimates(y, X, 0.25)

    for arr in vars:
        for element in arr:
            if element is not None:
                if element < 0:
                    raise ValueError("Variance estimates must be non-negative")
    
    if tau < 0:
        raise ValueError("Estimate of prior variance must be non-negative")
    
def test_empirical_bayes():
    """
        Test that empirical bayes function returns reasonable estimates when
        MLE returns reasonable estimates
    """
    estimates = []
    for i in range(1000):
        y, X, beta = generate_data(40, 10, 0.2, 0.2, 0.5, 1)
        n = np.shape(X)[1]
        ml_beta, ml_cov = logistic_irls(y,X)
        if np.array_equal(ml_cov, np.zeros((n,n))):
            continue
        if np.any(np.diag(ml_cov) > 100):
            continue
        eb_beta, eb_cov, tau2 = empirical_bayes(y, X, ml_beta, ml_cov)
        estimates.append(eb_beta)
    
    estimates = np.array(estimates)
    if estimates.max() > 100:
        raise ValueError("Empirical Bayes estimates diverging")
    
def test_rng_consistency():
    y, X, beta = generate_data(100, 4, 0.2, 0.2, 0.5, 1, rng=np.random.default_rng(32))
    ynew, Xnew, betanew = generate_data(100, 4, 0.2, 0.2, 0.5, 1, rng=np.random.default_rng(32))
    assert np.array_equal(y, ynew)
    assert np.array_equal(X, Xnew)
    assert np.array_equal(beta, betanew)