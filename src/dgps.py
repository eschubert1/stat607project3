import numpy as np
import pandas as pd


def generate_beta(n, tau0, tau1, rng=None):
    """
    Generate regression coefficients beta.

    Parameters
    ----------
    n : int
        Number of regression coefficients to generate.
    tau0 : float
        Standard deviation of the Gaussian noise component.
    tau1 : float
        Scale parameter of the exponential distribution for the signal component.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a new generator is created.
    """
    if rng is None:
        rng = np.random.default_rng()

    z = rng.binomial(1, 0.5, size=n)
    p = rng.exponential(scale=tau1, size=n)
    delta = rng.normal(loc=0, scale=tau0, size=n)
    beta = z*p + delta
    return beta

def generate_design_matrix(N, n, rho, rng=None):
    """
        Generate the design matrix for logistic regression. Rows are sampled
    from an exchangeable multivariate normal distribution with correlation rho,
    and then columns are dichotomized by n-samples from a U(-0.25,0.25)
    distribution.

    Parameters
    ----------
    N : int
        Number of observations
    n : int
        Number of covariates
    rho : float
        Correlation between multivariate normal observations
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a new generator is created.
    """
    if rng is None:
        rng = np.random.default_rng()

    Sigma = rho*np.ones((n,n)) + (1-rho)*np.identity(n)
    X = rng.multivariate_normal(np.zeros(n), Sigma, size=N)
    u = rng.uniform(-0.25, 0.25, size=n)
    for i in range(n):
        X[:,i] = X[:,i] > u[i]

    return X

def generate_response(X, beta, sigma, rng=None):
    """
        Generate the response for logistic regression.

    Parameters
    ----------
    X : (N, n) array of floats
        Design matrix of the linear predictor
    beta : n-dimensional array of floats
        Coefficients for the linear predictor
    sigma : float
        Standard deviation of random normal noise
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a new generator is created.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N = np.shape(X)[0]
    e = rng.normal(0, scale=sigma, size=N)
    linpred = X@beta + e
    alpha = np.mean(linpred)
    probs = 1/(1+np.exp(-alpha - linpred))
    y = rng.binomial(1, probs, size=N)
    return y

def generate_data(N, n, tau0, tau1, rho, sigma, rng=None):
    """
        Generate data for logistic regression simulation

    Parameters
    ----------
    N : int
        Number of observations
    n : int
        Number of covariates
    tau0 : float
        Standard deviation of the Gaussian noise component.
    tau1 : float
        Scale parameter of the exponential distribution for the signal component.
    rho : float
        Correlation between covariates
    sigma : float
        Standard deviation of random Gaussian noise
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a new generator is created.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    beta = generate_beta(n, tau0, tau1, rng=rng)
    X = generate_design_matrix(N, n, rho, rng=rng)
    y = generate_response(X, beta, sigma, rng=rng)
    return y, X, beta
