import numpy as np
import pandas as pd
import scipy


def generate_beta(n, tau0, tau1, effect_prob=0.2, rng=None):
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
    effect_prob : float, optional
        Probability that an effect is present. Default is 0.2
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a new generator is created.

    Returns
    -------
    beta : array
        Coefficients for regression model
    """
    if rng is None:
        rng = np.random.default_rng()

    z = rng.binomial(1, effect_prob, size=n)
    p = rng.exponential(scale=tau1, size=n)
    delta = rng.normal(loc=0, scale=tau0, size=n)
    beta = z*p + delta
    return beta

def generate_design_matrix(N, n, rho, covariate_dist = "bernoulli", rng=None):
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
    covariate_dist: string, optional
        Distribution of covariates, either 'normal' or 'bernoulli'. 
        Default is 'bernoulli'.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. 
        If None, a new generator is created.
    """
    if rng is None:
        rng = np.random.default_rng()

    if covariate_dist not in ["bernoulli", "normal"]:
        ValueError("covariate_dist should either be bernoulli or normal")

    Sigma = rho*np.ones((n,n)) + (1-rho)*np.identity(n)
    X = rng.multivariate_normal(np.zeros(n), Sigma, size=N)
    if(covariate_dist == "bernoulli"):
        u = rng.uniform(-0.25, 0.25, size=n)
        for i in range(n):
            X[:,i] = X[:,i] > u[i]

    return X

def generate_response(X, beta, sigma, expected_prop = 0.5, rng=None):
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
    expected_prop : float, optional
        The expected proportion of responses equal to 1. Default is 0.5.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. 
        If None, a new generator is created.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N = np.shape(X)[0]
    e = rng.normal(0, scale=sigma, size=N)
    linpred = X@beta + e

    if expected_prop == 0.5:
        alpha = np.mean(linpred)
    else:
        def exp_prop(alpha):
            return np.sum(1/(1+np.exp(-alpha-linpred))) - N*expected_prop
        alpha = scipy.optimize.root_scalar(exp_prop, x0 = np.mean(linpred), 
                                           method='newton').root
    probs = 1/(1+np.exp(-alpha - linpred))
    y = rng.binomial(1, probs, size=N)
    return y

def generate_data(N, n, tau0, tau1, rho, sigma, effect_prob = 0.2,
                  covariate_dist="bernoulli", expected_prop = 0.5, rng=None):
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
    effect_prob : float
        Probability of a regression predictor representing a real effect
    covariate_dist: string, optional
        Distribution of covariates, either 'normal' or 'bernoulli'. 
        Default is 'bernoulli'.
    expected_prop : float, optional
        The expected proportion of responses equal to 1. Default is 0.5.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a new generator is created.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    beta = generate_beta(n, tau0, tau1, effect_prob=effect_prob, rng=rng)
    X = generate_design_matrix(N, n, rho, 
                            covariate_dist=covariate_dist, rng=rng)
    y = generate_response(X, beta, sigma, expected_prop=expected_prop, rng=rng)
    return y, X, beta
