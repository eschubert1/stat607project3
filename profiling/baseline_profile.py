import numpy as np
from datetime import date
import copy
import time
import pandas as pd
import scipy
import pickle
import warnings
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
from great_tables import GT, loc

# Log warnings
logging.basicConfig(
        filename="profiling/baseline_warnings.log",  # Name of the file to save warnings
        level=logging.WARNING,    # Set the logging level to WARNING
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

logging.captureWarnings(True)

config_param = {
    'run_date': date.today(), 
    'n_sim_multiplier' : [8000, 16000],
    'n_obs': [100, 40, 500, 2000],
    'n_predict' : [4, 10, 20], 
    'rho': [0.5, 0, 0.9],
    'sigma': [0.5, 2],
    'tau0': [0.2, 0, 0.4],
    'tau1': [0.2, 0, 0.4],
    'covariate_dist' : ['bernoulli', 'normal'],
    'expected_prop' : [0.5, 0.25]
}

config_kwargs = {
    'tol' : 1e-8,
    'max_iter' : 1000,
    'alpha' : 0.10,
    'effect_prob' : 0.2
}

def configure_parameters(config_param, config_kwargs):
    """
        Return a list of dictionary tuples with parameter and
        keyword configurations for each scenario.
    """
    # Unpack parameters
    (run_date, n_sim_multiplier, n_obs, n_predict, rho, sigma, tau0, 
     tau1, covariate_dist, expected_prop) = config_param.values()

    # Define standard simulation scenarios
    standard_scenarios = []
    for N in n_obs:
        for n in n_predict:
            # Do not include combinations where N/n is too small or large
            if N/n <= 2 or N/n >= 200:
                continue

            # Increase number of simulation iterations for small samples
            if N < 100:
                multiplier = n_sim_multiplier[1]
            else:
                multiplier = n_sim_multiplier[0]

            new_scenario = {
                'param_id' : f"{N}_{n}_standard",
                'run_date' : run_date,
                'n_sim' : multiplier/n,
                'n_obs' : N,
                'n_predict' : n,
                'rho' : rho[0],
                'sigma' : sigma[0],
                'tau0' : tau0[0],
                'tau1' : tau1[0],
                'covariate_dist' : covariate_dist[0],
                'expected_prop' : expected_prop[0]
            }
            standard_scenarios.append((new_scenario, config_kwargs))

    # Define non-standard simulation scenarios:
    non_standard_scenarios = []
    default = {
        'param_id' : 'default',
        'run_date' : date.today(), 
        'n_sim' : 800,
        'n_obs': 100,
        'n_predict' : 10, 
        'rho': 0.5,
        'sigma': 0.5,
        'tau0': 0.2,
        'tau1': 0.2,
        'covariate_dist' : 'bernoulli',
        'expected_prop' : 0.5
    }
    update = ['rho', 'sigma', 'tau', 'covariate_dist', 'expected_prop']
    for key in update:
        if key == 'tau':
            for t0 in config_param['tau0'][1:]:
                for t1 in config_param['tau1'][1:]:
                    new_scenario = copy.deepcopy(default)
                    new_scenario.update({'tau0' : t0})
                    new_scenario.update({'tau1' : t1})
                    new_scenario.update({'param_id' : f"{key}_{t0}_{t1}_nonstandard"})
                    non_standard_scenarios.append((new_scenario, config_kwargs))
        else:
            for val in config_param[key][1:]:
                new_scenario = copy.deepcopy(default)
                new_scenario.update({key : val})
                new_scenario.update({'param_id' : f"{key}_{val}_nonstandard"})
                non_standard_scenarios.append((new_scenario, config_kwargs))
    
    all_scenarios = standard_scenarios + non_standard_scenarios
    return all_scenarios

def generate_rng_streams(scenarios, seed):
    rng = np.random.default_rng(seed=seed)
    N = len(scenarios)
    child_rngs = rng.spawn(N)
    return child_rngs

ALL_SCENARIOS = configure_parameters(config_param, config_kwargs)
RNG_STREAMS = generate_rng_streams(ALL_SCENARIOS, 503823)

@profile
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

@profile
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

@profile
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

@profile
def mle(y, X):
    """
        Fit logistic regression model via maximum likelihood and return coefficients

    Parameters
    ----------
    X : array-like
        Matrix of predictors
    y : array-like
        Vector of responses

    Returns
    -------
    ml_coefs : array
        Maximum likelihood coefficients for beta
    ml_cov : array
        Inverse of the observed information matrix
    """
    model = sm.GLM(y, X, family=sm.families.Binomial())
    model_results = model.fit()
    ml_coefs = model_results.params
    information_matrix = -model.hessian(ml_coefs)
    try:
        ml_cov = np.linalg.inv(information_matrix)
    except:
        ml_cov = np.zeros(np.shape(information_matrix))
   
    return ml_coefs, ml_cov

def logistic_irls(y, X, tol=1e-8, max_iter=1000):
    N = np.shape(X)[0]
    n = np.shape(X)[1]

    Z = np.hstack((np.ones((N,1)), X))
    beta = np.ones(n+1)
    convergence = 1
    step_size = 1
    iter = 0
    while (convergence > tol) & (iter < max_iter):
        iter = iter+1
        mu = 1/(1+np.exp(-Z@beta))
        S = np.diag(mu*(1-mu))
        beta_new = beta + step_size*np.linalg.inv(Z.T@S@Z)@Z.T@(y-mu)
        convergence = np.sqrt(sum(abs(beta_new-beta)))
        step_size = step_size/2
        beta = beta_new
    mu_hat = 1/(1+np.exp(-X@beta))
    S_hat = np.diag(mu_hat*(1-mu_hat))
    mle_cov = np.linalg.inv(Z.T@S@Z)
    return beta[1:], mle_cov[1:,1:]

# Still sometimes returns negative variance estimates...
@profile
def empirical_bayes(y, X, ml_coefs, ml_cov, tol=1e-8, max_iter=100):
    """
        Compute empirical bayes estimate beta_hat and its covariance.

    Parameters
    ----------
    y : array-like
        Vector of responses
    X : array-like
        Design matrix
    ml_coefs : array-like
        Maximum likelihood estimate of beta
    ml_cov : array_like
        Inverse of observed information matrix at beta
    tol : float, optional
        Convergence tolerance for method of moments estimation.
        Default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations for method of moments estimation.
        Default is 100.

    Returns
    -------
    beta_hat : array_like
        Empirical bayes estimate of beta
    covb : array_like
        Estimated covariance matrix of beta_hat
    tau2 : float
        Estimate of tau^2
    """
    n = np.size(X, axis=1)
    p = 1
    Vhat = ml_cov
    Z = np.ones((n,1))

    # Initial guess for tau^2 is 0, (see ref 9 in paper) repeat until convergence
    tau2 = 0
    n_iter = 0
    converge = 1
    while(converge > tol and n_iter < max_iter):
        W = np.linalg.inv(Vhat + tau2*np.identity(n))
        pi_hat = np.linalg.inv(Z.T@W@Z)@Z.T@W@ml_coefs
        mu_hat = Z@pi_hat
        e = ml_coefs - mu_hat
        R = e.T@W@e/np.sum(W)
        Vbar = np.sum(W@Vhat)/np.sum(W)
        tau_new = n/(n-p)*R-Vbar
        if tau_new < 0:
            tau_new = 0 # Variance estimate cannot be negative
            break
        converge = abs(tau2-tau_new)
        n_iter = n_iter + 1
        tau2 = tau_new

    # Recompute with final tau2
    W = np.linalg.inv(Vhat + tau2*np.identity(n))
    pi_hat = np.linalg.inv(Z.T@W@Z)@Z.T@W@ml_coefs
    mu_hat = Z@pi_hat
    e = ml_coefs - mu_hat
    R = e.T@W@e/np.sum(W)
    Vbar = np.sum(W@Vhat)/np.sum(W)
    
    H = Z@np.linalg.inv(Z.T@W@Z)@Z.T@W
    T = tau2*np.identity(n)
    Vstar = (n-p-2)*Vhat/(n-p)
    Tstar = T + Vhat - Vstar
    G = W@(Vstar@H@W+Tstar)
    beta_hat = G@ml_coefs

    B = (n-2-p)/(n-p)*W*Vhat
    #beta_hat = B@(np.ones(n))*mu_hat + (np.identity(n)-B)@ml_coefs


    H = Z@np.linalg.inv(Z.T@W@Z)@Z.T@W
    A = 2/(n-p)*B@np.outer(e, e)@B.T
    Vbar = W@Vhat/np.sum(W)

    covb = np.zeros(n)
    for i in range(n):
        covb[i] = (Vhat[i,i] - (1-H[i,i])*(Vhat@B)[i,i] + 
        (Vbar[i,i] + tau2)*W[i,i]*A[i,i])
        covb[i] = max(0, covb[i]) # Set to 0 if variance is negative

    return beta_hat, covb, tau2

@profile
def semi_bayes(y, X, ml_coefs, ml_cov, tau_guess):
    """
        Compute Semi-Bayes estimate beta_hat and its covariance.

    Parameters
    ----------
    y : array-like
        Vector of responses
    X : array-like
        Design matrix
    ml_coefs : array-like
        Maximum likelihood estimate of beta
    ml_cov : array_like
        Inverse of observed information matrix at beta
    tau_guess : float
        Assumed prior of tau^2, the variance of the stage II random errors

    Returns
    -------
    beta_hat : array_like
        Semi-bayes estimate of beta
    covb : array_like
        Estimated covariance matrix of beta_hat
    """
    n = np.size(X, axis=1)
    p = 1
    Vhat = ml_cov
    mu_hat = np.mean(ml_coefs)

    W = np.linalg.inv(Vhat + tau_guess*np.identity(n))
    B = W@Vhat
    beta_hat = B@np.ones(n)*mu_hat + (np.identity(n)-B)@ml_coefs

    # Compute covariance estimate with adjustment
    Z = np.ones((n,1))
    H = Z@np.linalg.inv(Z.T@W@Z)@Z.T@W
    
    covb = np.zeros(n)
    for i in range(n):
        covb[i] = Vhat[i,i] - (1-H[i,i])*(Vhat@B)[i,i]

    return beta_hat, covb

@profile
def preliminary_testing(y, X, ml_coefs, ml_covs, alpha=0.10):
    """
        Compute estimate of beta via maximum likelihood with subset selection

    Parameters
    ----------
    y : array-like
        Vector of responses
    X : array-like
        Design matrix
    ml_coefs : array-like
        Maximum likelihood estimate of beta
    ml_cov : array_like
        Inverse of observed information matrix at beta
    alpha : float
        Threshold for selecting covariates.

    Returns
    -------
    beta_hat : array_like
        Semi-bayes estimate of beta
    beta_cov : array_like
        Diagonal of inverse of observed information matrix from the full model,
        evaluated at beta_hat
    """
    n = np.size(X, axis=1)
    keep = [False]*n

    # Determine which covariates to retain
    for i in range(n):
        t = ml_coefs[i]/np.sqrt(ml_covs[i,i])
        keep[i] = 2*(1-norm.cdf(abs(t))) <= alpha

    # If nothing selected, return nothing
    if(not np.any(keep)):
        return [None]*n, [None]*n

    # Compute new estimates
    Xnew = X[:,keep]
    b_hat, b_cov = mle(y, Xnew)
    beta_hat = np.zeros(n)
    beta_hat[keep] = b_hat

    # Update covariance estimate to use hessian from original model
    og_model = sm.GLM(y, X, family=sm.families.Binomial())
    information_matrix = -og_model.hessian(beta_hat)
    beta_cov = np.diag(np.linalg.inv(information_matrix))

    return beta_hat, beta_cov

def compute_estimates(y, X, tau_guess, tol=1e-8, max_iter=100, alpha=0.10):
    """
        Compute different estimates for beta for comparison in simulation.

    Parameters
    ----------
    y : array-like
        Vector of responses
    X : array-like
        Design matrix
    ml_coefs : array-like
        Maximum likelihood estimate of beta
    ml_cov : array_like
        Inverse of observed information matrix at beta
    tau_guess : float
        Assumed prior of tau^2, the variance of the stage II random errors
    tol : float, optional
        Convergence tolerance for method of moments estimation.
        Default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations for method of moments estimation.
        Default is 100.
    alpha : float
        Threshold for selecting covariates.

    Returns
    -------
    estimates : array_like
        (6,n) matrix of estimates of beta, each row 
        representing a different method. The last row may have 'None'
        in all entries if no predictors were selected.
    variances : array_like
        (6,n) matrix of estimates of variances of beta_hat, each row 
        representing a different method. The last row may have 'None'
        in all entries if no predictors were selected.
    tau2 : float
        Empirical Bayes estimate of tau^2

    See also
    --------
    mle, empirical_bayes, semi_bayes, preliminary_testing
    """
    n = np.shape(X)[1]
    ml_beta, ml_cov = mle(y,X)
    if np.array_equal(ml_cov, np.zeros((n,n))):
        return np.array([[None]*n]*6), np.array([[None]*n]*6), None
    eb_beta, eb_cov, tau2 = empirical_bayes(y, X, ml_beta, ml_cov, tol=tol, max_iter=max_iter)
    sb_half_beta, sb_half_cov = semi_bayes(y, X, ml_beta, ml_cov, tau_guess=0.5*tau_guess)
    sb_beta, sb_cov = semi_bayes(y, X, ml_beta, ml_cov, tau_guess=tau_guess)
    sb_two_beta, sb_two_cov = semi_bayes(y, X, ml_beta, ml_cov, tau_guess=2*tau_guess)
    pt_beta, pt_cov = preliminary_testing(y, X, ml_beta, ml_cov, alpha=alpha)

    estimates = [ml_beta, eb_beta, sb_half_beta, sb_beta, sb_two_beta, pt_beta]
    estimates = np.array(estimates)

    variances = [np.diag(ml_cov), eb_cov, sb_half_cov,
                   sb_cov, sb_two_cov, pt_cov]
    variances = np.array(variances)

    return estimates, variances, tau2

@profile
def find_valid_est(estimates):
    """
        Find estimates which are not None
    """
    if len(np.shape(estimates)) == 2:
        valid_indices = ([index for index, element 
                          in enumerate(estimates[:,0]) if element != None])
    else:
        valid_indices = ([index for index, element 
                          in enumerate(estimates[:,0,0]) if element != None])
    return(valid_indices)

@profile
def pivot_beta(beta, n_obs, n_coefs, n_methods):
    """
        Reorganize beta for easier element-wise operations
    """
    B = np.repeat(beta, n_methods, axis=0)
    B = np.reshape(B, (n_obs, n_methods, n_coefs))

    return B

@profile
def confint(estimates, variances):
    """
        Compute confidence intervals for coefficient estimates.

    Arguments
    ---------
        estimates : array-like
            3-dimensional array of estimates
            indexed by (trial, method, coefficient)
        variances : array-like
            3-dimensional array of variances 
            indexed by (trial, method, coefficient)
    """
    variances = np.array(variances, dtype=float)
    lower = estimates - 1.96*np.sqrt(variances)
    upper = estimates + 1.96*np.sqrt(variances)

    return lower, upper

@profile
def mean_coverage_rate(lower, upper, B):
    """
        Compute mean coverage rates for each method
    """
    covered = (lower <= B) & (upper >= B)
    if len(np.shape(covered)) == 2:
        mcr = np.nanmean(covered)
    else:
        mcr = np.nanmean(covered, axis=(0,2))

    return mcr

@profile
def mean_length(lower, upper):
    """
        Compute mean length of confidence intervals
    """
    dists = upper - lower
    if len(np.shape(dists)) == 2:
        mean_dist = np.nanmean(dists)
    else:
        mean_dist = np.nanmean(dists, axis=(0,2))

    return mean_dist

@profile
def rmse(estimates, B):
    """
        Compute the RMSE for each method
    """
    squared_errors = (estimates-B)**2
    squared_errors = np.array(squared_errors, dtype=float)
    if len(np.shape(squared_errors)) == 2:
        mse = np.nanmean(squared_errors)
    else:
        mse = np.nanmean(squared_errors, axis=(0,2))
    rmse = np.sqrt(mse)

    return rmse

@profile
def percent_rmse(rmse):
    """
        Compute the rmse as a percentage of MLE RMSE
    """
    pct_rmse = rmse/rmse[0]
    return pct_rmse

@profile
def percent_tau_zero(taus):
    """
        Compute the proportion of iterations where tau^2 was estimated to be 0.
    """
    iszero = [tau == 0 for tau in taus]
    pct_zero = np.nanmean(iszero)
    
    return pct_zero

@profile
def pct_select_none(n_obs, valid_indices):
    """
        Compute the proportion of trials where preliminary testing selected
        no predictors.
    """
    return (n_obs-len(valid_indices))/n_obs

@profile
def avg_pct_selected(pt_estimates, valid_indices):
    """
        Compute the average proportion of predictors selected
    """
    n_obs = np.size(pt_estimates, axis=0)
    n = np.size(pt_estimates, axis=1)
    valid_ests = pt_estimates[valid_indices,:]
    selected = [est != 0 for est in valid_ests]
    pct_selected = np.sum(selected, axis=1)/n
    mean_selected = np.sum(pct_selected)/n_obs

    return mean_selected

def compute_metrics(estimates, variances, taus, beta):
    """
        Compute all metrics for simulation study.
    """
    n_obs = np.size(beta, axis=0)
    n_methods = np.size(estimates, axis=1)
    n_coefs = np.size(beta, axis=1)

    B = pivot_beta(beta, n_obs, n_coefs, n_methods)
    pt_estimates = estimates[:,5,:]
    pt_variances = variances[:,5,:]
    valid_pt_indices = find_valid_est(pt_estimates)
    valid_pt_est = pt_estimates[valid_pt_indices,:]
    valid_pt_var = pt_variances[valid_pt_indices,:]

    non_pt_estimates = estimates[:,0:5,:]
    non_pt_variances = variances[:,0:5,:]
    valid_non_pt_indices = find_valid_est(non_pt_estimates)
    valid_non_pt_est = non_pt_estimates[valid_non_pt_indices,:,:]
    valid_non_pt_var = non_pt_variances[valid_non_pt_indices,:,:]

    # Compute metrics for non-PT methods
    lower, upper = confint(valid_non_pt_est, valid_non_pt_var)
    non_pt_mcr = mean_coverage_rate(lower, upper, 
                                    B[valid_non_pt_indices,0:5,:])
    non_pt_length = mean_length(lower, upper)
    non_pt_rmse = rmse(valid_non_pt_est, B[valid_non_pt_indices,0:5,:])
    pct_tau_zero = percent_tau_zero(taus[valid_non_pt_indices])

    # Compute metrics for PT method
    lower, upper = confint(valid_pt_est, valid_pt_var)
    pt_mcr = mean_coverage_rate(lower, upper, B[valid_pt_indices,5,:])
    pt_length = mean_length(lower, upper)
    pt_rmse = rmse(valid_pt_est, B[valid_pt_indices,5,:])
    pct_none = pct_select_none(n_obs, valid_pt_indices)
    mean_selected = avg_pct_selected(pt_estimates, valid_pt_indices)

    # Combine metrics
    mean_coverage_rates = np.concatenate((non_pt_mcr, [pt_mcr]))
    mean_cf_lengths = np.concatenate((non_pt_length, [pt_length]))
    rmses = np.concatenate((non_pt_rmse, [pt_rmse]))
    relative_rmse = percent_rmse(rmses)
    
    return (mean_coverage_rates, mean_cf_lengths, rmses, relative_rmse,
        pct_none, mean_selected, pct_tau_zero)

def simulate_data(all_scenarios, rng_streams):
    """
        Generate and save data
        for all simulation scenarios.
    """
    time.sleep(2)
    sim = 0
    for scenario in all_scenarios:
        # Unpack parameters
        (param_id, run_date, n_sim, N, n, rho, sigma, tau0, tau1,
          covariate_dist, expected_prop) = scenario[0].values()
        tol, max_iter, alpha, effect_prob = scenario[1].values()

        betas = []
        dataframes = []
        rng_states = []

        rng = rng_streams[sim]
        n_sim = int(n_sim)
        sim = sim+1
        for i in range(n_sim):
            rng_states.append(rng.bit_generator.state)
            y, X, beta = generate_data(N, n, tau0, tau1, rho, sigma, rng=rng)
            X = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(n)])
            y = pd.Series(y, name='y')
            df = pd.concat([y, X], axis=1)
            dataframes.append(df)
            betas.append(beta)

        # Save results
        save_data(scenario, dataframes, betas, rng_states)

        estimate(scenario, dataframes)

def estimate(scenario, dataframes):
    """
        Compute estimates for all iterations of a simulation scenario
    """
    # Unpack parameters
    (param_id, run_date, n_sim, N, n, rho, sigma, tau0, tau1,
          covariate_dist, expected_prop) = scenario[0].values()
    tol, max_iter, alpha, effect_prob = scenario[1].values()

    true_tau = effect_prob*(1-effect_prob)*tau1**2 + tau0**2

    estimates = []
    variances = []
    taus = []

    # Unpack data
    n_sim = int(n_sim)
    for i in range(n_sim):
        y = dataframes[i].iloc[:,0].to_numpy()
        X = dataframes[i].iloc[:,1:].to_numpy()

        try:
            ests, vars, tau = compute_estimates(y, X, tau_guess=true_tau, 
                                    tol=tol, max_iter=max_iter, alpha=alpha)
            
            # Ignore estimates where MLE variance was very large
            if np.any(vars[0,:] > 100):
                ests = np.array([[None]*n]*6)
                vars = np.array([[None]*n]*6)
                tau = None

        except Exception as e:
            warnings.warn(f'{e}')
            ests = np.array([[None]*n]*6)
            vars = np.array([[None]*n]*6)
            tau = None
        
        estimates.append(ests)
        variances.append(vars)
        taus.append(tau)

    try:
        estimates = np.array(estimates)
        variances = np.array(variances)
        taus = np.array(taus)
    except Exception:
        print(param_id)

    save_estimates(scenario, estimates, variances, taus)

def estimate_all(all_scenarios):
    for scenario in all_scenarios:
        param_id = scenario[0]["param_id"]
        with open(f"data/simulated/{param_id}_data.pkl", "rb") as file:
            scn, dataframes, betas, rng_states = pickle.load(file)

        estimate(scn, dataframes)

def evaluate(all_scenarios):
    mean_coverage_rates =[]
    mean_cf_lengths = []
    rmses = []
    relative_rmse = []
    pct_none = []
    mean_selected = []
    pct_tau_zero = []
    for scenario in all_scenarios:
        # Unpack parameters
        (param_id, run_date, n_sim, N, n, rho, sigma, tau0, tau1,
          covariate_dist, expected_prop) = scenario[0].values()
        
        param_list = [param_id, n_sim, N, n, rho, sigma, tau0, 
                      tau1, covariate_dist, expected_prop]
        
        # Load data
        with open(f"data/simulated/{param_id}_data.pkl", "rb") as file:
            scn, dataframes, betas, rng_states = pickle.load(file)

        # Load estimates
        with open(f"results/raw/estimates/{param_id}_estimates.pkl", "rb") as file:
            scn, estimates, variances, taus = pickle.load(file)

        sim_metrics = compute_metrics(estimates, variances, taus, betas)

        (mcrs, mlengths, rm, rel_rm,
        pct_0, mean_select, pct_tau_0) = sim_metrics

        mean_coverage_rates.append(np.concatenate((param_list, mcrs)))
        mean_cf_lengths.append(np.concatenate((param_list, mlengths)))
        rmses.append(np.concatenate((param_list, rm)))
        relative_rmse.append(np.concatenate((param_list, rel_rm)))
        pct_none.append(np.concatenate((param_list, [pct_0])))
        mean_selected.append(np.concatenate((param_list, [mean_select])))
        pct_tau_zero.append(np.concatenate((param_list, [pct_tau_0])))

    mean_coverage_rates = np.array(mean_coverage_rates)
    mean_cf_lengths = np.array(mean_cf_lengths)
    rmses = np.array(rmses)
    relative_rmse = np.array(relative_rmse)
    pct_none = np.array(pct_none)
    mean_selected = np.array(mean_selected)
    pct_tau_zero = np.array(pct_tau_zero)

    all_metrics = (mean_coverage_rates, mean_cf_lengths, rmses, relative_rmse,
        pct_none, mean_selected, pct_tau_zero)

    save_metrics(all_metrics)

@profile
def save_data(scenario, dataframes, betas, rng_states):
    params = scenario[0]
    param_id = params['param_id']
    with open(f"data/simulated/{param_id}_data.pkl", "wb") as file:
        pickle.dump((scenario, dataframes, betas, rng_states), file)

@profile
def save_estimates(scenario, estimates, variances, taus):
    params = scenario[0]
    param_id = params['param_id']
    with open(f"results/raw/estimates/{param_id}_estimates.pkl", "wb") as file:
        pickle.dump((scenario, estimates, variances, taus), file)

@profile
def save_metrics(sim_metrics):
    with open(f"results/raw/metrics/simulation_metrics.pkl", "wb") as file:
        pickle.dump(sim_metrics, file)

@profile
def save_intervals(all_scenarios):
    # Save all intervals of beta 1
    intervals = pd.DataFrame(columns=["param_id", "N", "n", 
                                      "method", "lower", "upper"])
    methods = ["ML", "EB", "Semi-Bayes 0.5", "Semi-Bayes", "Semi-Bayes 2", "PT"]
    for scenario, kwargs in all_scenarios:
        param_id = scenario["param_id"]
        N = scenario["n_obs"]
        n = scenario["n_predict"]
        n_sim = scenario["n_sim"]
        with open(f"results/raw/estimates/{param_id}_estimates.pkl", "rb") as file:
            scn, estimates, variances, taus = pickle.load(file)

        with open(f"data/simulated/{param_id}_data.pkl", "rb") as file:
            scn, dataframes, betas, rng_states = pickle.load(file)

        for i in range(6):
            valid_indices = find_valid_est(estimates[:,i,:])
            lower, upper = confint(estimates[valid_indices,i,:], 
                               variances[valid_indices,i,:])
            lower = lower[:,0]
            upper = upper[:,0]
            method_col = [methods[i]]*len(lower)
            n_sim_col = [n_sim]*len(lower)
            N_col = [N]*len(lower)
            n_col = [n]*len(lower)
            param_col = [param_id]*len(lower)
            betas = np.array(betas)
            beta_col = betas[valid_indices, 0]
            new_intervals = {
                'param_id' : param_col,
                'n_sim' : n_sim_col,
                'N' : N_col,
                'n' : n_col,
                'method' : method_col,
                'lower' : lower,
                'upper' : upper,
                'beta' : beta_col
            }
            confints = pd.DataFrame(new_intervals)
            intervals = pd.concat([intervals, confints], ignore_index=True)

    with open(f"results/raw/metrics/simulation_intervals.pkl", "wb") as file:
        pickle.dump(intervals, file)  

if __name__ == "__main__":
    simulate_data(ALL_SCENARIOS, RNG_STREAMS)
    evaluate(ALL_SCENARIOS)
    save_intervals(ALL_SCENARIOS)