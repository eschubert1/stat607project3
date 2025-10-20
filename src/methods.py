import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

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
        tau_new = max(0, tau_new) # Variance estimate cannot be negative
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