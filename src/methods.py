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
    model_results = model.fit(method="IRLS")
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
    beta = np.zeros(n+1)
    mu = 0.5*np.ones(N)
    convergence = 1
    step_size = 1
    iter = 0
    while (convergence > tol) & (iter < max_iter):
        iter = iter+1
        s = np.sqrt(mu*(1-mu))
        ZS = Z*s[:, np.newaxis]
        
        update = np.linalg.solve(ZS.T@ZS, Z.T@(y-mu))
        beta_new = beta + step_size*update
        convergence = np.linalg.norm(beta_new-beta)
        step_size = step_size/2
        beta = beta_new
        mu = 1/(1+np.exp(-Z@beta))
    if iter > max_iter:
        print("Iteration limit reached!")
    ZS = Z*np.reshape(np.sqrt(mu*(1-mu)), (N, 1))
    mle_cov = np.linalg.inv(ZS.T@ZS)
    return beta[1:], mle_cov[1:,1:]

# Made some improvements - reduced matrix inversions by computing eigendecomposition once.
# Eliminated for loop when computing standard errors
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
    eigen, eigenvectors = np.linalg.eig(Vhat)
    d1 = 1/eigen
    Q = np.vstack(eigenvectors)
    while(converge > tol and n_iter < max_iter):
        if tau2 == 0:
            W = Q@np.diag(d1)@Q.T
            Vnum = n
        else:
            W = (np.eye(n)-Q@np.diag(1/(d1+1/tau2))@Q.T/tau2)/tau2
            Vnum = np.sum(W@Vhat)

        e = ml_coefs - Z@np.linalg.inv(Z.T@W@Z)@Z.T@W@ml_coefs
        denom = np.sum(W)
        R = e.T@W@e/denom
        Vbar = Vnum / denom
        tau_new = n/(n-p)*R - Vbar
        if tau_new < 0:
            tau_new = 0 # Variance estimate cannot be negative
            break
        converge = abs(tau2-tau_new)
        n_iter = n_iter + 1
        tau2 = tau_new

    # Recompute with final tau2
    if tau2 == 0:
            W = Q@np.diag(d1)@Q.T
            Vnum = np.eye(n)
    else:
        W = (np.eye(n)-Q@np.diag(1/(d1+1/tau2))@Q.T/tau2)/tau2
        Vnum = W@Vhat

    H = Z@np.linalg.inv(Z.T@W@Z)@Z.T@W
    e = ml_coefs - H@ml_coefs
    denom = np.sum(W)
    R = e.T@W@e/denom
    Vbar = Vnum / denom

    B = Vnum * (n-p-2)/(n-p)
    Be = B@e
    beta_hat = ml_coefs - Be
    vb = np.diag(Vhat@B)
    h = np.diag(H)
    a = np.diag(2*np.outer(Be, Be)/(n-p))
    covb = np.diag(Vhat) - (1-h)*vb + (np.diag(Vbar)+tau2)*np.diag(W)*a
    covb[covb < 0] = 0

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
    mu_hat = np.mean(ml_coefs)

    W = np.linalg.inv(ml_cov + tau_guess*np.identity(n))
    B = W@ml_cov
    beta_hat = B@np.ones(n)*mu_hat + (np.identity(n)-B)@ml_coefs

    # Compute covariance estimate with adjustment
    Z = np.ones((n,1))
    H = Z@np.linalg.inv(Z.T@W@Z)@Z.T@W
    
    covb = np.diag(ml_cov) - (1-np.diag(H))*np.diag(ml_cov@B)

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
    N = np.size(X, axis=0)
    n = np.size(X, axis=1)
    keep = [False]*n

    # Determine which covariates to retain
    t = ml_coefs / np.sqrt(np.diag(ml_covs))
    keep = 2*(1-norm.cdf(abs(t))) <= alpha

    # If nothing selected, return nothing
    if(not np.any(keep)):
        return [None]*n, [None]*n

    # Compute new estimates
    Xnew = X[:,keep]
    b_hat, b_cov = logistic_irls(y, Xnew)
    beta_hat = np.zeros(n)
    beta_hat[keep] = b_hat

    # Update covariance estimate to use hessian from original model
    mu = 1/(1+np.exp(-X@beta_hat))
    s = np.sqrt(mu*(1-mu))
    XS = X*s[:,np.newaxis]
    beta_cov = np.diag(np.linalg.inv(XS.T@XS))

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
    ml_beta, ml_cov = logistic_irls(y,X)
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

