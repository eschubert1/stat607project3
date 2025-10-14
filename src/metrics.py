import numpy as np

def find_valid_est(pt_estimates):
    """
        Find preliminary testing estimates which are not None
    """
    valid_indices = [index for index, element in enumerate(pt_estimates[:,0]) if element != None]
    return(valid_indices)

def pivot_beta(beta, n_obs, n_coefs, n_methods):
    """
        Reorganize beta for easier element-wise operations
    """
    B = np.repeat(beta, n_methods, axis=0)
    B = np.reshape(B, (n_obs, n_methods, n_coefs))

    return B

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

def mean_coverage_rate(lower, upper, B):
    """
        Compute mean coverage rates for each method
    """
    covered = lower <= B & upper >= B
    mcr = np.mean(covered, axis=(0,2))

    return mcr

def mean_length(lower, upper):
    """
        Compute mean length of confidence intervals
    """
    dists = upper - lower
    mean_dist = np.mean(dists, axis=(0,2))

    return mean_dist

def rmse(estimates, B):
    """
        Compute the RMSE for each method
    """
    squared_errors = (estimates-B)**2
    squared_errors = np.array(squared_errors, dtype=float)
    mse = np.mean(squared_errors, axis=(0,2))
    rmse = np.sqrt(mse)

    return rmse

def percent_rmse(rmse):
    """
        Compute the rmse as a percentage of MLE RMSE
    """
    pct_rmse = rmse/rmse[0]
    return pct_rmse

def percent_tau_zero(taus):
    """
        Compute the proportion of iterations where tau^2 was estimated to be 0.
    """
    iszero = [tau == 0 for tau in taus]
    pct_zero = np.mean(iszero)
    
    return pct_zero

def pct_select_none(n_obs, valid_indices):
    """
        Compute the proportion of trials where preliminary testing selected
        no predictors.
    """
    return (n_obs-len(valid_indices))/n_obs

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
    n_obs = np.size(estimates, axis=0)
    n_methods = np.size(estimates, axis=1)
    n_coefs = np.size(estimates, axis=2)

    B = pivot_beta(beta, n_obs, n_coefs, n_methods)
    pt_estimates = estimates[:,5,:]
    pt_variances = variances[:,5,:]
    valid_indices = find_valid_est(pt_estimates)
    valid_pt_est = pt_estimates[valid_indices,:]
    valid_pt_var = pt_estimates[valid_indices,:]

    non_pt_estimates = estimates[:,0:4,:]
    non_pt_variances = variances[:,0:4,:]

    # Compute metrics for non-PT methods
    lower, upper = confint(non_pt_estimates, non_pt_variances)
    non_pt_mcr = mean_coverage_rate(lower, upper, B[:,0:4,:])
    non_pt_length = mean_length(lower, upper)
    non_pt_rmse = rmse(non_pt_estimates, B[:,0:4,:])
    pct_tau_zero = percent_tau_zero(taus)

    # Compute metrics for PT method
    lower, upper = confint(valid_pt_est, valid_pt_var)
    pt_mcr = mean_coverage_rate(lower, upper, B[:,5,:])
    pt_length = mean_length(lower, upper)
    pt_rmse = rmse(valid_pt_est, B[:,5,:])
    pct_none = pct_select_none(n_obs, valid_indices)
    mean_selected = avg_pct_selected(pt_estimates, valid_indices)

    # Combine metrics
    mean_coverage_rates = non_pt_mcr + pt_mcr
    mean_cf_lengths = non_pt_length + pt_length
    rmses = non_pt_rmse + pt_rmse
    relative_rmse = percent_rmse(rmses)
    
    return (mean_coverage_rates, mean_cf_lengths, rmses, relative_rmse,
        pct_none, mean_selected, pct_tau_zero)
