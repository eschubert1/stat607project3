import numpy as np

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
    covered = (lower <= B) & (upper >= B)
    if len(np.shape(covered)) == 2:
        mcr = np.nanmean(covered)
    else:
        mcr = np.nanmean(covered, axis=(0,2))

    return mcr

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
    pct_zero = np.nanmean(iszero)
    
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
