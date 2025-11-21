import numpy as np
import pandas as pd
import sys
import pickle
import logging
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from great_tables import GT, loc

from src.dgps import generate_data
from src.methods import compute_estimates
from src.metrics import compute_metrics, confint, find_valid_est

if(sys.argv == "log"):
# Log warnings
    logging.basicConfig(
            filename="warnings.log",  # Name of the file to save warnings
            level=logging.WARNING,    # Set the logging level to WARNING
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    logging.captureWarnings(True)

def simulate_data(all_scenarios, rng_streams):
    """
        Generate and save data
        for all simulation scenarios.
    """
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


def save_data(scenario, dataframes, betas, rng_states):
    params = scenario[0]
    param_id = params['param_id']
    with open(f"data/simulated/{param_id}_data.pkl", "wb") as file:
        pickle.dump((scenario, dataframes, betas, rng_states), file)

def save_estimates(scenario, estimates, variances, taus):
    params = scenario[0]
    param_id = params['param_id']
    with open(f"results/raw/estimates/{param_id}_estimates.pkl", "wb") as file:
        pickle.dump((scenario, estimates, variances, taus), file)

def save_metrics(sim_metrics):
    with open(f"results/raw/metrics/simulation_metrics.pkl", "wb") as file:
        pickle.dump(sim_metrics, file)

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

def generate_figures():
    pass

def zipper_plot():
    with open(f"results/raw/metrics/simulation_intervals.pkl", "rb") as file:
        intervals = pickle.load(file)

    # Filter to standard scenarios
    df_standard = intervals[intervals["param_id"].str.contains("_standard")]
    #df_standard = df_standard[df_standard["method"] != "eb"]

    # Make sure N, n, n_sim, lower, upper, and beta are floats
    df_standard['N'] = pd.to_numeric(df_standard['N'])
    df_standard['n'] = pd.to_numeric(df_standard['n'])
    df_standard['n_sim'] = pd.to_numeric(df_standard['n_sim'])
    df_standard['lower'] = pd.to_numeric(df_standard['lower'])
    df_standard['upper'] = pd.to_numeric(df_standard['upper'])
    df_standard['beta'] = pd.to_numeric(df_standard['beta'])

    # Determine if interval covers beta
    df_standard['covered'] = np.where((np.where(df_standard['lower'] <= df_standard['beta'], 1, 0) +
                            np.where(df_standard['upper'] >= df_standard['beta'], 1, 0))==2,
                            0, 1)

    # Make intervals relative to beta 1 (center by beta)
    df_standard['lower'] = df_standard['lower'] - df_standard['beta']
    df_standard['upper'] = df_standard['upper'] - df_standard['beta']

    # Make length column
    df_standard['length'] = df_standard['upper'] - df_standard['lower']

    # Sort intervals
    df_sorted = df_standard.sort_values(by=['N', 'method', 'covered', 'length'])

    # Compute ranks per group
    df_sorted['row_num'] = range(1, len(df_sorted)+1)
    df_sorted['row_num'] = np.where(np.isnan(df_sorted['length']), np.nan, df_sorted['row_num'])
    df_sorted['row_num'] = np.where(df_sorted['length'] == 0, np.nan, df_sorted['row_num'])
    df_sorted['rank'] = df_sorted.groupby(['N', 'method'])['row_num'].rank(method='average', 
                                                        ascending=True, pct=True, na_option='bottom')

    def divide_two_cols(df_sub):
        return df_sub['rank'] / float(df_sub['n_sim'].max())

    #df_sorted['rel_rank'] = df_sorted.groupby(['N', 'method']).apply(divide_two_cols, 
    #            include_groups=False).reset_index().set_index('level_2').drop(['N', 'method'],axis=1)

    df_sorted['zero'] = 0
    df_sorted['one'] = 1
    # Create grid
    g = sns.FacetGrid(df_sorted, row='N', col='method', hue='covered')

    g = g.map(plt.hlines, 'rank', 'lower', 'upper').set_axis_labels(
        x_var="Confidence interval for $\\beta_1$", 
        y_var="Percentage of observations")
    
    g = g.map(plt.vlines, 'zero', 'zero', 'one', color='black', lw=0.5).set_axis_labels(
        x_var="Confidence interval for $\\beta_1$", 
        y_var="Percentage of simulation iterations")
    
    #g = g.map(plt.scatter, 'beta', 'rel_rank', color='black', s=1)

    plt.tight_layout()

    plt.savefig("results/figures/zipper_plot.pdf")
    plt.savefig("results/figures/zipper_plot.png")

def make_tables():
    with open(f"results/raw/metrics/simulation_metrics.pkl", "rb") as file:
        sim_metrics = pickle.load(file)

    # Unpack metrics
    (mean_coverage_rates, mean_cf_lengths, rmses, relative_rmse,
        pct_none, mean_selected, pct_tau_zero) = sim_metrics
    
    col_names = ["param_id", "n_sim", "N", "n", "rho", "sigma", "tau0", "tau1",
                 "covariate_dist", "expected_prob", "ML", "EB", "Semi-Bayes 0.5",
                 "Semi-Bayes", "Semi-Bayes 2", "PT"]
    
    def make_numeric(df):
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

    def results_table(metrics, title, file_name, subset="_standard"):
        tbl = pd.DataFrame(metrics, columns=col_names)

        tbl = tbl[tbl["param_id"].str.contains(subset)]
        
        tbl = tbl[["N", "n", "ML", "EB", "Semi-Bayes 0.5",
                 "Semi-Bayes", "Semi-Bayes 2", "PT"]]
        make_numeric(tbl)
        tbl = tbl.round(2)
        gt_table = (
            GT(tbl)
            .tab_header(title=title)
            )
        
        gt_table.save(f"results/figures/{file_name}.pdf")
        gt_table.save(f"results/figures/{file_name}.png")
    
    # Make table for mean coverage
    results_table(mean_coverage_rates, 
                  "Mean Coverage Rates by Method", 
                  "mean_coverage_table", 
                  subset="_standard")
    results_table(mean_cf_lengths, 
                  "Mean Interval Lengths by Method", 
                  "mean_lengths_table",
                  subset="_standard")
    results_table(rmses, "Mean RMSE by Method", "mean_rmse_table", subset="_standard")
    results_table(relative_rmse, "RMSE by Method Relative to ML", 
                  "relative_rmse_table", subset="_standard")


def mean_rate_by_length():
    with open(f"results/raw/metrics/simulation_metrics.pkl", "rb") as file:
        sim_metrics = pickle.load(file)

    # Unpack metrics
    (mean_coverage_rates, mean_cf_lengths, rmses, relative_rmse,
        pct_none, mean_selected, pct_tau_zero) = sim_metrics
    
    # Join confidence interval metrics
    df = np.hstack((mean_coverage_rates, mean_cf_lengths[:, -7:-1]))

    # Convert to pandas dataframe
    df = pd.DataFrame(df, columns = ["param_id", "n_sim", "N", "n", "rho", 
                                     "sigma", "tau0", "tau1", "covariate_dist", 
                                     "expected_prop", "coverage_rate-mle",
                        "coverage_rate-eb", "coverage_rate-sb_half", 
                        "coverage_rate-sb", "coverage_rate-sb_two", 
                        "coverage_rate-pt", "cf_length-mle", "cf_length-eb", 
                        "cf_length-sb_half", "cf_length-sb", 
                        "cf_length-sb_two", "cf_length-pt"])

    # Filter to standard scenarios
    df_standard = df[df["param_id"].str.contains("_standard")]

    # Pivot longer
    df_standard = pd.wide_to_long(df_standard, 
                         stubnames=["coverage_rate", "cf_length"],
                         i = 'param_id',
                         j = "method", sep="-", suffix=r'\w+')

    # Convert method to column from index
    method_col = []
    for i in df_standard.index:
        method_col.append(i[1])
    df_standard['method'] = method_col

    # Make sure coverage rate, N, n, and cf length are floats
    df_standard['N'] = pd.to_numeric(df_standard['N'])
    df_standard['n'] = pd.to_numeric(df_standard['n'])
    df_standard['coverage_rate'] = pd.to_numeric(df_standard['coverage_rate'])
    df_standard['cf_length'] = pd.to_numeric(df_standard['cf_length'])

    # Create grid
    g = sns.FacetGrid(df_standard, row='N', hue='method')

    g = g.map(plt.scatter, 'cf_length', 'coverage_rate')

    plt.show()