import numpy as np
from datetime import date
from time import time
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from great_tables import GT, loc
from src.config import generate_rng_streams
from src.dgps import generate_data
from src.methods_old import compute_estimates_old
from src.methods import compute_estimates

config_param = {
    'run_date': date.today(), 
    'n_sim_multiplier' : [8000, 16000],
    #'n_obs': [100, 500, 2000, 5000, 10000],
    'n_obs': np.logspace(2, 4, 20).astype(np.int64),
    'n_predict' : [10, 25, 50, 100], 
    'rho': [0.5],
    'sigma': [0.5],
    'tau0': [0.2],
    'tau1': [0.2],
    'covariate_dist' : ['bernoulli'],
    'expected_prop' : [0.5]
}

config_kwargs = {
    'tol' : 1e-8,
    'max_iter' : 1000,
    'alpha' : 0.10,
    'effect_prob' : 0.2
}

def configure_complexity(config_param, config_kwargs):
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
            # Increase number of simulation iterations for small samples
            if N < 100:
                multiplier = n_sim_multiplier[1]
            else:
                multiplier = n_sim_multiplier[0]

            new_scenario = {
                'param_id' : f"{N}_{n}_standard",
                'run_date' : run_date,
                'n_sim' : 10,
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

    all_scenarios = standard_scenarios
    return all_scenarios

ALL_SCENARIOS = configure_complexity(config_param, config_kwargs)
RNG_STREAMS = generate_rng_streams(ALL_SCENARIOS, 503823)

def simulate_complexity(all_scenarios, rng_streams):
    """
        Generate and save data
        for all simulation scenarios.
    """
    runtimes_old = []
    runtimes_new = []
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

        t0_old = time()
        estimate_complexity(scenario, dataframes, "old")
        t1_old = time()
        elapsed_old = (t1_old-t0_old)/n_sim

        t0_new = time()
        estimate_complexity(scenario, dataframes, "new")
        t1_new = time()
        elapsed_new = (t1_new-t0_new)/n_sim
        runtimes_old.append([N, n, elapsed_old])
        runtimes_new.append([N, n, elapsed_new])
    return runtimes_old, runtimes_new

def estimate_complexity(scenario, dataframes, method="new"):
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
            if method == "old":
                ests, vars, tau = compute_estimates_old(y, X, tau_guess=true_tau, 
                                    tol=tol, max_iter=max_iter, alpha=alpha)
            else:
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

def complexity_plot(runtimes, path, title):
    runtimes = pd.DataFrame(runtimes)
    runtimes.columns = ['N', 'n', 'elapsed']

    x = np.log(runtimes['N'].to_numpy())
    y = np.log(runtimes['elapsed'].to_numpy())

    j = 0
    colors = ["Orange", "Gold", "Blue", "Purple", "Red", "Brown", "Green"]
    for i in np.unique(runtimes['n']):
        ii = np.where(runtimes['n'].to_numpy() == i)
        slope, intercept = np.polyfit(x[ii], y[ii], 1)
        
        line_of_best_fit = slope * x[ii] + intercept
        plt.scatter(x[ii], y[ii], color=colors[j], label=f'{i} features')
        plt.plot(x[ii], line_of_best_fit, color=colors[j])
        r = len(ii[0])
        plt.text(x[ii[0][r-1]], y[ii[0][r-1]], f'Slope: {np.round(slope, decimals=2)}', ha='right', va='center')
        j += 1

    plt.legend()
    plt.title(title)
    plt.xlabel("Log Sample size")
    plt.ylabel("Log Simulation time (s)")
    plt.savefig(path)
    plt.close()

if __name__ == "__main__":
    runtimes_old, runtimes_new = simulate_complexity(ALL_SCENARIOS, RNG_STREAMS)
    complexity_plot(runtimes_old,
                    title="Old average runtime as a function of sample size",
                    path = 'results/figures/baseline_complexity.pdf')
    complexity_plot(runtimes_new,
                    title="New average runtime as a function of sample size",
                    path = 'results/figures/updated_complexity.pdf')