import numpy as np
import pandas as pd
import pickle

from src.dgps import generate_data
from src.methods import compute_estimates
from src.metrics import compute_metrics

def simulate_data(all_scenarios, rng_streams):
    """
        Generate and save data
        for all simulation scenarios.
    """
    for scenario in all_scenarios:
        # Unpack parameters
        (param_id, run_date, n_sim, N, n, rho, sigma, tau0, tau1,
          covariate_dist, expected_prop) = scenario[0].values()
        tol, max_iter, alpha, effect_prob = scenario[1].values()

        betas = []
        dataframes = []
        rng_states = []
        for i in range(n_sim):
            rng = rng_streams[i]
            rng_states.append(rng.bit_generator.state())
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
    for i in range(n_sim):
        y = dataframes[i].iloc[:,0].to_numpy()
        X = dataframes[i].iloc[:,1:].to_numpy()
    
        ests, vars, tau = compute_estimates(y, X, tau_guess=true_tau, 
                                    tol=tol, max_iter=max_iter, alpha=alpha)
        
        estimates.append(ests)
        variances.append(vars)
        taus.append(tau)

    estimates = np.array(estimates)
    variances = np.array(variances)
    taus = np.array(taus)

    save_estimates(scenario, estimates, variances, taus)

def estimate_all(all_scenarios):
    for scenario in all_scenarios:
        param_id = scenario[0]["param_id"]
        with open(f"../data/simulated/{param_id}_data.pkl") as file:
            scn, dataframes, betas, rng_states = pickle.load(file)

        estimate(scn, dataframes)

def evaluate(all_scenarios):
    for scenario in all_scenarios:
        param_id = scenario[0]["param_id"]
        
        # Load data
        with open(f"../data/simulated/{param_id}_data.pkl") as file:
            scn, dataframes, betas, rng_states = pickle.load(file)

        # Load estimates
        with open(f"../results/raw/estimates/{param_id}_estimates.pkl") as file:
            scn, estimates, variances, taus = pickle.load(file)

        sim_metrics = compute_metrics(estimates, variances, taus, betas)

        save_metrics(scn, sim_metrics)


def save_data(scenario, dataframes, betas, rng_states):
    params = scenario[0]
    param_id = scenario['param_id']
    with open(f"../data/simulated/{param_id}_data.pkl") as file:
        pickle.dump(scenario, dataframes, betas, rng_states)

def save_estimates(scenario, estimates, variances, taus):
    params = scenario[0]
    param_id = scenario['param_id']
    with open(f"../results/raw/estimates/{param_id}_estimates.pkl") as file:
        pickle.dump(scenario, estimates, variances, taus)

def save_metrics(scenario, sim_metrics):
    params = scenario[0]
    param_id = scenario['param_id']
    with open(f"../results/metrics/{param_id}_metrics.pkl") as file:
        pickle.dump(scenario, sim_metrics)


