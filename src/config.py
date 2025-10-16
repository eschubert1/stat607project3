import numpy as np
from datetime import date

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
    'max_iter' : 100,
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
            if N/n <= 2 or N/n >= 50:
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
                    new_scenario = default
                    new_scenario.update({'tau0' : t0})
                    new_scenario.update({'tau1' : t1})
                    new_scenario.update({'param_id' : f"{key}_{t0}_{t1}_nonstandard"})
                    non_standard_scenarios.append((new_scenario, config_kwargs))
        else:
            for val in config_param[key][1:]:
                new_scenario = default
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
RNG_STREAMS = generate_rng_streams(ALL_SCENARIOS, 123)