import numpy as np
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from SA import SA
from plot import line_plot
from functools import partial

def run_sim_SA(i, dimension, num_i, initial_temperature, MC_lengths):
    return SA(dimension, i, num_i, initial_temperature, MC_lengths).run_simulation_sa()

def run_sim_HC(i, dimension, num_i):
    return SA(dimension, i, num_i, no_SA=True).run_simulation_hc()

def run_multiple_simulation(dimension=51, num_i=10, num_run=50, alg_type='SA', initial_temperature=1, MC_lengths=np.array([1, 50, 100]), save_file=None, load_file=None): 
    
    results_all = None

    print(f'Running Algorithm {alg_type}...')
    if alg_type == 'SA': 

        file_prefix = f'dimension_{dimension}_solved_by_SA_initial_temperature_{initial_temperature}_num_i_{num_i}_num_run_{num_run}'
        if load_file: 
            with open(f'data/{file_prefix}_distances.pkl', 'rb') as f:
                distances = pickle.load(f)
            with open(f'data/{file_prefix}_routes.pkl', 'rb') as f:
                routes = pickle.load(f)
            return distances, routes
        
        num_MC_lengths = len(MC_lengths)
        num_schedules = 4
        distances = np.zeros((num_run, num_schedules, num_MC_lengths, num_i+1))
        routes = np.zeros((num_run, num_schedules, num_MC_lengths, dimension))

        run_sim = partial(run_sim_SA, dimension=dimension, num_i=num_i, initial_temperature=initial_temperature, MC_lengths=MC_lengths)
                
    elif alg_type == 'HC':
    
        file_prefix = f'dimension_{dimension}_solved_by_HC_with_num_i_{num_i}_num_run_{num_run}'
        if load_file: 
            with open(f'data/{file_prefix}_distances.pkl', 'rb') as f:
                distances = pickle.load(f)
            with open(f'data/{file_prefix}_routes.pkl', 'rb') as f:
                routes = pickle.load(f)
            return distances, routes

        distances = np.zeros((num_run, num_i+1))
        routes = np.zeros((num_run, dimension))

        run_sim = partial(run_sim_HC, dimension=dimension, num_i=num_i)
    
    else:
        raise ValueError(f'No algo type as {alg_type}')

    print('Starting Simulation SA...')
    with ProcessPoolExecutor(max_workers=10) as ex:
        results_all = list(ex.map(run_sim, range(num_run)))
    print('Finished Simulation SA!')

    for n in range(num_run):
        distances[n] = results_all[n][0]
        routes[n] = results_all[n][1]
        
    if save_file: 
        with open(f'data/{file_prefix}_distances.pkl', 'wb') as f: 
             pickle.dump(distances, f)
        with open(f'data/{file_prefix}_routes.pkl', 'wb') as f: 
             pickle.dump(routes, f)
    
    return distances, routes

if __name__ == '__main__':

    dim = 280
    num_i = 1000000
    num_run = 50
    results_hc, route = run_multiple_simulation(dimension=dim, num_i=num_i, num_run=num_run, alg_type='HC', save_file=True)
    max_diff_dist = np.max((results_hc[:, :-1] - results_hc[:, 1:]).mean(axis=0))
    initial_temps = [max_diff_dist/2, max_diff_dist, max_diff_dist*2]
    for initial_temp in initial_temps:
        run_multiple_simulation(dimension=dim, num_i=num_i, num_run=num_run, alg_type='SA', initial_temperature=initial_temp, save_file=True)

    dim = 442
    results_hc, route = run_multiple_simulation(dimension=dim, num_i=num_i, num_run=num_run, alg_type='HC', save_file=True)
    max_diff_dist = np.max((results_hc[:, :-1] - results_hc[:, 1:]).mean(axis=0))
    initial_temps = [max_diff_dist/2, max_diff_dist, max_diff_dist*2]
    for initial_temp in initial_temps:
        run_multiple_simulation(dimension=dim, num_i=num_i, num_run=num_run, alg_type='SA', initial_temperature=initial_temp, save_file=True)