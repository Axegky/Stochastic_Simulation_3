import numpy as np
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from SA import SA
from plot import line_plot
from functools import partial

def run_sim_SA(A):
    return A.run_simulation_sa()
def run_sim_HC(A):
    return A.run_simulation_hc()

def run_multiple_simulation(dimension=51, num_i=10, num_run=50, alg_type='SA', initial_temperature=1, MC_lengths=np.array([1, 50, 100]), save_file=None, load_file=None): 
    
    results_all = None

    if alg_type == 'SA': 

        file_prefix = f'dimension_{dimension}_solved_by_SA_initial_temperature_{initial_temperature}_num_i_{num_i}_num_run_{num_run}'
        if load_file: 
            with open(f'data/{file_prefix}_distances.pkl', 'rb') as f:
                distances = pickle.load(f)
            with open(f'data/{file_prefix}_routes.pkl', 'rb') as f:
                routes = pickle.load(f)
            return distances, routes
        
        SAs = [SA(dimension, i, num_i, initial_temperature, MC_lengths) for i in range(num_run)]
        num_MC_lengths = len(MC_lengths)
        num_schedules = SAs[0].num_schedules
        distances = np.zeros((num_run, num_schedules, num_MC_lengths, num_i+1))
        routes = np.zeros((num_run, num_schedules, num_MC_lengths, dimension))

        for sa in SAs:
            print(sa.exchange_pos)

        print('Starting Simulation SA...')
        with ProcessPoolExecutor() as ex:
            results_all = list(ex.map(run_sim_SA, SAs))
        print('Finished Simulation SA!')
                
    elif alg_type == 'HC':
    
        file_prefix = f'dimension_{dimension}_solved_by_HC_with_num_i_{num_i}_num_run_{num_run}'
        if load_file: 
            with open(f'data/{file_prefix}_distances.pkl', 'rb') as f:
                distances = pickle.load(f)
            with open(f'data/{file_prefix}_routes.pkl', 'rb') as f:
                routes = pickle.load(f)
            return distances, routes

        SAs = [SA(dimension, i, num_i, initial_temperature, MC_lengths, no_SA=True) for i in range(num_run)]
        distances = np.zeros((num_run, num_i+1))
        routes = np.zeros((num_run, dimension))

        for sa in SAs:
            print(sa.exchange_pos)
    
        print('Starting Simulation HC...')
        with ProcessPoolExecutor() as ex:
            results_all = list(ex.map(run_sim_HC, SAs))
        print('Finished Simulation HC!')
    
    else:
        raise ValueError(f'No algo type as {alg_type}')

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

    # results_hc, route = run_multiple_simulation(dimension=442, num_i=1000000, num_run=50, alg_type='HC', save_file=True)
    dim = 280
    results_hc, route = run_multiple_simulation(dimension=dim, num_i=100000, num_run=50, initial_temperature=1000, alg_type='HC', save_file=True)
    # print(results_sa[:,0,0,:2])
    # print(results_sa[:,1,0,:2])
    # print(np.where(np.not_equal(np.mean(results_sa[:,1,0,:], axis=0),np.mean(results_sa[:,0,0,:], axis=0)))[0].shape)
    # line_plot(results_sa[:,0,0,:], 'lin')
    # line_plot(results_sa[:,1,0,:], 'exp')
    # line_plot(results_sa[:,2,0,:], 'hybrid')
    # line_plot(results_sa[:,3,0,:], 'inv')
    line_plot(results_hc, f'{dim}_hc')

