import numpy as np
import pandas as pd
import pickle
import tsplib95
from concurrent.futures import ProcessPoolExecutor, as_completed
from SA import SA

def seed():
    np.random.seed(42)

def run_multiple_simulation(dimension=51, num_i=1000, num_run=50, MC_length=1, alg_type='SA', SA_type='geo', temperature=1, save_file=None, load_file=None): 
    
    seed()
    results = np.zeros((num_run, num_i))

    if alg_type == 'SA': 

        file_prefix = f'dimension_{dimension}_solved_by_SA_with_type_{SA_type}_initial_temperature_{temperature}_MClength_{MC_length}_num_i_{num_i}_num_run_{num_run}'
        if load_file: 
            with open(f'data/{file_prefix}_results.pkl', 'rb') as f:
                results = pickle.load(f)
            with open(f'data/{file_prefix}_route.pkl', 'rb') as f:
                route = pickle.load(f)
            return results, route

        if SA_type == 'lin': 
            for i in range(num_run): 
                    results[i,:], route = SA(dimension, num_i, temperature, MC_length).run_simulation_sa_lin()
        elif SA_type == 'geo': 
            for i in range(num_run): 
                results[i,:], route = SA(dimension, num_i, temperature, MC_length).run_simulation_sa_geo()
        elif SA_type == 'log': 
            for i in range(num_run): 
                    results[i,:], route = SA(dimension, num_i, temperature, MC_length).run_simulation_sa_log()
    
    elif alg_type == 'HC':

        file_prefix = f'dimension_{dimension}_solved_by_HC_with_num_i_{num_i}_num_run_{num_run}'
        if load_file: 
            with open(f'data/{file_prefix}_results.pkl', 'rb') as f:
                results = pickle.load(f)
            with open(f'data/{file_prefix}_route.pkl', 'rb') as f:
                route = pickle.load(f)
            return results, route

        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(num_run): 
                futures.append(executor.submit(SA(dimension, num_i, temperature).run_simulation_hc))
            for i, future in enumerate(as_completed(futures)):
                try:
                    results[i], route = future.result()
                except Exception as e:
                    print(f"Simulation {i} failed with error: {e}")
    if save_file: 
        with open(f'data/{file_prefix}_results.pkl', 'wb') as f: 
             pickle.dump(results, f)
        with open(f'data/{file_prefix}_route.pkl', 'wb') as f: 
             pickle.dump(route, f)
    
    return results, route


if __name__ == '__main__':

    results_hc, route = run_multiple_simulation(dimension=51, num_i=1000000, num_run=50, alg_type='HC', save_file=True)
    print(results_hc[:,-1])
