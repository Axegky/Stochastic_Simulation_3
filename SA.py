import numpy as np
import pandas as pd
import pickle
import tsplib95

def load_file(file_path): 
    '''Load the coordinates of each dimension.'''
    dic = tsplib95.load(file_path)

    x_coordinates = [coords[0] for _, coords in list(dic.node_coords.items())]
    y_coordinates = [coords[1] for _, coords in list(dic.node_coords.items())]
    
    return np.array(x_coordinates), np.array(y_coordinates)

class SA(): 
    def __init__(self, dimension, num_i, temperature=1, MC_length=1):
        if dimension == 51: 
            self.dimension = 51
            self.x_coordinates, self.y_coordinates = load_file("TSP-Configurations\eil51.tsp.txt")
        elif dimension == 280:
            self.dimension = 280
            self.x_coordinates, self.y_coordinates = load_file("TSP-Configurations\a280.tsp.txt")
        elif dimension == 442: 
            self.dimension = 442
            self.x_coordinates, self.y_coordinates = load_file("TSP-Configurations\pcb442.tsp.txt")
        else: 
            raise ValueError('No such file')
        
        rng = np.random.default_rng()
        self.route = rng.choice(range(self.dimension), size=self.dimension, replace=False)
        self.distance_matrix = self.get_distance_matrix()
        self.distance = self.get_distance(self.route)
        self.T = temperature
        self.coef_cooling_lin = temperature * (1 - 0.9 ** num_i) / num_i
        self.coef_cooling_log = np.log(2)
        self.num_i = num_i
        self.MC_length = MC_length
        self.rng = np.random.default_rng()

    def get_distance_matrix(self): 
        distance_matrix = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension): 
            distance_matrix[i] = (self.x_coordinates - self.x_coordinates[i])**2
            distance_matrix[i] += (self.y_coordinates - self.y_coordinates[i])**2
        
        return np.sqrt(distance_matrix)
    
    def get_distance(self, route):
        reshape_route = np.array((route[:-1], route[1:])).T
        reshape_route = np.vstack((reshape_route, np.array([[route[-1], route[0]]])))

        return self.distance_matrix[reshape_route[:, 0], reshape_route[:, 1]].sum()
    
    def two_opt(self): 
        position_1, position_2 = np.sort(self.rng.choice(range(self.dimension), size=2, replace=False))
        new_route = np.copy(self.route)
        new_route[position_1:position_2+1] = self.route[np.arange(position_2, position_1-1, -1)]
        new_distance = self.get_distance(new_route) 

        return new_route, new_distance

    def acceptance_criteria_sa(self, new_route, new_distance): 
        '''acceptance criteria for simulated annealing algorithm'''
        random = np.random.rand()
        if random <= min(np.exp(-(new_distance-self.distance)/self.T),1): 
            self.route = new_route
            self.distance = new_distance

    def acceptance_criteria_hc(self, new_route, new_distance): 
        '''acceptance criteria for hill climbing algorithm'''
        if new_distance < self.distance:
            self.route = new_route
            self.distance = new_distance

    def cooling_schedule_lin(self): 
        # Balance the temperature after num_i iterations of linear and geometric cooling schedules
        self.T -= self.coef_cooling_lin

    def cooling_schedule_geo(self): 
        self.T *= 0.9

    def cooling_schedule_log(self): 
        self.current_iteration += 1
        self.T = self.coef_cooling_log / np.log(1+self.current_iteration)

    def run_simulation_sa_lin(self): 
        distance_list = np.zeros(self.num_i)
        for i in range(self.num_i): 
            new_route, new_distance = self.two_opt()
            self.acceptance_criteria_sa(new_route, new_distance)
            self.cooling_schedule_lin()
            distance_list[i] = self.distance
        
        return distance_list, self.route
    
    def run_simulation_sa_geo(self): 
        distance_list = np.zeros(self.num_i)
        counter = 0
        for i in range(self.num_i): 
            new_route, new_distance = self.two_opt()
            self.acceptance_criteria_sa(new_route, new_distance)
            if counter == self.MC_length: 
                self.cooling_schedule_geo()
                counter = 0
            distance_list[i] = self.distance
            counter += 1
        
        return distance_list, self.route
    
    def run_simulation_sa_log(self): 
        distance_list = np.zeros(self.num_i)
        counter = 0
        self.current_iteration = 0
        for i in range(self.num_i): 
            new_route, new_distance = self.two_opt()
            self.acceptance_criteria_sa(new_route, new_distance)
            if counter == self.MC_length: 
                self.cooling_schedule_log()
                counter = 0
            distance_list[i] = self.distance
        
        return distance_list, self.route
    
    def run_simulation_hc(self): 
        distance_list = np.zeros(self.num_i)
        for i in range(self.num_i): 
            new_route, new_distance = self.two_opt()
            self.acceptance_criteria_hc(new_route, new_distance)
            distance_list[i] = self.distance
        
        return distance_list, self.route
    