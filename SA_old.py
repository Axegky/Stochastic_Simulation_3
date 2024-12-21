import numpy as np
import tsplib95

def load_file(file_path): 
    '''Load the coordinates of each dimension.'''
    dic = tsplib95.load(file_path)

    x_coordinates = [coords[0] for _, coords in list(dic.node_coords.items())]
    y_coordinates = [coords[1] for _, coords in list(dic.node_coords.items())]
    
    return np.array(x_coordinates), np.array(y_coordinates)

def set_seed(seed=42):
    np.random.seed(seed)

class SA(): 
    def __init__(self, dimension, seed, num_i=int(10e5), initial_temperature=1, MC_length=1, error=1e-4):
        if dimension == 51: 
            self.dimension = 51
            self.x_coordinates, self.y_coordinates = load_file("TSP-Configurations\eil51.tsp.txt")
        elif dimension == 280:
            self.dimension = 280
            self.x_coordinates, self.y_coordinates = load_file("TSP-Configurations/a280.tsp.txt")
        elif dimension == 442: 
            self.dimension = 442
            self.x_coordinates, self.y_coordinates = load_file("TSP-Configurations\pcb442.tsp.txt")
        else: 
            raise ValueError('No such file')
        
        rng = np.random.default_rng(seed=seed)
        self.route = rng.choice(range(self.dimension), size=self.dimension, replace=False)
        self.distance_matrix = self.get_distance_matrix()
        self.distance = self.get_distance(self.route)
        self.T = initial_temperature
        num_cooling = num_i / MC_length
        self.coef_cooling_lin = (initial_temperature-error) / num_cooling
        self.coef_cooling_geo = - num_cooling / np.log(error/initial_temperature)
        self.coef_cooling_inv = (initial_temperature/error-1)/num_cooling
        self.num_i = num_i
        self.MC_length = MC_length

        self.nodes = range(self.dimension)


        self.rng = np.random.default_rng(seed=seed)
        self.initial_route = self.rng.choice(self.nodes, size=self.dimension, replace=False).astype(np.uint16)
        self.random_numbers = self.rng.random(num_i+1).astype(np.float16)

        set_seed(seed)
        self.exchange_pos = np.sort([np.random.choice(self.nodes, size=2, replace=False) for _ in range(num_i+1)]).astype(np.int16)

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
    
    def two_opt(self, i): 
        try:
            position_1, position_2 = self.exchange_pos[i]
            new_route = np.copy(self.route)
            new_route[position_1:position_2+1] = self.route[np.arange(position_2, position_1-1, -1)]
            new_distance = self.get_distance(new_route) 
        except:
            print(i)

        return new_route, new_distance

    def acceptance_criteria_sa(self, new_route, new_distance): 
        '''acceptance criteria for simulated annealing algorithm'''
        random = np.random.rand()
        if random <= np.exp(min(-(new_distance-self.distance)/self.T,0)): 
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
        self.T *= np.exp(-1/self.coef_cooling_geo)

    def cooling_schedule_inv(self, counter_cooling): 
        self.T *= (1+self.coef_cooling_inv*counter_cooling) / (1+self.coef_cooling_inv*counter_cooling+self.coef_cooling_inv)

    def run_simulation_sa_lin(self): 
        distance_list = np.zeros(self.num_i)
        counter = 1
        for i in range(self.num_i): 
            new_route, new_distance = self.two_opt()
            self.acceptance_criteria_sa(new_route, new_distance)
            if counter == self.MC_length: 
                self.cooling_schedule_lin()
                counter = 0
            distance_list[i] = self.distance
            counter += 1
        
        return distance_list, self.route
    
    def run_simulation_sa_geo(self): 
        distance_list = np.zeros(self.num_i)
        counter = 1
        for i in range(self.num_i): 
            new_route, new_distance = self.two_opt(i)
            self.acceptance_criteria_sa(new_route, new_distance)
            if counter == self.MC_length: 
                self.cooling_schedule_geo()
                counter = 0
            distance_list[i] = self.distance
            counter += 1

        return distance_list, self.route
    
    def run_simulation_sa_inv(self): 
        distance_list = np.zeros(self.num_i)
        counter = 1
        counter_cooling = 0
        for i in range(self.num_i): 
            new_route, new_distance = self.two_opt(i)
            self.acceptance_criteria_sa(new_route, new_distance)
            if counter == self.MC_length: 
                self.cooling_schedule_inv(counter_cooling)
                counter_cooling += 1
                counter = 0
            distance_list[i] = self.distance
            counter += 1

        return distance_list, self.route
    
    def run_simulation_hc(self): 
        distance_list = np.zeros(self.num_i)
        for i in range(self.num_i): 
            new_route, new_distance = self.two_opt(i)
            print(new_distance)
            self.acceptance_criteria_hc(new_route, new_distance)
            distance_list[i] = self.distance
        
        return distance_list, self.route