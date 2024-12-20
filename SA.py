import numpy as np
import tsplib95
import warnings

def load_file(file_path): 
    '''Load the coordinates of each dimension.'''
    dic = tsplib95.load(file_path)

    x_coordinates = [coords[0] for _, coords in list(dic.node_coords.items())]
    y_coordinates = [coords[1] for _, coords in list(dic.node_coords.items())]
    
    return np.array(x_coordinates), np.array(y_coordinates)

def set_seed(seed=42):
    np.random.seed(seed)

class SA(): 
    def __init__(self, dimension, seed, num_i=int(10e5), initial_temperature=1, MC_lengths=np.array([1, 50, 100]), error=1e-4, no_SA=False):

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
        
        self.initial_T = initial_temperature
        self.error = error
        self.num_i = num_i
        self.distance_matrix = self.__get_distance_matrix()

        self.nodes = range(self.dimension)

        self.rng = np.random.default_rng(seed=seed)
        self.initial_route = self.rng.choice(self.nodes, size=self.dimension, replace=False).astype(np.uint16)
        self.random_numbers = self.rng.random(num_i+1).astype(np.float16)

        set_seed(seed)
        self.exchange_pos = np.sort([np.random.choice(self.nodes, size=2, replace=False) for _ in range(num_i+1)]).astype(np.int16)

        if not no_SA:
            self.MC_lengths = MC_lengths
            num_cooling = num_i / MC_lengths
            self.num_diff_MC_lengths = len(MC_lengths)
            self.num_schedules = 4
            self.Ts = np.zeros((self.num_schedules, self.num_diff_MC_lengths, self.num_i+1), dtype=np.float16)
            xs = np.tile(np.arange(num_i+1), (self.num_diff_MC_lengths, 1))
            xs = np.floor(xs/MC_lengths[:, np.newaxis])

            self.coef_cooling_lin = (self.initial_T-error) / num_cooling
            self.coef_cooling_geo = - num_cooling / np.log(error/self.initial_T)
            self.coef_cooling_inv = (self.initial_T/error - 1)/num_cooling
            self.Ts[0] = self.gen_cooling_schedule_lin(xs)
            self.Ts[1] = self.gen_cooling_schedule_geo(xs)
            self.Ts[2] = 0.5*self.Ts[0] + 0.5*self.Ts[1]
            self.Ts[3] = self.gen_cooling_schedule_inv(xs)

    def __get_distance_matrix(self): 
        coordinates = np.vstack((self.x_coordinates, self.y_coordinates)).T
        return np.sqrt(np.sum((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2, axis=2))
    
    def __get_distance(self, route):
        return self.distance_matrix[route, np.append(route[1:], route[0])].sum()
    
    def __get_distances(self, routes):
        """Get distances for multiple route"""
        end_points = np.concatenate((routes[:, :, 1:], routes[:, :, 0, None]), axis=-1)
        new_distances = self.distance_matrix[routes.flatten(), end_points.flatten()].reshape(routes.shape).sum(axis=-1)

        return new_distances

    # def __get_distances(self, routes):
    #     """Get distances for multiple route"""
    #     new_distances = np.zeros(routes.shape[:-1])
    #     for schedule_idx, route_per_MC_length in enumerate(routes):
    #         for MC_length_idx, route in enumerate(route_per_MC_length):
    #             new_distances[schedule_idx, MC_length_idx] = self.__get_distance(route)

    #     return new_distances
    
    def __two_opt(self, route, i):
        """Change single route."""
        position_1, position_2 = self.exchange_pos[i]
        new_route = route.copy()
        new_route[position_1:position_2+1] = route[np.arange(position_2, position_1-1, -1)]
        new_distance = self.__get_distance(new_route) 

        return new_route, new_distance

    def __two_opts(self, routes, i):
        """Change multiple routes."""
        position_1, position_2 = self.exchange_pos[i]
        new_routes = routes.copy()
        new_routes[:, :, position_1:position_2+1] = routes[:, :, np.arange(position_2, position_1-1, -1)]
        new_distances = self.__get_distances(new_routes) 

        return new_routes, new_distances

    def __acceptance_criteria_sa(self, new_routes, new_distances, current_routes, current_distances, i):
        mask = (self.random_numbers[i] <= np.exp(np.minimum(-(new_distances-current_distances)/self.Ts[:, :, i], 0)))
        current_routes[mask] = new_routes[mask]
        current_distances[mask] = new_distances[mask]
        return current_routes, current_distances

    def __acceptance_criteria_hc(self, new_route, new_distance, current_route, current_distance): 
        '''acceptance criteria for hill climbing algorithm'''
        if new_distance < current_distance:
            current_route = new_route
            current_distance = new_distance
        
        return current_route, current_distance

    def run_simulation_sa(self): 
        current_routes = np.tile(self.initial_route, (self.num_schedules, self.num_diff_MC_lengths, 1))
        current_distances = self.__get_distances(current_routes)
        distances_list = np.zeros((self.num_schedules, self.num_diff_MC_lengths, self.num_i+1))

        for i in range(self.num_i+1):
            new_routes, new_distances = self.__two_opts(current_routes, i)
            current_routes, current_distances = self.__acceptance_criteria_sa(new_routes, new_distances, current_routes, current_distances, i)
            distances_list[:, :, i] = current_distances
    
        return distances_list, current_routes
    
    def run_simulation_hc(self): 
        current_route = self.initial_route
        current_distance = self.__get_distance(current_route)
        distance_list = np.zeros((self.num_i+1))

        for i in range(self.num_i+1): 
            new_route, new_distance = self.__two_opt(current_route, i)
            current_route, current_distance = self.__acceptance_criteria_hc(new_route, new_distance, current_route, current_distance)

            distance_list[i] = current_distance
        
        return distance_list, current_route
    
    def gen_cooling_schedule_lin(self, xs): 
        
        return self.initial_T - self.coef_cooling_lin[:, np.newaxis]*xs
    
    def gen_cooling_schedule_geo(self, xs): 

        return self.initial_T * np.exp(-xs/self.coef_cooling_geo[:, np.newaxis])

    def gen_cooling_schedule_inv(self, xs): 

        return self.initial_T * (1/(1+self.coef_cooling_inv[:, np.newaxis]*xs))
    
if __name__=='__main__':
    sa = SA(51, 1)
    print(sa.run_simulation_sa()[0])