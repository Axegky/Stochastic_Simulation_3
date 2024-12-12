import numpy as np
import pandas as pd
import tsplib95

def load_file(): 
    coordinate = tsplib95.load(file_path)
    return coordinate



# class SA(): 
#     def __init__(self, T, initial_parameters):
#         df = pd.read_csv("Reverse-Engineering Predator-Prey System\predator-prey-data.csv")

#         self.T = T
#         self.time = df['t'].to_numpy()
#         self.data_x = df['x'].to_numpy()
#         self.data_y = df['y'].to_numpy()
#         self.initial_conditions = [self.data_x[0], self.data_y[0]]
#         self.alpha = initial_parameters[0]
#         self.beta = initial_parameters[1]
#         self.delta = initial_parameters[2]
#         self.gamma = initial_parameters[3]

    

#     def cooling_schedules_linear(self): 
#         self.T -= 1
    
#     def calculate_diff(self, x, y, type==1): 
#         if type == 1: 
#             diff = np.sum(np.absolute(x - self.data_x) + np.absolute(y - self.data_y))
#         else: 
#             diff = np.sum(np.square(np.absolute(x - self.data_x)) + np.square(np.absolute(y - self.data_y)))
        
#         return diff

    
#     def acceptance_criteria(self, diff): 
#         if diff < self.last_diff: 
#             self.SA_update()
#         else: 
#             random = np.random.rand()
            

        


#     def run_simulation(self): 
#         self.last_diff = 
#         for _ in range(max_iter=1000): 
#             current_result = self.solve_odes()
#             x, y = current_result.T
#             diff = self.calculate_diff(self, x, y, type==1)
#             SA_update()
            
            


#             if diff == 0:
#                 break

#         return self.alpha, self.beta, self.delta, self.gamma
