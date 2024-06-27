import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Grid_of_CA:
    def __init__(self, size, num_states, neighborhood_func, boundary_condition_func):
        self.size = size
        self.num_states = num_states
        self.grid = np.zeros(size, dtype=int)  
        self.neighborhood_func = neighborhood_func
        self.boundary_condition_func = boundary_condition_func

    def initialize(self, init_func):
        self.grid = init_func(self.size, self.num_states)


    def evolve(self, rule_func):
        new_grid = np.zeros_like(self.grid)
        for index, _ in np.ndenumerate(self.grid):
            neighbours = self.neighborhood_func(self.grid, index) 
            new_grid[index] = rule_func(self.grid[index], neighbours) 
        self.grid = self.boundary_condition_func(new_grid)     


class One_Dim_CA(Grid_of_CA):
    def __init__(self, length, num_states, neighborhood_func, boundary_condition_func):
        super().__init__((length,), num_states, neighborhood_func, boundary_condition_func)

    def plot(self):
        plt.figure(figsize=(10, 1))
        plt.imshow([self.grid], cmap='binary', interpolation='nearest') 
        plt.xticks([]) 
        plt.yticks([])  
        plt.show() 


class Two_Dim_CA(Grid_of_CA):
    def __init__(self, rows, cols, num_states, neighborhood_fn, boundary_condition_fn):
        super().__init__((rows, cols), num_states, neighborhood_fn, boundary_condition_fn) 

    def plot(self): 
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap='binary', interpolation='nearest')
        plt.xticks([]) 
        plt.yticks([])  
        plt.show()

        