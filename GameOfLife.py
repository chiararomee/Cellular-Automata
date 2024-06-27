from CellularAutomata import Two_Dim_CA
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

def moore_neighborhood(grid, index):
    rows, cols = grid.shape
    row, col = index
    neighbors = [
        grid[(row - 1) % rows, (col - 1) % cols],
        grid[(row - 1) % rows, col],
        grid[(row - 1) % rows, (col + 1) % cols],
        grid[row, (col - 1) % cols],
        grid[row, (col + 1) % cols],
        grid[(row + 1) % rows, (col - 1) % cols],
        grid[(row + 1) % rows, col],
        grid[(row + 1) % rows, (col + 1) % cols]
    ]
    return neighbors

def periodic_boundary(grid):
    return grid

def game_of_life_rules(cell_state, neighbors, recovery_time):
    sum_neighbors = sum(neighbors)
    if cell_state == 1:
        if sum_neighbors < 2 or sum_neighbors > 3:
            return 0, 0
        else:
            return 1, recovery_time
    else:
        if sum_neighbors == 3:
            return 1, 0
        else:
            return 0, recovery_time

def random_initialization(size, num_states=2): 
    return np.random.choice([0, 1], size=size)

def glider_initialization(size):
    grid = np.zeros(size, dtype=int)
    glider = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])
    grid[1:4, 1:4] = glider
    return grid

rows, cols = 40, 40

class Two_Dim_CA:
    def __init__(self, rows, cols, num_states, neighborhood_fn, boundary_condition_fn):
        self.size = (rows, cols)
        self.num_states = num_states
        self.grid = np.zeros(self.size, dtype=int)
        self.recovery_time_grid = np.zeros(self.size, dtype=int)
        self.neighborhood_fn = neighborhood_fn
        self.boundary_condition_fn = boundary_condition_fn

    def initialize(self, init_func):
        self.grid = init_func(self.size, self.num_states)
        self.recovery_time_grid = np.zeros(self.size, dtype=int)
        print("Initial grid state:")
        print(self.grid)

    def evolve(self, rule_func):
        new_grid = np.zeros_like(self.grid)
        new_recovery_time_grid = np.zeros_like(self.recovery_time_grid)
        for index, _ in np.ndenumerate(self.grid):
            neighbors = self.neighborhood_fn(self.grid, index)
            new_state, new_recovery_time = rule_func(self.grid[index], neighbors, self.recovery_time_grid[index])
            new_grid[index] = new_state
            new_recovery_time_grid[index] = new_recovery_time
        self.grid = self.boundary_condition_fn(new_grid)
        self.recovery_time_grid = new_recovery_time_grid

    def plot(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap='binary', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        plt.show()

ca = Two_Dim_CA(rows, cols, num_states=2, neighborhood_fn=moore_neighborhood, boundary_condition_fn=periodic_boundary)
ca.initialize(random_initialization)

def update(frame_num):
    ca.evolve(game_of_life_rules)
    im.set_array(ca.grid)
    return [im]

fig, ax = plt.subplots()
im = ax.imshow(ca.grid, cmap='binary', interpolation='nearest')
plt.ion()

ani = FuncAnimation(fig, update, frames=200, interval=100, blit=True)
plt.show()