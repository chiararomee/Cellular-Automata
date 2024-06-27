import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
class Grid_of_CA:
   def __init__(self, size, num_states, neighborhood_func, boundary_condition_func):
       self.size = size
       self.num_states = num_states
       self.grid = np.zeros(size, dtype=int)  # Initialize the grid with zeros
       self.recovery_time_grid = np.zeros(size, dtype=int)  # Track recovery time
       self.neighborhood_func = neighborhood_func
       self.boundary_condition_func = boundary_condition_func

   def initialize(self, init_func):
       self.grid = init_func(self.size, self.num_states)
       print("Initial grid state:")
       print(self.grid)

   def evolve(self, rule_func):
       new_grid = np.zeros_like(self.grid)
       new_recovery_time_grid = np.zeros_like(self.recovery_time_grid)
       for index, _ in np.ndenumerate(self.grid):
           neighbours = self.neighborhood_func(self.grid, index)  # Get neighbours of current cell
           new_state, new_recovery_time = rule_func(self.grid[index], neighbours, self.recovery_time_grid[index])  # Apply rules and update new grid
           new_grid[index] = new_state
           new_recovery_time_grid[index] = new_recovery_time
       self.grid = self.boundary_condition_func(new_grid)
       self.recovery_time_grid = new_recovery_time_grid

class Two_Dim_CA(Grid_of_CA):
   def __init__(self, rows, cols, num_states, neighborhood_fn, boundary_condition_fn):
       super().__init__((rows, cols), num_states, neighborhood_fn, boundary_condition_fn)

   def plot(self):
       plt.figure(figsize=(10, 10))
       plt.imshow(self.grid, cmap='binary', interpolation='nearest')
       plt.xticks([])
       plt.yticks([])
       plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import random


colors = ['white', 'red', 'green', 'black']
cmap = ListedColormap(colors)

def moore_neighborhood(grid, index):
   rows, cols = grid.shape
   row, col = index
   neighbors = []
   if row > 0 and col > 0:
       neighbors.append(grid[row - 1, col - 1])
   if row > 0:
       neighbors.append(grid[row - 1, col])
   if row > 0 and col < cols - 1:
       neighbors.append(grid[row - 1, col + 1])
   if col > 0:
       neighbors.append(grid[row, col - 1])
   if col < cols - 1:
       neighbors.append(grid[row, col + 1])
   if row < rows - 1 and col > 0:
       neighbors.append(grid[row + 1, col - 1])
   if row < rows - 1:
       neighbors.append(grid[row + 1, col])
   if row < rows - 1 and col < cols - 1:
       neighbors.append(grid[row + 1, col + 1])
   return neighbors

def periodic_boundary(grid):
   return grid

def random_initialization(size, num_states):
   grid = np.zeros(size, dtype=int)
   num_infected = 20
   positions = np.random.choice(range(size[0]*size[1]), num_infected, replace=False)
   rows, cols = np.unravel_index(positions, size)
   grid[rows, cols] = 1
   return grid

def infected_initialization(size, num_states):
   grid = np.zeros(size, dtype=int)
   num_infected = 20
   grid[:num_infected] = 1
   return grid

def probablistic_rule(state, neighbors, recovery_time, infection_rate, deceased_rate):
   if state == 0:
       for neighbor in neighbors:
           if neighbor == 1:
               if random.uniform(0, 1) < infection_rate:
                   return 1, 10
   elif state == 1:
       if random.uniform(0, 1) < deceased_rate:
           return 3, 0
       elif recovery_time > 0:
           return 1, recovery_time - 1
       else:
           return 2, 5
   elif state == 2:
       if recovery_time > 0:
           return 2, recovery_time - 1
       else:
           return 0, 0
   elif state == 3:
       return 3, 0
   return state, recovery_time

ca = Two_Dim_CA(rows=50, cols=50, num_states=4, neighborhood_fn=moore_neighborhood, boundary_condition_fn=periodic_boundary)
ca.initialize(init_func=random_initialization)

def update(frame):
   ca.evolve(rule_func=lambda state, neighbors, recovery_time: probablistic_rule(state, neighbors, recovery_time, infection_rate=0.3, deceased_rate=0.01))
   ax.imshow(ca.grid, cmap=cmap, interpolation='nearest')
   return ax,

fig, ax = plt.subplots()
ax.imshow(ca.grid, cmap=cmap, interpolation='nearest')
plt.xticks([])
plt.yticks([])

ani = FuncAnimation(fig, update, frames=100, interval=200)
plt.show()
