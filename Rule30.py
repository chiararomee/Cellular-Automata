import numpy as np
from CellularAutomata import One_Dim_CA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def neighborhood_1d(grid, index):
    length = len(grid)
    left = grid[index[0] - 1] if index[0] > 0 else grid[-1]  
    right = grid[index[0] + 1] if index[0] < length - 1 else grid[0]  
    return [left, right]


def rule_30(state, neighbors):
    left, right = neighbors
    return int(left != (state or right)) 


def boundary_condition_1d(grid):
    return grid

def create_rule_30_ca(length):
    num_states = 2  
    ca_1d = One_Dim_CA(length, num_states, neighborhood_1d, boundary_condition_1d)
    
   
    initial_state = np.zeros(length)
    initial_state[length // 2] = 1
    ca_1d.initialize(lambda size, num_states: initial_state)
    return ca_1d 

def animate_ca():
    length = 101
    ca_1d = create_rule_30_ca(length)
    steps = 100 
    evolution = np.zeros((steps, length))
    evolution[0] = ca_1d.grid
    

    for step in range(1, steps):
        ca_1d.evolve(rule_30)
        evolution[step] = ca_1d.grid


    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(evolution, cmap='binary', interpolation='nearest', aspect='auto')

    def update(frame):
        im.set_array(evolution[:frame + 1])
        return [im]
    
    ani = FuncAnimation(fig, update, frames=steps, interval=200, blit=True)

    plt.show()


if __name__ == "__main__":
    animate_ca()


            
            
