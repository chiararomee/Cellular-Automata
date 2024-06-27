how to use

Rule 30
Using the algorithm we wrote is very easy. We 
ourselves use Jupyter Notebook as executor of the Rule 30 algorithm, so we definitely recommend doing the same. A new file must be created, where we write '\%matplotlib notebook' at the top of the Jupyter Notebook file, to make sure the animations work correctly. Then, we copy and paste the files 'CellularAutomata' and 'Rule30' into the file (these files can be found in GitHub, under the name 'CellularAutomata.py' and 'Rule30.py'). We recommend copying and pasting the entire python document called 'CellularAutomata.py' first. That way, you import all the packages needed right away, as they are at the top of this document. Note that this code pastes under the first line that says '\%matplotlib notebook'. Now you can copy and paste the code from 'Rule30.py'. The first four lines in this code import the same packages as the 'CellularAutomata.py' file, and imports the class Two\_Dim\_CA, but since this class is already defined in the document now, these first four sentences do not need to be put in the Jupyter Notebook document, and therefore do not need to be copied. Below that, the real code starts for the Rule 30 alagorithm, and so only that needs to be placed in the document. 
Before we now run the programme, we can check whether the parameters have been filled in as desired. We have set a certain length, and a certain amount of time steps. This can be adjusted as desired. In addition, the initial state could be adjusted, but since elementary CAs are known for using an initial state where only the middle cell is alive, this is not recommended. 

Game of Life
To run the Game of Life simulation in  Jupyter Notebook, start by opening a new file in Jupyter Notebook. Next, prepare the environment by copying and pasting the content of the 'CellularAutomata.py' (which can be found in GitHub) file into a new cell in your notebook. This file should include all necessary imports and the definition of the Two_Dim_CA class. Ensure that all required packages, such as NumPy and Matplotlib, are installed. Then, copy and paste the file 'GameofLife.py' and paste it at the bottom of your Jupyter Notebook file. Make sure to, again, put'\%matplotlib notebook' at the top of the Jupyter Notebook file, to make sure the animations work correctly.
This code includes the necessary functions and the main simulation logic for defining the Moore-8 neighborhood, setting boundary conditions, implementing the Game of Life rules, and initializing the grid with random states or specific patterns like the glider.
You can choose between random initialization, which assigns random states to cells, and specific pattern initialization, like placing a glider in the grid. The grid size is set to 40x40 by default, but you can adjust it as needed.
Create an instance of the Two_Dim_CA class with the specified grid size, neighborhood function, and boundary conditions. Initialize the grid using either random or pattern-based initialization.
Finally, set up the plot and animate the cellular automaton. Create a plot, display the initial state of the grid, and use the FuncAnimation function to animate the simulation. Adjust parameters such as the number of frames and the interval between frames to control the animation's duration and speed.
By following these steps, you can easily set up and run the Game of Life simulation in Jupyter Notebook. 


Covid Simulation
To run the COVID-19 simulation using Cellular Automata directly in Python, ensure that you have Python installed along with the required packages NumPy and Matplotlib, which can be installed using pip. Prepare the 'CellularAutomata.py' file, which should include all necessary imports and the definition of the Two_Dim_CA class, and place this file in your project directory. Then you can import the 'Two_Dim_CA' class from  'CellularAutomata.py' in our new python file by implementing this line at the top of your file: 'from CellularAutomata import Two_Dim_CA'.
Two initialization functions are provided: one for random initialization, assigning infected states to 20 randomly selected cells, and another for specific pattern initialization, setting the first 20 cells as infected. The grid size is set to 50x50 by default, but this can be adjusted as needed.
An instance of the 'Two_Dim_CA' class is created with the specified grid size, neighborhood function, and boundary conditions. The grid is then initialized using either the random or specific pattern initialization.
Probabilistic rules are defined to govern the behavior of each cell based on its state and the states of its neighbors, simulating the chances of infection, recovery, and death. You can change these parameters to which values you want. Say you want to simulate a situation where people the virus mutates then we want the infection rate to be greater. These rules take into account infection and deceased rates, which can be adjusted to explore different scenarios.
Finally, the plot is set up and the Cellular Automaton is animated using the FuncAnimation function, which creates and displays the animation. Parameters such as the number of frames and the interval between frames can be adjusted to control the duration and speed of the animation.
By following these steps and running the script, you can simulate the spread of COVID-19 and explore the dynamics of disease spread using Cellular Automata directly in Python.
