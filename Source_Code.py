import numpy as np
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors

# Define constants for the state of the forest
EMPTY = 0
TREE = 1
BURNING = 2

# Function to initialize the forest with trees and burning trees
def initialize_forest(n, probTree, probBurning):
    # Create an empty grid with dimensions n x n
    grid = np.empty((n, n), dtype=np.int64)

    # Fill the grid with trees and burning trees according to the given probabilities
    for i in range(n):
        for j in range(n):
            rand = np.random.random()
            if rand < 1 - probTree - probBurning:
                grid[i, j] = EMPTY
            elif rand < 1 - probBurning:
                grid[i, j] = TREE
            else:
                grid[i, j] = BURNING

    return grid

# Function to initialize the forest in parallel
def initialize_forest_parallel(n, probTree, probBurning):
    # determine the number of CPUs and create a processing pool
    CPUs = mp.cpu_count()
    pool = mp.Pool(CPUs)

    # Use starmap to initialize multiple grids in parallel
    results = pool.starmap(initialize_forest, [(n, probTree, probBurning)])

    # Close the pool of workers
    pool.close()
    pool.join()

    # Reshape the array of grids into a single grid with dimensions n x n
    return np.array(results).reshape((n, n))

# Function to extend the boundaries of the grid to implement boundary conditions
def extend_boundaries(grid, boundary_type='reflective'):
    # Get the dimensions of the grid
    n_rows, n_cols = grid.shape

    # Create a new grid with dimensions (n_rows+2) x (n_cols+2) to add ghost cells
    extended_grid = np.zeros((n_rows + 2, n_cols + 2), dtype=int)
    # Copy the original grid to the center of the new grid
    extended_grid[1:-1, 1:-1] = grid
    # Copy the edge cells to the ghost cells
    extended_grid[0, 1:-1] = grid[-1, :]
    extended_grid[-1, 1:-1] = grid[0, :]
    extended_grid[1:-1, 0] = grid[:, -1]
    extended_grid[1:-1, -1] = grid[:, 0]
    extended_grid[0, 0] = grid[-1, -1]
    extended_grid[-1, -1] = grid[0, 0]
    extended_grid[0, -1] = grid[-1, 0]
    extended_grid[-1, 0] = grid[0, -1]

    # Apply the boundary conditions
    if boundary_type == 'absorbing':
        extended_grid[[0, -1], :] = 0
        extended_grid[:, [0, -1]] = 0
    elif boundary_type == 'reflective':
        extended_grid[0, :] = extended_grid[1, :]
        extended_grid[-1, :] = extended_grid[-2, :]
        extended_grid[:, 0] = extended_grid[:, 1]
        extended_grid[:, -1] = extended_grid[:, -2]

    return extended_grid


# Function to get the neighbors of a cell in the grid
def get_neighbors(grid, i, j, neighborhood='moore'):
    n = grid.shape[0]
    
    # Define the indices of the neighbors based on the neighborhood type
    NW, N, NE = (i-1, j-1), (i-1, j), (i-1, j+1) # NorthWest, North, NorthEast
    W, E = (i, j-1), (i, j+1) # West, East
    SW, S, SE = (i+1, j-1), (i+1, j), (i+1, j+1) # SouthWest, South, SouthEast

    if neighborhood == 'moore':
        indices = [NW, N, NE, W, E, SW, S, SE]
    elif neighborhood == 'von_neumann':
        indices = [N, S, W, E]
    else:
        raise ValueError('Invalid neighborhood type')

    # Create an empty list to store the neighbors
    neighbors = []

    # Iterate through the indices and append the valid neighbors to the list
    for x, y in indices:
        if x >= 0 and x < n and y >= 0 and y < n:
            neighbors.append(grid[x][y])

    # Convert the list to a NumPy array and return it
    return np.array(neighbors)


# Function that spreads the fire to the cell
def spread(grid, i, j, probImmune):
    neighbors = get_neighbors(grid, i, j)
    # If any of the neighbors are burning, the cell will also catch fire
    if any(neighbors == BURNING):
        # If the probability of becoming immune to fire is greater than a random number between 0 and 1, the cell becomes empty (immune)
        if np.random.random() < probImmune:
            return EMPTY
        # Otherwise, the cell catches fire
        else:
            return BURNING

    # If none of the neighbors are burning, the cell remains unchanged
    return grid[i][j]


# Function to simulate the spread of fire on a grid for one time step
def apply_spread(grid, probLightning, probImmune):
    n = grid.shape[0]
    extended_grid = extend_boundaries(grid, 'reflective')
    new_grid = np.zeros_like(grid)

    # iterate over the cells of the extended grid
    for i in range(1, n+1):
        for j in range(1, n+1):
            # if the cell is burning, set the corresponding cell in the new grid to empty
            if extended_grid[i][j] == BURNING:
                new_grid[i-1][j-1] = EMPTY
            # if the cell is a tree, simulate the spread of fire using the spread function
            elif extended_grid[i][j] == TREE:
                new_grid[i-1][j-1] = spread(extended_grid, i, j, probImmune)
            # if the cell is empty and a random number is less than the lightning probability, set the corresponding cell in the new grid to burning
            elif extended_grid[i][j] == EMPTY and np.random.random() < probLightning:
                new_grid[i-1][j-1] = BURNING
            # if none of the above conditions are met, set the corresponding cell in the new grid to empty
            else:
                new_grid[i-1][j-1] = EMPTY

    return new_grid


# Function to simulate the apply spread of fire in parallel
def apply_spread_parallel(grid, probLightning, probImmune):
    n = grid.shape[0]
    extended_grid = extend_boundaries(grid, 'reflective')
    new_grid = np.zeros_like(grid)

    # determine the number of CPUs and create a processing pool
    CPUs = mp.cpu_count()
    pool = mp.Pool(CPUs)

    # execute spread function for each cell in the grid
    results = pool.starmap(spread, [(extended_grid, i, j, probImmune) for i in range(1, n+1) for j in range(1, n+1)])

    # close the pool
    pool.close()
    pool.join()

    # reshape the results into a 2D array
    results = np.array(results).reshape((n, n))

    # update new grid based on the results of spread function
    new_grid[grid == BURNING] = EMPTY
    new_grid = np.where(grid == TREE, results, new_grid)
    new_grid[(grid == EMPTY) & (np.random.random(size=(n,n)) < probLightning)] = BURNING

    return new_grid

# Function to simulate the forest fire in sequential implementation
def simulate_forest_fire(n, probTree, probBurning, probLightning, probImmune, num_steps):
    colors_list = [(0.2,0,0), (0,0.5,0), (1,0,0), 'orange']
    cmap = colors.ListedColormap(colors_list)
    bounds = [0, 1, 2, 3]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig = plt.figure(figsize=(25/3, 6.25))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    # Initialize the grid and plot it
    grid = initialize_forest(n, probTree, probBurning)
    im = ax.imshow(grid, cmap=cmap, norm=norm)

    # Define an update function for the animation
    def update(i):
        nonlocal grid
        grid = apply_spread(grid, probLightning, probImmune)
        im.set_array(grid)
        return [im]

    # Create the animation object and save it to a file
    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=100, blit=True)
    ani.save('forest_fire_animation_sequential.gif', writer='pillow')

    # Show the animation
    plt.show()




# Function to simulate forest fire for parallel implementation
def simulate_forest_fire_parallel(n, probTree, probBurning, probLightning, probImmune, num_steps):
    colors_list = [(0.2,0,0), (0,0.5,0), (1,0,0), 'orange']
    cmap = colors.ListedColormap(colors_list)
    bounds = [0, 1, 2, 3]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig = plt.figure(figsize=(25/3, 6.25))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    # Initialize the grid and plot it
    grid = initialize_forest_parallel(n, probTree, probBurning)
    im = ax.imshow(grid, cmap=cmap, norm=norm)

    # Define an update function for the animation
    def update(i):
        nonlocal grid
        grid = apply_spread_parallel(grid, probLightning, probImmune)
        im.set_array(grid)
        return [im]

    # Create the animation object and save it to a file
    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=100, blit=True)
    ani.save('forest_fire_animation_parallel.gif', writer='pillow')

    # Show the animation
    plt.show()
    

if __name__ == '__main__':
  # Parameters for the simulation
  n100, n400, n800, n1000, n1200, n2000 = 100, 400, 800, 1000, 1200, 2000
  probTree = 0.8
  probBurning = 0.01
  probLightning = 0.001
  probImmune = 0.3
  t = 200

  # No parallelisation
  # -----------------------Starts here execution time for n ranging from 100 to 2000--------------------------------------
  startTime = time.time()
  simulate_forest_fire(n100, probTree, probBurning, probLightning, probImmune, t)
  print("Execution time for 100x100 is %s seconds - No Parallelisation" % (time.time() - startTime))

  # -----------------------Ends here execution time for n ranging from 100 to 2000--------------------------------------


  # parallelisation - multiprocessing
  # -----------------------Starts here execution time for n ranging from 100 to 2000--------------------------------------
  startTime = time.time()
  simulate_forest_fire_parallel(n100, probTree, probBurning, probLightning, probImmune, t)
  print("Execution time for 100x100 is %s seconds - Parallelisation with Multiprocessing Pool.starmap" % (time.time() - startTime))

  # -----------------------Ends here execution time for n ranging from 100 to 2000--------------------------------------



