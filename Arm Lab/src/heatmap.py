import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def read_coordinates(filename):
    """Read coordinates from a file."""
    with open(filename, 'r') as file:
        coordinates = [list(map(int, line.strip().split(', '))) for line in file]
    return np.array(coordinates)

def plot_heatmap(errors, x_labels, y_labels, title):
    """Plot a heatmap of the errors with custom x and y labels."""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    im = ax.imshow(errors, cmap='viridis', interpolation='nearest')

    # Setting the x and y axes labels
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotating the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    cbar = plt.colorbar(im)
    # add colorbar label
    cbar.set_label('Total Euclidean Error [mm]', rotation=270, labelpad=15)
    plt.title(title)
    plt.xlabel('World X Coordinates [mm]')
    plt.ylabel('World Y Coordinates [mm]')
    plt.show()

def calculate_rmse(data_predicted, data_actual):
    """
    Calculate the RMSE for three dimensions (x, y, z) separately.
    
    Parameters:
    - data_predicted: numpy array of predicted coordinates with shape (n, 3)
    - data_actual: numpy array of actual coordinates with shape (n, 3)
    
    Returns:
    - A tuple containing the RMSE values for the x, y, and z dimensions.
    """
    if data_predicted.shape != data_actual.shape:
        raise ValueError("The shape of predicted and actual data must be the same.")
    
    # Calculate the squared differences for each dimension
    squared_diffs = (data_predicted - data_actual) ** 2
    
    # Calculate mean squared errors for each dimension
    mse_x = np.mean(squared_diffs[:, 0])
    mse_y = np.mean(squared_diffs[:, 1])
    mse_z = np.mean(squared_diffs[:, 2])
    
    # Calculate the square root of MSEs to get RMSEs
    rmse_x = np.sqrt(mse_x)
    rmse_y = np.sqrt(mse_y)
    rmse_z = np.sqrt(mse_z)
    
    return rmse_x, rmse_y, rmse_z

def calculate_std_dev(data_predicted, data_actual):
    """
    Calculate the standard deviation of the errors for three dimensions (x, y, z) separately.
    
    Parameters:
    - data_predicted: numpy array of predicted coordinates with shape (n, 3)
    - data_actual: numpy array of actual coordinates with shape (n, 3)
    
    Returns:
    - A tuple containing the standard deviation of the errors for the x, y, and z dimensions.
    """
    if data_predicted.shape != data_actual.shape:
        raise ValueError("The shape of predicted and actual data must be the same.")
    
    # Calculate the differences for each dimension
    diffs = data_predicted - data_actual
    
    # Calculate standard deviation for each dimension
    std_dev_x = np.std(diffs[:, 0])
    std_dev_y = np.std(diffs[:, 1])
    std_dev_z = np.std(diffs[:, 2])
    
    return std_dev_x, std_dev_y, std_dev_z

# Paths to the files
path_to_block_coordinates = '../launch/block_coordinates.txt'
path_to_world_coordinates = '../launch/world.txt'

# Load coordinates
block_coordinates = read_coordinates(path_to_block_coordinates)
world_coordinates = read_coordinates(path_to_world_coordinates)

# Calculate total Euclidean error for each location (x, y)
total_errors = np.sqrt(np.sum((block_coordinates[:, :2] - world_coordinates[:, :2]) ** 2, axis=1))

# Calculate total Euclidean error for each location (x, y, z)
# total_errors = np.sqrt(np.sum((block_coordinates[:, :3] - world_coordinates[:, :3]) ** 2, axis=1))

# Reshape the errors array to match the 10x6 layout
errors_reshaped = total_errors.reshape((10, 6))
errors_reshaped = errors_reshaped.T
errors_reshaped = np.flip(errors_reshaped, axis=0)

# Extract unique x and y world coordinates to use as labels
x_labels = np.unique(world_coordinates[:, 0])
y_labels = np.unique(world_coordinates[:, 1])

# Since there are more labels than grid dimensions, you might need to select or average them
# Here's a simple selection for demonstration
x_labels = x_labels[:10]  # Adjust based on your specific needs
y_labels = y_labels[:6]  # Adjust based on your specific needs

# Plot heatmap of total errors with world coordinates as labels
plot_heatmap(errors_reshaped, x_labels, y_labels, 'Accuracy of Block Detection Algorithm')

rmse_x, rmse_y, rmse_z = calculate_rmse(block_coordinates, world_coordinates)
print(f"RMSE X: {rmse_x}, RMSE Y: {rmse_y}, RMSE Z: {rmse_z}")

std_dev_x, std_dev_y, std_dev_z = calculate_std_dev(block_coordinates, world_coordinates)
print(f"Standard Deviation X: {std_dev_x}, Standard Deviation Y: {std_dev_y}, Standard Deviation Z: {std_dev_z}")