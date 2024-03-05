import csv
import numpy as np
from kinematics import *
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 

filepath = r'bag_files/joint_angles.csv' # run in /src
dh_params = np.array([[0.0,-1.5707,103.91,1.57],
                      [205.73,0.0,0.0,-1.31],
                      [200.0,0.0,0.0,1.31],
                      [0.0,-1.5708,0.0,-1.57],
                      [0.0,0.0,175,0.0]])

def read_csv_to_numpy_array(filepath):
    data = []
    with open(filepath, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(i) for i in row])
    return np.array(data)


def plot_joint_states(joint_states):
    time_points = np.arange(joint_states.shape[0]) 
    plt.figure(figsize=(10, 6)) 
    for i in range(joint_states.shape[1]):  
        plt.plot(time_points, joint_states[:, i], label=f'Joint {i+1}')

    plt.title('Joint Angles Over Time') 
    plt.xlabel('Time Point')  
    plt.ylabel('Joint Angle (rad)') 
    plt.legend()  
    plt.grid(True) 
    plt.show()


def plot_trajectory(joint_states, dh_params):
    x, y, z = [], [], []
    
    for state in joint_states:
        T = FK_dh(dh_params, state, len(dh_params))
        pos = T[:3, 3]
        x.append(pos[0])
        y.append(pos[1])
        z.append(pos[2])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x, y, z, label='End Effector Trajectory')
    
    ax.set_title('End Effector Trajectory in 3D Space')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.legend()
    
    plt.show()
    
    
    
if __name__ == "__main__":
    joint_states = read_csv_to_numpy_array(filepath)
    plot_joint_states(joint_states)
    plot_trajectory(joint_states,dh_params)