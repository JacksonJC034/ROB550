import yaml
import numpy as np

def read_camera_matrix(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        matrix_data = data['camera_matrix']['data']
        return np.array(matrix_data).reshape(3, 3)

def read_factory_calibration(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        k_index = lines.index('k:\n')
        matrix_data = []
        for i in range(1, 10):
            matrix_data.extend([float(num) for num in lines[k_index + i].split()])
        return np.array(matrix_data).reshape(3, 3)

def calculate_elementwise_percentage_difference(matrix1, matrix2):
    percentage_diff_matrix = np.zeros_like(matrix1)

    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            if matrix1[i, j] == 0 and matrix2[i, j] == 0:
                percentage_diff_matrix[i, j] = 0
            else:
                divisor = matrix2[i, j] if matrix2[i, j] != 0 else np.finfo(float).eps
                percentage_diff_matrix[i, j] = np.abs((matrix1[i, j] - divisor) / divisor * 100)

    return percentage_diff_matrix

yaml_paths = ['calibration_data/trial1/ost.yaml',
              'calibration_data/trial2/ost.yaml',
              'calibration_data/trial3/ost.yaml',
              'calibration_data/trial4/ost.yaml']
factory_calibration_path = 'calibration_data/factory_calibration.txt'

matrices = [read_camera_matrix(path) for path in yaml_paths]
average_matrix = sum(matrices) / len(matrices)

factory_matrix = read_factory_calibration(factory_calibration_path)

percentage_diff_matrix = calculate_elementwise_percentage_difference(average_matrix, factory_matrix)

print("Average Matrix:\n", average_matrix)
print("Factory Calibration Matrix:\n", factory_matrix)
print("Element-wise Percentage Difference Matrix:\n", percentage_diff_matrix)