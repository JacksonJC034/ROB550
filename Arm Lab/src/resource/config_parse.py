import sys
import numpy as np

def parse_dh_param_file(dh_config_file):
    assert(dh_config_file is not None)
    f_line_contents = None
    with open(dh_config_file, "r") as f:
        f_line_contents = f.readlines()

    assert(f.closed)
    assert(f_line_contents is not None)
    # maybe not the most efficient/clean/etc. way to do this, but should only have to be done once so NBD
    dh_params = np.asarray([line.rstrip().split(',') for line in f_line_contents[1:]])
    dh_params_numpy = np.zeros((dh_params.shape[0], 4))
    for i in range(dh_params.shape[0]):
        print(len(dh_params[i]))
        if len(dh_params[i]) > 1:
            dh_params_numpy[i] = np.array(dh_params[i])
    print("The dh_params are: ", dh_params)
    dh_params_numpy = dh_params_numpy.astype(float)
    return dh_params_numpy


### TODO: parse a pox parameter file
def parse_pox_param_file(pox_config_file):
    assert (pox_config_file is not None)
    f_line_contents = None
    with open(pox_config_file, "r") as f:
        f_line_contents = f.readlines()
    assert (f.closed)
    assert (f_line_contents is not None)
    M_matrix_params = np.asarray([line.rstrip().split(',') for line in f_line_contents[1:5]])
    S_matrix_params = np.asarray([line.rstrip().split(',') for line in f_line_contents[6:11]])
    M_matrix_numpy = np.zeros((M_matrix_params.shape[0], 4))
    S_matrix_numpy = np.zeros((S_matrix_params.shape[0], 6))
    for i in range(M_matrix_params.shape[0]):
        print(M_matrix_params[i])
        if len(M_matrix_params[i]) > 1:
            M_matrix_numpy[i] = np.array(M_matrix_params[i])
    for i in range(S_matrix_params.shape[0]):
        print(S_matrix_params[i])
        if len(S_matrix_params[i]) > 1:
            S_matrix_numpy[i] = np.array(S_matrix_params[i])
    M_matrix_numpy = M_matrix_numpy.astype(float)
    S_matrix_numpy = S_matrix_numpy.astype(float)
    print("The M_matrix is: ", M_matrix_numpy)
    print("The S_matrix is: ", S_matrix_numpy)
    return M_matrix_numpy, S_matrix_numpy
