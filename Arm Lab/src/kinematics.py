"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    transforms = [np.zeros((4, 4)) for i in range(link)]
    for i in range(link):
        transforms[i] = get_transform_from_dh(dh_params[i][0], dh_params[i][1], dh_params[i][2], joint_angles[i] + dh_params[i][3])
    T = np.eye(4)
    for i in range(link):
        T = np.matmul(T, transforms[i])
    return T


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix
    """
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    # print("a: ", a)
    # print("alpha: ", alpha)
    # print("d: ", d)
    # print("theta: ", theta)
    T = np.array([[c_theta, -s_theta*c_alpha, s_theta*s_alpha, a*c_theta],
                  [s_theta, c_theta*c_alpha, -c_theta*s_alpha, a*s_theta ],
                  [0, s_alpha, c_alpha, d],
                  [0, 0, 0, 1]])
    return T


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    # TODO: Handle the singularity cases
    zyz_euler = np.zeros(3)
    zyz_euler[0] = np.arctan2(T[1, 2], T[0, 2])
    zyz_euler[1] = np.arccos(T[2, 2])
    zyz_euler[2] = np.arctan2(T[2, 1], -T[2, 0])
    return zyz_euler


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    pose = np.zeros(6)
    pose[0:3] = T[0:3, 3]
    pose[3:6] = get_euler_angles_from_T(T)
    return pose


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    T = m_mat
    #print("The size of screw list is: ", len(s_lst))
    #print("The screw list is: ", s_lst)
    #print("The joint angles are: ", joint_angles)
    #print("The size of joint angles is: ", len(joint_angles))
    #print("The m matrix is: ", m_mat)
    # reverse index of joint angles
    for i in range(len(joint_angles)-1, -1, -1):
        w = s_lst[i][0:3]
        v = s_lst[i][3:6]
        T = np.matmul(expm(to_s_matrix(w, v)*joint_angles[i]), T)
    #print("The homogeneous matrix is: ", T)
    return T


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    s_matrix = np.zeros((4, 4))
    s_matrix[0:3, 0:3] = to_skew_symmetric(w)
    s_matrix[0:3, 3] = v
    return s_matrix

def to_skew_symmetric(w):
    """!
    @brief      Convert to skew symmetric.
    @param      w     { parameter_description }
    @return     { description_of_the_return_value }
    """
    w_skew = np.zeros((3, 3))
    w_skew[0, 1] = -w[2]
    w_skew[0, 2] = w[1]
    w_skew[1, 0] = w[2]
    w_skew[1, 2] = -w[0]
    w_skew[2, 0] = -w[1]
    w_skew[2, 1] = w[0]
    return w_skew


def IK_geometric(dh_params, pose, theta4, theta5=None):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    x, y, z = pose[0:3]
    if z < 0:
        z = 0.01
    # print("x = ", x)
    # print("y = ", y)
    # y = y - 174.15
    # phi, theta, psi = pose[3:6]
    # print("Theta4: ", theta4)
    joint_angles = np.zeros((1, 5))
    joint_angles[0, 0] = np.arctan2(-x, y)
    end_effector_length = 174.15
    # end_effector_length = 190.15

    
    z = z + end_effector_length*np.sin(theta4)
    a = np.sqrt(x**2 + y**2) - end_effector_length*np.cos(theta4)
    # print("a = ", a)
    # x = a*np.sin(joint_angles[0, 0])
    # y = a*np.cos(joint_angles[0, 0])
    # print("x = ", x)
    # print("y = ", y)
    # print("z = ", z)
    # x = x - end_effector_length*np.sin(joint_angles[0, 0])
    # y = y - end_effector_length*np.cos(joint_angles[0, 0])
    # a = np.sqrt(x**2 + y**2)
    b = z - 103.91
    c = np.sqrt(a**2 + b**2)
    d = 205.73
    # d = 206.15
    alpha = np.arctan(50/200)
    beta1 = np.arccos((c**2 + d**2 - 200**2)/(2*c*d))
    if np.isnan(beta1):
        # avoid infinite loop from recursion
        if theta4 == 0:
            # raise ValueError
            return None
        
        #print("c, d, c**2 + d**2 - 200**2 / 2cd", c, d, (c**2 + d**2 - 200**2)/(2*c*d))
        # raise ValueError
        # change theta4 to 0 to avoid non-reachable points
        theta4 = np.pi/4
        theta4 = 0
        # call the function again
        # but since we are approaching the block from angle zero, we
        # can only approach from one side
        return IK_geometric(dh_params, pose, theta4, theta5)
    
    # if np.isnan(beta1):
    #     theta4 = np.pi/4
    
    beta2 = np.arccos((200**2 + d**2 - c**2)/(2*200*d))
    gamma = np.arctan2(b,a)

    joint_angles[0, 1] = np.pi/2 - (alpha + beta1) - gamma # elbow up
    joint_angles[0, 2] = -beta2 + np.pi/2 + alpha # elbow up

    joint_angles[0, 3] = theta4 - joint_angles[0, 1] - joint_angles[0, 2]
    if theta5 is not None and theta4 != 0:
        joint_angles[0, 4] = joint_angles[0, 0] + theta5
    else:
        joint_angles[0, 4] = 0
    if theta4 == np.pi/4:
        joint_angles[0, 4] = 0
    # print("Joint angles using inverse kinematics in degrees: ", joint_angles*180/np.pi)
    # print("x, y, a, b, c, d", x, y, a, b, c, d)
    # print("alpha in degrees: ", alpha*180/np.pi)
    # print("b = ", b)
    # print("a = ", a)
    # print("x = ", x)
    # print("y = ", y)
    # print("beta1 in degrees: ", beta1*180/np.pi)
    # print("beta2 in degrees: ", beta2*180/np.pi)
    # print("gamma in degrees: ", gamma*180/np.pi)
    # print("beta1 in degrees= ", beta1*180/np.pi)
    # print("Joint angles 2 : ", joint_angles[0, 1]*180/np.pi)
    # print("Joint angles 3: ", joint_angles[0, 2]*180/np.pi)
    # convert numpy array to list
    # remove the first dimension
    joint_angles = joint_angles.reshape(5)
    joint_angles = joint_angles.tolist()
    # print("Joint angles getting returned is: ", joint_angles)
    return joint_angles


