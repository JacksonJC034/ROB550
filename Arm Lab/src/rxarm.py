"""!
Implements the RXArm class.

The RXArm class contains:

* last feedback from joints
* functions to command the joints
* functions to get feedback from joints
* functions to do FK and IK
* A function to read the RXArm config file

You will upgrade some functions and also implement others according to the comments given in the code.
"""
import numpy as np
from functools import partial
from kinematics import FK_dh, FK_pox, get_pose_from_T, IK_geometric
import time
import csv
import sys, os

from builtins import super
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from resource.config_parse import parse_dh_param_file, parse_pox_param_file
from sensor_msgs.msg import JointState
import rclpy

sys.path.append('../../interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/interbotix_xs_modules/xs_robot') 
from arm import InterbotixManipulatorXS
from mr_descriptions import ModernRoboticsDescription as mrd

"""
TODO: Implement the missing functions and add anything you need to support them
"""
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def _ensure_initialized(func):
    """!
    @brief      Decorator to skip the function if the RXArm is not initialized.

    @param      func  The function to wrap

    @return     The wraped function
    """
    def func_out(self, *args, **kwargs):
        if self.initialized:
            return func(self, *args, **kwargs)
        else:
            print('WARNING: Trying to use the RXArm before initialized')

    return func_out


class RXArm(InterbotixManipulatorXS):
    """!
    @brief      This class describes a RXArm wrapper class for the rx200
    """
    def __init__(self, dh_config_file=None, pox_param_file=None):
        """!
        @brief      Constructs a new instance.

                    Starts the RXArm run thread but does not initialize the Joints. Call RXArm.initialize to initialize the
                    Joints.

        @param      dh_config_file  The configuration file that defines the DH parameters for the robot
        """
        super().__init__(robot_model="rx200")
        self.joint_names = self.arm.group_info.joint_names
        self.num_joints = 5
        # Gripper
        self.gripper_state = True
        # State
        self.initialized = False
        # Cmd
        self.position_cmd = None
        self.moving_time = 2.0
        self.accel_time = 0.5
        # Feedback
        self.position_fb = None
        self.velocity_fb = None
        self.effort_fb = None
        # DH Params
        self.dh_params = []
        self.dh_config_file = dh_config_file
        if (dh_config_file is not None):
            print("DH config file is: ", dh_config_file)
            self.dh_params = RXArm.parse_dh_param_file(self)
        #POX params
        self.M_matrix = []
        self.S_list = []
        self.pox_param_file = pox_param_file
        if (self.pox_param_file is not None):
            self.M_matrix, self.S_list = RXArm.parse_pox_param_file(self)
            # print("M matrix is: ", self.M_matrix)
            # print("S list is: ", self.S_list)
        
        self.use_dh = False if self.dh_params == [] else True

        

    def initialize(self):
        """!
        @brief      Initializes the RXArm from given configuration file.

                    Initializes the Joints and serial port

        @return     True is succes False otherwise
        """
        self.initialized = False
        # Wait for other threads to finish with the RXArm instead of locking every single call
        time.sleep(0.25)
        """ Commanded Values """
        self.position = [0.0] * self.num_joints  # radians
        """ Feedback Values """
        self.position_fb = [0.0] * self.num_joints  # radians
        self.velocity_fb = [0.0] * self.num_joints  # 0 to 1 ???
        self.effort_fb = [0.0] * self.num_joints  # -1 to 1

        # Reset estop and initialized
        self.estop = False
        self.enable_torque()
        self.moving_time = 1.5
        self.accel_time = 0.5
        self.arm.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=False)
        self.gripper.release()
        self.initialized = True
        return self.initialized

    def sleep(self):
        self.moving_time = 2.0
        self.accel_time = 1.0
        self.arm.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=True)
        self.arm.go_to_sleep_pose(moving_time=self.moving_time,
                              accel_time=self.accel_time,
                              blocking=False)
        self.initialized = False

    def set_positions(self, joint_positions, moving_time=None, accel_time=None):
        """!
         @brief      Sets the positions.

         @param      joint_angles  The joint angles
         """
        if moving_time is not None:
            self.moving_time = moving_time
            if accel_time is None:
                self.accel_time = accel_time
        print("Setting joint positions to: ", joint_positions)
        self.arm.set_joint_positions(joint_positions,
                                 moving_time=self.moving_time,
                                 accel_time=self.accel_time,
                                 blocking=False)

    def set_moving_time(self, moving_time):
        self.moving_time = moving_time

    def set_accel_time(self, accel_time):
        self.accel_time = accel_time

    def disable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.core.robot_set_motor_registers('group', 'all', 'Torque_Enable', 0)

    def enable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.core.robot_set_motor_registers('group', 'all', 'Torque_Enable', 1)

    def get_positions(self):
        """!
        @brief      Gets the positions.

        @return     The positions.
        """
        #print("Running get positions and returning: ", self.position_fb)
        return self.position_fb

    def get_velocities(self):
        """!
        @brief      Gets the velocities.

        @return     The velocities.
        """
        return self.velocity_fb

    def get_efforts(self):
        """!
        @brief      Gets the loads.

        @return     The loads.
        """
        return self.effort_fb


#   @_ensure_initialized
    def get_ee_pose(self):
        """!
        @brief      TODO Get the EE pose.

        @return     The EE pose as [x, y, z, phi] or as needed.
        """
        if self.use_dh:
            joint_positions = self.get_positions()
            joint_positions = np.array(joint_positions)
            b = 1.3253
            dh_params = self.get_dh_parameters()
            dh_params = np.array(dh_params)
            print("The shape of the dh_params is: ", dh_params.shape)
            print("The dh params are : ", dh_params)
            dh_params[1][3] = joint_positions[1] - b
            dh_params[2][3] = joint_positions[2] + b
            dh_params[3][3] = joint_positions[3] - (np.pi / 2)
            pos = FK_dh(dh_params=dh_params, joint_angles=joint_positions, link=5)
            pos = pos[0:3, 3]
            # convert the numpy array to a list
            pos = pos.tolist()
        else:
            joint_positions = self.get_positions()
            joint_positions = np.array(joint_positions)
            T = FK_pox(joint_angles=joint_positions, m_mat=self.M_matrix, s_lst=self.S_list)
            pos = get_pose_from_T(T)
            pos = pos.tolist()
        return pos

    def get_joint_angles(self, pose, theta4, theta5 = None):
        """!
        @brief      Gets the joint positions.

        @return     The joint angles based on IK
        """
        #theta4 = pose[4] - np.pi/2
        joint_angles = IK_geometric(None, pose, theta4, theta5)
        joint_angles[1] = joint_angles[1] - 0.033
        joint_angles[2] = joint_angles[2] - 0.024
        return joint_angles

    def move_two_points(self, pose1, pose2, num_waypoints):
        """!
        @brief      Moves the end effector from pose1 to pose2 in num_waypoints steps.

        @param      pose1         The pose 1
        @param      pose2         The pose 2
        @param      num_waypoints The number waypoints
        """
        joint_angles1 = self.get_joint_angles(pose1, np.pi/2)
        self.set_positions(joint_angles1)
        time.sleep(1)
        # try to move to a position that is at the same height as the first position
        # but with x, y coordinates of the second position
        pose_intermediate = [pose2[0], pose2[1], pose1[2]]
        joint_angles_intermediate = self.get_joint_angles(pose_intermediate, np.pi/2)
        self.set_positions(joint_angles_intermediate)
        time.sleep(1)
        joint_angles2 = self.get_joint_angles(pose2, np.pi/2)
        self.set_positions(joint_angles2)
        time.sleep(1)

    @_ensure_initialized
    def get_wrist_pose(self):
        """!
        @brief      TODO Get the wrist pose.

        @return     The wrist pose as [x, y, z, phi] or as needed.
        """
        return [0, 0, 0, 0]

    def parse_pox_param_file(self):
        """!
        @brief      TODO Parse a PoX config file

        @return     0 if file was parsed, -1 otherwise 
        """
        # print("Parsing POX config file...")
        M_matrix, S_list = parse_pox_param_file(self.pox_param_file)
        # print("POX config file parse exit.")
        return M_matrix, S_list

    def parse_dh_param_file(self):
        print("Parsing DH config file...")
        dh_params = parse_dh_param_file(self.dh_config_file)
        print("DH config file parse exit.")
        return dh_params

    def get_dh_parameters(self):
        """!
        @brief      Gets the dh parameters.

        @return     The dh parameters.
        """
        return self.dh_params


class RXArmThread(QThread):
    """!
    @brief      This class describes a RXArm thread.
    """
    updateJointReadout = pyqtSignal(list)
    updateEndEffectorReadout = pyqtSignal(list)

    def __init__(self, rxarm, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      RXArm  The RXArm
        @param      parent  The parent
        @details    TODO: set any additional initial parameters (like PID gains) here
        """
        QThread.__init__(self, parent=parent)
        self.rxarm = rxarm
        self.node = rclpy.create_node('rxarm_thread')
        self.subscription = self.node.create_subscription(
            JointState,
            '/rx200/joint_states',
            self.callback,
            10
        )
        self.subscription  # prevent unused variable warning
        rclpy.spin_once(self.node, timeout_sec=0.5)

    def callback(self, data):
        self.rxarm.position_fb = np.asarray(data.position)[0:5]
        self.rxarm.velocity_fb = np.asarray(data.velocity)[0:5]
        self.rxarm.effort_fb = np.asarray(data.effort)[0:5]
        self.updateJointReadout.emit(self.rxarm.position_fb.tolist())
        self.updateEndEffectorReadout.emit(self.rxarm.get_ee_pose())
        #for name in self.rxarm.joint_names:
        #    print("{0} gains: {1}".format(name, self.rxarm.get_motor_pid_params(name)))
        if (__name__ == '__main__'):
            # print(self.rxarm.position_fb)
            pass

    def run(self):
        """!
        @brief      Updates the RXArm Joints at a set rate if the RXArm is initialized.
        """
        while True:
            rclpy.spin_once(self.node) 
            time.sleep(0.02)


if __name__ == '__main__':
    # rclpy.init() # for test
    rxarm = RXArm(pox_param_file= '../config/rx200_pox.csv')
    print(rxarm.joint_names)
    armThread = RXArmThread(rxarm)
    armThread.start()
    try:
        # joint_positions = [5.0, -47, 27, 0, 1.57]
        rxarm.initialize()
        rxarm.arm.go_to_home_pose()
        time.sleep(3)
        position1 = [250, 275, 30]
        position2 = [250, 275, 10]
        joint_positions1 = rxarm.get_joint_angles(position1, np.pi/2, 45*np.pi/180)
        joint_positions1[1] = joint_positions1[1]
        joint_positions1[2] = joint_positions1[2]
        if joint_positions1 is None:
            print("No solution")
            exit(1)
        rxarm.set_positions(joint_positions1)
        time.sleep(5)
        # for _ in range(10):
        #     joint_positions1 = rxarm.get_joint_angles(position1, np.pi/4)
        #     # rxarm.set_gripper_pressure(0.5)
        #     rxarm.set_positions(joint_positions1)
        #     time.sleep(5)
        #     rxarm.move_two_points(position1, position2, 10)
        #     joint_positions2 = rxarm.get_joint_angles(position2, np.pi/4)
        #     rxarm.set_positions(joint_positions2)
        #     time.sleep(5)
        #     rxarm.gripper.release()
        #     rxarm.set_positions(joint_positions1)
        #     time.sleep(5)
        #     rxarm.set_positions(joint_positions2)
        #     time.sleep(5)
        #     rxarm.gripper.grasp()
        #     rxarm.arm.go_to_home_pose()
        print("Joint positions are: ", rxarm.get_positions())
        pose = rxarm.get_ee_pose()
        print("End effector pose is: ", pose)
        rxarm.gripper.grasp()
        rxarm.arm.go_to_home_pose()
        rxarm.gripper.release()
        rxarm.sleep()

    except KeyboardInterrupt:
        print("Shutting down")

    rclpy.shutdown()