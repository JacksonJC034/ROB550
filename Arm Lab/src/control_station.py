#!/usr/bin/python
"""!
Main GUI for Arm lab
"""
import os, sys
script_path = os.path.dirname(os.path.realpath(__file__))

import argparse
import cv2
import numpy as np
import rclpy
import time
from functools import partial
from kinematics import FK_dh

from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QFileDialog

from resource.ui import Ui_MainWindow
from rxarm import RXArm, RXArmThread
from camera import Camera, VideoThread
from state_machine import StateMachine, StateMachineThread
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi

ALPHA = 6 + 180
ALPHA_RADIUS = ALPHA * D2R
COSALPHA = np.cos(ALPHA_RADIUS)
SINALPHA = np.sin(ALPHA_RADIUS)

INTRINSIC_MATRIX = np.array([[921.2914875, 0, 650.25501], [0, 924.10414, 354.34914], [0, 0, 1]])
INV_INTRINSIC_MATRIX = np.linalg.inv(INTRINSIC_MATRIX)

EXTRINSIC_MATRIX = np.array([[1, 0, 0, 10], [0, COSALPHA, -SINALPHA, 130], [0, SINALPHA, COSALPHA, 1025], [0, 0, 0, 1]])
# print("extrinsic matrix",EXTRINSIC_MATRIX)
INV_EXTRINSIC_MATRIX = np.linalg.inv(EXTRINSIC_MATRIX)


class Gui(QMainWindow):
    """!
    Main GUI Class

    Contains the main function and interfaces between the GUI and functions.
    """
    def __init__(self, parent=None, dh_config_file=None, pox_config_file=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        """ Groups of ui commonents """
        self.joint_readouts = [
            self.ui.rdoutBaseJC,
            self.ui.rdoutShoulderJC,
            self.ui.rdoutElbowJC,
            self.ui.rdoutWristAJC,
            self.ui.rdoutWristRJC,
        ]
        self.joint_slider_rdouts = [
            self.ui.rdoutBase,
            self.ui.rdoutShoulder,
            self.ui.rdoutElbow,
            self.ui.rdoutWristA,
            self.ui.rdoutWristR,
        ]
        self.joint_sliders = [
            self.ui.sldrBase,
            self.ui.sldrShoulder,
            self.ui.sldrElbow,
            self.ui.sldrWristA,
            self.ui.sldrWristR,
        ]
        """Objects Using Other Classes"""
        self.camera = Camera()
        # print("Creating rx arm...")
        if dh_config_file is not None:
            print("Creating rx arm with dh config file...")
            self.rxarm = RXArm(dh_config_file=dh_config_file)
        elif pox_config_file is not None:
            print("Creating rx arm with pox config file...")
            self.rxarm = RXArm(pox_param_file=pox_config_file)
        else:
            print("Creating rx arm with default config file...")
            self.rxarm = RXArm()
        print("Done creating rx arm instance.")
        self.sm = StateMachine(self.rxarm, self.camera)
        """
        Attach Functions to Buttons & Sliders
        TODO: NAME AND CONNECT BUTTONS AS NEEDED
        """
        # Video
        self.ui.videoDisplay.setMouseTracking(True)
        self.ui.videoDisplay.mouseMoveEvent = self.trackMouse
        self.ui.videoDisplay.mousePressEvent = self.calibrateMousePress

        # Buttons
        # Handy lambda function falsethat can be used with Partial to only set the new state if the rxarm is initialized
        #nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state if self.rxarm.initialized else None)
        nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state)
        self.ui.btn_estop.clicked.connect(self.estop)
        self.ui.btn_init_arm.clicked.connect(self.initRxarm)
        self.ui.btn_torq_off.clicked.connect(
            lambda: self.rxarm.disable_torque())
        self.ui.btn_torq_on.clicked.connect(lambda: self.rxarm.enable_torque())
        self.ui.btn_sleep_arm.clicked.connect(lambda: self.rxarm.sleep())

        #User Buttons
        self.ui.btnUser1.setText("Calibrate")
        #self.ui.btnUser1.clicked.connect(partial(nxt_if_arm_init, 'calibrate'))
        self.ui.btnUser1.clicked.connect(lambda: self.sm.calibrate())
        self.ui.btnUser2.setText('Open Gripper')
        #self.ui.btnUser2.clicked.connect(lambda: self.rxarm.gripper.release())
        self.ui.btnUser2.clicked.connect(lambda: self.sm.open_gripper())
        self.ui.btnUser3.setText('Close Gripper')
        # self.ui.btnUser3.clicked.connect(lambda: self.rxarm.gripper.grasp())
        self.ui.btnUser3.clicked.connect(lambda: self.sm.close_gripper())
        self.ui.btnUser4.setText('Execute')
        self.ui.btnUser4.clicked.connect(partial(nxt_if_arm_init, 'execute'))

        self.ui.btnUser5.setText("Save Point")
        self.ui.btnUser5.clicked.connect(lambda: self.sm.teach_and_repeat())

        self.world_coordinates = np.array([0, 0, 0])

        #Define our task buttons

        self.ui.btnUser6.setText("Run Task 1")
        self.ui.btnUser6.clicked.connect(lambda: self.sm.task_1())

        self.ui.btnUser7.setText("Run Task 2")
        self.ui.btnUser7.clicked.connect(lambda: self.sm.task_2())

        self.ui.btnUser8.setText("Run Task 3")
        self.ui.btnUser8.clicked.connect(lambda: self.sm.task_3())

        self.ui.btnUser9.setText("Run Task 4")
        self.ui.btnUser9.clicked.connect(lambda: self.sm.task_4())

        self.ui.btnUser10.setText("Run Task 5")
        self.ui.btnUser10.clicked.connect(lambda: self.sm.task_5())

        self.ui.btnUser10.setText("Run Task 6")
        self.ui.btnUser10.clicked.connect(lambda: self.sm.task_bonus())

        self.ui.btnUser11.setText("sweep")
        self.ui.btnUser11.clicked.connect(lambda: self.sm.sweep())

    
        



        # Sliders
        for sldr in self.joint_sliders:
            sldr.valueChanged.connect(self.sliderChange)
        self.ui.sldrMoveTime.valueChanged.connect(self.sliderChange)
        self.ui.sldrAccelTime.valueChanged.connect(self.sliderChange)
        # Direct Control
        self.ui.chk_directcontrol.stateChanged.connect(self.directControlChk)
        # Status
        self.ui.rdoutStatus.setText("Waiting for input")
        """initalize manual control off"""
        self.ui.SliderFrame.setEnabled(False)
        """Setup Threads"""

        # State machine
        self.StateMachineThread = StateMachineThread(self.sm)
        self.StateMachineThread.updateStatusMessage.connect(
            self.updateStatusMessage)
        self.StateMachineThread.start()
        self.VideoThread = VideoThread(self.camera)
        self.VideoThread.updateFrame.connect(self.setImage)
        self.VideoThread.start()
        self.ArmThread = RXArmThread(self.rxarm)
        self.ArmThread.updateJointReadout.connect(self.updateJointReadout)
        self.ArmThread.updateEndEffectorReadout.connect(
            self.updateEndEffectorReadout)
        self.ArmThread.start()

        

    """ Slots attach callback functions to signals emitted from threads"""

    @pyqtSlot(str)
    def updateStatusMessage(self, msg):
        self.ui.rdoutStatus.setText(msg)

    @pyqtSlot(list)
    def updateJointReadout(self, joints):
        pose = self.rxarm.get_ee_pose()
       # print("Passing the theta4 to the IK", joints[3]*180/np.pi)
        # ik_joint_angles= self.rxarm.get_joint_angles(pose, joints[3])
        # output the difference betweeen the joint angles
        # for rdout, joint, ik_joint_angle in zip(self.joint_readouts, joints, ik_joint_angles):
        #    rdout.setText(str('%+.2f' % (joint * R2D)) + " " + str('%+.2f' % (ik_joint_angle * R2D)))
        for rdout, joint in zip(self.joint_readouts, joints):
            rdout.setText(str('%+.2f' % (joint * R2D)))

    ### TODO: output the rest of the orientation according to the convention chosen
    @pyqtSlot(list)
    def updateEndEffectorReadout(self, pos):
        self.ui.rdoutX.setText(str("%+.2f mm" % (pos[0])))
        self.ui.rdoutY.setText(str("%+.2f mm" % (pos[1])))
        self.ui.rdoutZ.setText(str("%+.2f mm" % (pos[2])))
        self.ui.rdoutPhi.setText(str("%+.2f rad" % (pos[3])))
        self.ui.rdoutTheta.setText(str("%+.2f" % (pos[4])))
        self.ui.rdoutPsi.setText(str("%+.2f" % (pos[5])))

    @pyqtSlot(QImage, QImage, QImage, QImage)
    def setImage(self, rgb_image, depth_image, tag_image, grid_image):
        """!
        @brief      Display the images from the camera.

        @param      rgb_image    The rgb image
        @param      depth_image  The depth image
        """
        if (self.ui.radioVideo.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(rgb_image))
        if (self.ui.radioDepth.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(depth_image))
        if (self.ui.radioUsr1.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(tag_image))
        if (self.ui.radioUsr2.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(grid_image))

    """ Other callback functions attached to GUI elements"""

    def estop(self):
        self.rxarm.disable_torque()
        self.sm.set_next_state('estop')

    def sliderChange(self):
        """!
        @brief Slider changed

        Function to change the slider labels when sliders are moved and to command the arm to the given position
        """
        for rdout, sldr in zip(self.joint_slider_rdouts, self.joint_sliders):
            rdout.setText(str(sldr.value()))

        self.ui.rdoutMoveTime.setText(
            str(self.ui.sldrMoveTime.value() / 10.0) + "s")
        self.ui.rdoutAccelTime.setText(
            str(self.ui.sldrAccelTime.value() / 20.0) + "s")
        self.rxarm.set_moving_time(self.ui.sldrMoveTime.value() / 10.0)
        self.rxarm.set_accel_time(self.ui.sldrAccelTime.value() / 20.0)

        # Do nothing if the rxarm is not initialized
        if self.rxarm.initialized:
            joint_positions = np.array(
                [sldr.value() * D2R for sldr in self.joint_sliders])
            # Only send the joints that the rxarm has
            self.rxarm.set_positions(joint_positions[0:self.rxarm.num_joints])

    def directControlChk(self, state):
        """!
        @brief      Changes to direct control mode

                    Will only work if the rxarm is initialized.

        @param      state  State of the checkbox
        """
        if state == Qt.Checked and self.rxarm.initialized:
            # Go to manual and enable sliders
            self.sm.set_next_state("manual")
            self.ui.SliderFrame.setEnabled(True)
        else:
            # Lock sliders and go to idle
            self.sm.set_next_state("idle")
            self.ui.SliderFrame.setEnabled(False)
            self.ui.chk_directcontrol.setChecked(False)

    def trackMouse(self, mouse_event):
        """!
        @brief      Show the mouse position in GUI

                    TODO: after implementing workspace calibration display the world coordinates the mouse points to in the RGB
                    video image.

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """

        pt = mouse_event.pos()
        if self.camera.DepthFrameRaw.any() != 0:
            z = self.camera.DepthFrameRaw[pt.y()][pt.x()]
            #print("printing pt.y and pt.x")
            #print(pt.y())
            self.ui.rdoutMousePixels.setText("(%.0f,%.0f,%.0f)" %
                                             (pt.x(), pt.y(), z))     
            mouse_coordinates_original = np.array([[pt.x()], [pt.y()], [1]])
            # print("printing mouse coords")
            # print(mouse_coordinates_original)
            if self.camera.cameraCalibrated == True:
                self.camera.Inverse_homography_matrix = np.linalg.inv(self.camera.homography_matrix)
                mouse_coordinates = np.dot(self.camera.Inverse_homography_matrix, mouse_coordinates_original)
                mouse_coordinates = mouse_coordinates / mouse_coordinates[2]
            else:
                mouse_coordinates = mouse_coordinates_original
            #print("printing mouse coords")
            #print(mouse_coordinates[1])
            z = self.camera.DepthFrameRaw[int(mouse_coordinates[1, 0])][int(mouse_coordinates[0, 0])]                  
            camera_coordinates = z * np.matmul(INV_INTRINSIC_MATRIX, mouse_coordinates)
            camera_coordinates = np.vstack((camera_coordinates, [1]))
            tag_extrinsic_matrix = self.camera.extrinsic_matrix
            inv_tag_extrinsic_matrix = np.linalg.inv(tag_extrinsic_matrix)
            world_coordinates = np.matmul(inv_tag_extrinsic_matrix, camera_coordinates)
            z_calibration = (1194 * world_coordinates[0] + 2498 * world_coordinates[1] + 1560276)/149902
            self.world_coordinates = np.array([world_coordinates[0], world_coordinates[1], world_coordinates[2] - z_calibration])
            # print(world_coordinates)
            self.ui.rdoutMouseWorld.setText("(%.0f,%.0f,%.0f)" % (world_coordinates[0], world_coordinates[1], world_coordinates[2] - z_calibration))

    def calibrateMousePress(self, mouse_event):
        """!
        @brief Record mouse click positions for calibration

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """
        """ Get mouse posiiton """
        pt = mouse_event.pos()
        self.camera.last_click[0] = pt.x()
        self.camera.last_click[1] = pt.y()
        self.camera.new_click = True
        self.camera.last_world_coordinates = self.world_coordinates
        # self.camera.last_world_coordinates[2] = self.world_coordinates[2] - 19
        waypoint = np.array([self.camera.last_world_coordinates[0],self.camera.last_world_coordinates[1],self.camera.last_world_coordinates[2] + 150])
        
        if(self.sm.clicked == False):
            joint_angles = self.rxarm.get_joint_angles(waypoint, np.pi/2)
            self.rxarm.set_positions(joint_angles)
            time.sleep(3)
            joint_angles = self.rxarm.get_joint_angles(self.camera.last_world_coordinates, np.pi/2)
            self.rxarm.set_positions(joint_angles)
            time.sleep(3)
            self.rxarm.gripper.grasp()
            self.sm.clicked = True
            print("first click")
            return
       

        if(self.sm.clicked == True):
            joint_angles = self.rxarm.get_joint_angles(waypoint, np.pi/2)
            self.rxarm.set_positions(joint_angles)
            time.sleep(3)
            self.camera.last_world_coordinates[2] = self.camera.last_world_coordinates[2]+ 38 
            joint_angles = self.rxarm.get_joint_angles(self.camera.last_world_coordinates, np.pi/2)
            self.rxarm.set_positions(joint_angles)
            time.sleep(3)
            self.rxarm.gripper.release()
            self.sm.clicked = False
            print("second click")
            


        
        # print(self.camera.last_click)

    def initRxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.ui.SliderFrame.setEnabled(False)
        self.ui.chk_directcontrol.setChecked(False)
        self.rxarm.enable_torque()
        self.sm.set_next_state('initialize_rxarm')


### TODO: Add ability to parse POX config file as well
def main(args=None):
    """!
    @brief      Starts the GUI
    """
    app = QApplication(sys.argv)
    if args['dhconfig'] is None and (args['poxconfig'] is not None):
        app_window = Gui(pox_config_file=args['poxconfig'])
    elif args['poxconfig'] is None and (args['dhconfig'] is not None):
        app_window = Gui(dh_config_file=args['dhconfig'])
    else:
        app_window = Gui()
    app_window.show()


    

    # Set thread priorities
    app_window.VideoThread.setPriority(QThread.HighPriority)
    app_window.ArmThread.setPriority(QThread.NormalPriority)
    app_window.StateMachineThread.setPriority(QThread.LowPriority)

    sys.exit(app.exec_())


# Run main if this file is being run directly
### TODO: Add ability to parse POX config file as well
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c",
                    "--dhconfig",
                    required=False,
                    help="path to DH parameters csv file")
    ap.add_argument("-p",
                    "--poxconfig",
                    required=False,
                    help="path to POX config file")
    main(args=vars(ap.parse_args()))
