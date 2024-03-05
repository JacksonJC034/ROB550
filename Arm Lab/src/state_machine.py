"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
import cv2



class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
       
        self.clicked = False
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.gripper = None
        self.gripper_pos = []
        self.colors = self.camera.colors
        self.waypoints = [
             [-np.pi/2,       -0.5,      -0.3,          0.0,        0.0],
             [0.75*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
             [0.5*-np.pi/2,   -0.5,      -0.3,      np.pi/2,        0.0],
             [0.25*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
             [0.0,             0.0,       0.0,          0.0,        0.0],
             [0.25*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
             [0.5*np.pi/2,     0.5,       0.3,     -np.pi/3,        0.0],
             [0.75*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
             [np.pi/2,         0.5,       0.3,     -np.pi/3,        0.0],
             [0.0,             0.0,       0.0,          0.0,        0.0]]
        # self.waypoints = []


    def open_gripper(self):
        """!
        @brief      Open the gripper
        """
        self.rxarm.gripper.release()
        self.gripper = 0

    def close_gripper(self):
        """!
        @brief      Close the gripper
        """
        self.rxarm.gripper.grasp()
        self.gripper = 1

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        # rclpy.spin()
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def teach_and_repeat(self):
        print('worked')
        self.waypoint = self.rxarm.get_positions()
        # append the gripper state at the end of the waypoint
        self.gripper_pos.append(self.gripper)
        print("Current gripper state", self.gripper)
        print("current waypoint", self.waypoint)
        self.status_message = "State: Teach and Repeat - Teaching waypoint"
        self.waypoints.append(self.waypoint)
        print("all the waypoints", self.waypoints)

    def sweep(self):
        self.waypoint0 = [100, 0, 0, -90, 0]
        # convert to radians
        self.waypoint0 = [angle*np.pi/180 for angle in self.waypoint0]
        self.waypoint1 = [100, 53, -32, -97, 0]
        self.waypoint1 = [angle*np.pi/180 for angle in self.waypoint1]
        self.waypoint2 = [99.3, 83.31, -86, -88.2, 0]
        self.waypoint2 = [angle*np.pi/180 for angle in self.waypoint2]
        self.waypoint3 = [ 50, 85.31, -86, -88.2, 0]
        self.waypoint3 = [angle*np.pi/180 for angle in self.waypoint3]
        self.waypoint4 = [50, 85.31, -86, -15, 0]
        self.waypoint4 = [angle*np.pi/180 for angle in self.waypoint4]
        # self.waypoint5 = [0, 85.31, -86, -15, 0]
        self.waypoint5 = [20, 85.31, -86, -88.2, 0]
        self.waypoint5 = [angle*np.pi/180 for angle in self.waypoint5]
        self.waypoint6 = [-50, 85.31, -86, -88.2, 0]
        self.waypoint6 = [angle*np.pi/180 for angle in self.waypoint6]
        self.waypoint7 = [-50, 85.31, -86, -88.2, 0]
        self.waypoint7 = [angle*np.pi/180 for angle in self.waypoint7]
        self.waypoint8 = [-99.3, 85.31, -86, -88.2, 0]
        self.waypoint8 = [angle*np.pi/180 for angle in self.waypoint8]
        
        # self.waypoints = [self.waypoint0, self.waypoint1, self.waypoint2, self.waypoint3, self.waypoint4, self.waypoint5, self.waypoint6, self.waypoint7, self.waypoint8, self.waypoint7]
        self.waypoints = [self.waypoint0, self.waypoint1, self.waypoint2, self.waypoint3, self.waypoint5, self.waypoint6, self.waypoint8, self.waypoint7]
        #self.waypoints = [self.waypoint0, self.waypoint1, self.waypoint2, self.waypoint3, self.waypoint5, self.waypoint6, self.waypoint8]
        for i in range(len(self.waypoints)):
            self.rxarm.set_positions(self.waypoints[i], moving_time = 1.0, accel_time =0.4)
            time.sleep(1.2)
        self.rxarm.sleep()
        time.sleep(2)
        self.sorted_dict = self.camera.colored_dict
        time.sleep(1)
        self.initialize_rxarm()

    def pick_block(self, block):
        coordinates = block[0]
        orientation = block[1]
        waypoint = (coordinates[0], coordinates[1], coordinates[2] + 50)
        joint_angles_waypoint = self.rxarm.get_joint_angles(waypoint, np.pi/2, orientation*np.pi/180)
        coordinates = (coordinates[0], coordinates[1], coordinates[2] - 15)
        joint_angles = self.rxarm.get_joint_angles(coordinates, np.pi/2, orientation*np.pi/180)
        if joint_angles is not None:
            self.rxarm.set_positions(joint_angles_waypoint, moving_time=1.5)
            time.sleep(1.5)
            self.rxarm.set_positions(joint_angles, moving_time=2.5)
            time.sleep(2.5)
            self.rxarm.gripper.grasp()
            time.sleep(1)
            self.rxarm.set_positions(joint_angles_waypoint, moving_time=1.2)
            time.sleep(1)

    def place_block(self, coordinates):
        #joint_angles_waypoint = self.rxarm.get_joint_angles((coordinates[0], coordinates[1], coordinates[2] + 150), np.pi/2, 0)
        #joint_angles = self.rxarm.get_joint_angles(coordinates, np.pi/2, 0)

        joint_angles_waypoint = self.rxarm.get_joint_angles((coordinates[0], coordinates[1], coordinates[2] + 150), 0, 0)
        joint_angles = self.rxarm.get_joint_angles(coordinates, 0, 0)
        if joint_angles is not None:
            self.rxarm.set_positions(joint_angles_waypoint, moving_time=1.4)
            time.sleep(1.4)
            self.rxarm.set_positions(joint_angles, moving_time=1.2)
            time.sleep(1.5)
            self.rxarm.gripper.release()
            time.sleep(0.8)
            self.rxarm.set_positions(joint_angles_waypoint, moving_time=1)
            time.sleep(0.8)


    def task_1(self):
        """
        Task 1: Level 1 - Place 3 large blocks of color RGB in front of the arm
                Level 2 - 6 blocks, 3 of each size, random colors (ROYGBV), not stacked
                Level 3 - 9 blocks, random sizes, random colors (ROYGBV), possibly stacked two high
        in 180 seconds.
        Approach: 1. Get the colored dictionary from the camera
        2. Sort the dictionary based on the area of the blocks
        3. Get the world coordinates of the blocks.
        4. Get the depth of the blocks using depth view
        5. Use the depth and world coordinates to get the joint angles of the robot
        6. Move the robot to carry the blocks to the desired location
        7. Repeat for RGB blocks
        """

        #Level 3
        self.sweep()
        self.rxarm.initialize()
        time.sleep(1.5)
        self.task1_colors = self.camera.colors
        # loop through the colors and see if the small area blocks have x < 0 and 
        # large area blocks have x > 0, if not move them to -x positions
        self.place_coordinates_large = [(250, -25, 20), (325, -25, 20), (250, -100, 20), (325, -100, 20), (150, -25, 20), (150, -100, 20)]            
        self.place_coodinates_small = [(-250, -25, 20), (-325, -25, 20), (-250, -100, 20), (-325, -100, 20), (-150, -25, 20), (-150, -100, 20)]
        self.rxarm.sleep()
        time.sleep(1.5)
        self.sorted_dict = self.camera.sort_colored_dict_by_area()
        self.rxarm.initialize()
        time.sleep(1.5)
        self.blocks = []
        # the blocks contains ((coordinates), orientation, area, radial_distance)
        # and are then sorted based on the radial distance
        for color in self.task1_colors:
            try:
                self.sorted_dict[color]
            except KeyError:
                continue
            for block in self.sorted_dict[color]:
                if len(self.sorted_dict[color][block]) == 0:
                    continue
                coordinates = self.sorted_dict[color][block]['coords']
                orientation = self.sorted_dict[color][block]['orientation']
                area = self.sorted_dict[color][block]['area']
                radial_distance = np.sqrt(coordinates[0]**2 + coordinates[1]**2)
                self.blocks.append((coordinates, orientation, area, radial_distance))
        # sort the blocks based on their radial_distance
        self.blocks = sorted(self.blocks, key=lambda x: x[3])
        for block in self.blocks:
            self.rxarm.initialize()
            time.sleep(1)
            self.pick_block(block)
            time.sleep(1.5)
            if block[2] > 1500:
                self.place_block(self.place_coordinates_large.pop(0))
            else:
                self.place_block(self.place_coodinates_small.pop(0))

    
    def task_2(self):
        #Level 3 - 
       #self.sweep()
        #self.rxarm.initialize()
        time.sleep(1.5)
        self.task2_colors = self.camera.colors
        self.task2_coordinates = [(240, -10, 18), (240, -5, 50), (-240, -25, 20)]
        self.task2_coordinates_waypoints = [(240, -10, 105), (240,-5,100), (-240, -25, 70)]
        self.sorted_dict = self.camera.sort_colored_dict_by_area()
        done = 0
        # for big blocks
        for color in self.task2_colors:
            print("USING COLOR: ", color)
            area = 0
            try:
                coordinates = self.sorted_dict[color][0]['coords']
                # loop through all the blocks and pick the ones with area > 1500
            except KeyError:
                print("KeyError: ", color)
                continue

            for block in self.sorted_dict[color]:
                print('block: ', block)
                area = self.sorted_dict[color][block]['area']
                if area > 1500:
                    coordinates = self.sorted_dict[color][block]['coords']
                    orientation = self.sorted_dict[color][block]['orientation']
                    coordinates = (coordinates[0], coordinates[1], coordinates[2] - 20)
                    coordinates_waypoint = (coordinates[0], coordinates[1], coordinates[2] + 150)
                    joint_angles = self.rxarm.get_joint_angles(coordinates, np.pi/2, orientation*np.pi/180)
                    joint_angles_waypoint_before1 = self.rxarm.get_joint_angles(coordinates_waypoint, np.pi/2, 0)
                    joint_angles_waypoint_before2 = self.rxarm.get_joint_angles(coordinates_waypoint, np.pi/2, orientation*np.pi/180)
                    joint_angles_waypoint_after1 = self.rxarm.get_joint_angles(coordinates_waypoint, np.pi/2, orientation*np.pi/180)
                    joint_angles_waypoint_after2 = self.rxarm.get_joint_angles(coordinates_waypoint, np.pi/2, 0)
                    if joint_angles is not None:
                        self.rxarm.set_positions(joint_angles_waypoint_before1)
                        time.sleep(3)
                        self.rxarm.set_positions(joint_angles_waypoint_before2)
                        time.sleep(1)
                        self.rxarm.set_positions(joint_angles)
                        print(joint_angles)
                        time.sleep(3)
                        self.rxarm.gripper.grasp()
                        time.sleep(1)
                        self.rxarm.set_positions(joint_angles_waypoint_after1)
                        time.sleep(2)
                        self.rxarm.set_positions(joint_angles_waypoint_after2)
                        time.sleep(1)
                    joint_angles = self.rxarm.get_joint_angles(self.task2_coordinates[done], np.pi/2, 0)
                    joint_angles_waypoint = self.rxarm.get_joint_angles(self.task2_coordinates_waypoints[done], np.pi/2, 0)
                    if joint_angles is not None:
                        self.rxarm.set_positions(joint_angles_waypoint)
                        time.sleep(3)
                        self.rxarm.set_positions(joint_angles)
                        time.sleep(2)
                        self.rxarm.gripper.release()
                        time.sleep(2)
                        self.rxarm.set_positions(joint_angles_waypoint)
                        time.sleep(1)
                    done += 1
                else:
                    continue
            print("Done: ", done)
        # for small blocks
        # pick the last index of the sorted_dict
        done = 0
        self.task2_coordinates = [(250, -10, 85), (-240, -25, 50), (-240, -25, 67)]
        self.task2_coordinates_waypoints = [(250, -25, 125), (-200, -30, 125), (-250, -30, 125)]
        for color in self.task2_colors:
            print("USING SMALL COLOR: ", color)
            area = 0
            try:
                coordinates = self.sorted_dict[color][0]['coords']
                # loop through all the blocks and pick the ones with area > 1500
            except KeyError:
                print("KeyError: ", color)
                continue

            for block in self.sorted_dict[color]:
                print('block: ', block)
                area = self.sorted_dict[color][block]['area']
                if area < 1500:
                    coordinates = self.sorted_dict[color][block]['coords']
                    orientation = self.sorted_dict[color][block]['orientation']
                    print("Done: ", done)
                    coordinates = (coordinates[0], coordinates[1], coordinates[2] - 15)
                    coordinates_waypoint = (coordinates[0], coordinates[1], coordinates[2] + 100)
                    joint_angles = self.rxarm.get_joint_angles(coordinates, np.pi/2, orientation*np.pi/180)
                    joint_angles_waypoint_before1 = self.rxarm.get_joint_angles(coordinates_waypoint, np.pi/2, 0)
                    joint_angles_waypoint_before2 = self.rxarm.get_joint_angles(coordinates_waypoint, np.pi/2, orientation*np.pi/180)
                    joint_angles_waypoint_after1 = self.rxarm.get_joint_angles(coordinates_waypoint, np.pi/2, orientation*np.pi/180)
                    joint_angles_waypoint_after2 = self.rxarm.get_joint_angles(coordinates_waypoint, np.pi/2, 0)
                    if joint_angles is not None:
                        self.rxarm.set_positions(joint_angles_waypoint_before1)
                        time.sleep(3)
                        self.rxarm.set_positions(joint_angles_waypoint_before2)
                        time.sleep(3)
                        self.rxarm.set_positions(joint_angles)
                        print(joint_angles)
                        time.sleep(3)
                        self.rxarm.gripper.grasp()
                        time.sleep(3)
                        self.rxarm.set_positions(joint_angles_waypoint_after1)
                        time.sleep(3)
                        self.rxarm.set_positions(joint_angles_waypoint_after2)
                        time.sleep(3)
                    joint_angles = self.rxarm.get_joint_angles(self.task2_coordinates[done], np.pi/2, 0)
                    joint_angles_waypoint = self.rxarm.get_joint_angles(self.task2_coordinates_waypoints[done], np.pi/2, 0)
                    if joint_angles is not None:
                        self.rxarm.set_positions(joint_angles_waypoint)
                        time.sleep(3)
                        self.rxarm.set_positions(joint_angles)
                        time.sleep(3)
                        self.rxarm.gripper.release()
                        time.sleep(3)
                        self.rxarm.set_positions(joint_angles_waypoint)
                        time.sleep(3)
                        done += 1
                else:
                    continue



    def task_3(self):
        # Level 2 - arrange both large and small blocks in separate lines, each following the rainbow color order.
        # Level 3 - 12 blocks (6 small, 6 large, ROYGBV), with additional "distractor" objects.
        #   Possibly stacked but no more than four blocks high. Your objective for level 3 is to line up the small 
        #   and large blocks (cubes) as you did in level 2, but avoiding any other shaped block on the table.
        self.sweep()
        self.rxarm.initialize()
        self.rxarm.sleep()
        time.sleep(1.5)
        self.task3_coordinates_large = {'red': (-100, 275, 20), 'orange': (-50,275,20), 'yellow': (0, 275, 20), 'green': (50, 275, 20), 'blue': (100, 275, 20), 'purple': (150, 275, 20)}
        #self.task3_coordinates_small = {'red': (-100, 275, 40), 'orange': (-50,275,40), 'yellow': (0, 275, 40), 'green': (50, 275, 40), 'blue': (100, 275, 40), 'purple': (150, 275, 40)}
        self.task3_coordinates_small = {'red': (-100, 175, 8), 'orange': (-50,175,8), 'yellow': (0, 175, 8), 'green': (50, 175, 8), 'blue': (100, 175, 8), 'purple': (150, 175, 8)}
        self.task3_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        self.purple_coordinates = (0, 170, 20)
        # self.sorted_dict = self.camera.sort_colored_dict_by_area()
        #self.sorted_dict = self.camera.colored_dict
        # print("sorted_dict: ", self.sorted_dict)
        self.task3_blocks = []
        time.sleep(1.5)
        self.rxarm.initialize()
        
        placed = 0
        for color in self.task3_colors:
            try:
                self.sorted_dict[color]
            except KeyError:
                print("Keyerror", color)
                continue
            if len(self.sorted_dict[color]) == 0:
                print("sorted_dict[color] length is 0")
                continue
            for block in self.sorted_dict[color]:
                try:
                    self.sorted_dict[color][block]
                except:
                    print("Keyerror", block)
                    continue
                if len(self.sorted_dict[color][block]) == 0:
                    print("continuing the loop length = 0")
                    continue                
                # print(self.sorted_dict[color])
                coordinates = self.sorted_dict[color][block]['coords']
                orientation = self.sorted_dict[color][block]['orientation']
                area = self.sorted_dict[color][block]['area']
                radial_distance = np.sqrt(coordinates[0]**2 + coordinates[1]**2)
                print("Appending task3 blocks with: ", coordinates, orientation, area, radial_distance, color)
                self.task3_blocks.append((coordinates, orientation, area, radial_distance, color))
        self.task3_blocks = sorted(self.task3_blocks, key=lambda x: x[3])
        print("self.task3_blocks", self.task3_blocks)
        # first pick the large blocks and place them at the large coordinates
        for block in self.task3_blocks:
            self.rxarm.initialize()
            time.sleep(1)
            if block[2] > 1750:
                self.pick_block(block)
                time.sleep(1)
                self.place_block(self.task3_coordinates_large[block[4]])
                placed += 1
        
        # then pick the small blocks and place them at the small coordinates
        for block in self.task3_blocks:
            self.rxarm.initialize()
            time.sleep(1)
            if block[2] < 1750:
                self.pick_block(block)
                time.sleep(1)
                self.place_block(self.task3_coordinates_small[block[4]])
                placed += 1

    def task_4(self):
        #Level 2
        self.sweep()
        time.sleep(2)
        self.colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        # self.sorted_dict = self.camera.sort_colored_dict_by_area()
        #self.sorted_dict = self.camera.colored_dict
        time.sleep(2)
        self.task4_blocks = []
        self.depth_values_large = [25, 69, 97, 138, 180, 217, 240, 283, 324, 357, 400, 460]
        self.depth_values_small = [20, 42, 66, 87, 130, 138, 139, 145, 163, 181, 199, 217]
        # orange more x
        # yellow less x
        # green less x more z
        # blue more x more z
        # purple
        for color in self.colors:
            try:
                self.sorted_dict[color]
            except KeyError:
                print("Keyerror", color)
                continue
            if len(self.sorted_dict[color]) == 0:
                print("sorted_dict[color] length is 0")
                continue
            for block in self.sorted_dict[color]:
                try:
                    self.sorted_dict[color][block]
                except:
                    print("Keyerror", block)
                    continue
                if len(self.sorted_dict[color][block]) == 0:
                    print("continuing the loop length = 0")
                    continue                
                coordinates = self.sorted_dict[color][block]['coords']
                orientation = self.sorted_dict[color][block]['orientation']
                area = self.sorted_dict[color][block]['area']
                radial_distance = np.sqrt(coordinates[0]**2 + coordinates[1]**2)
                self.task4_blocks.append((coordinates, orientation, area, radial_distance, color))
        placed_large = 0
        placed_small = 0
        self.placing_waypoints = [-92, -38, 21, 21, -90]
        self.placing_waypoints = [angle*np.pi/180 for angle in self.placing_waypoints]
        iteration = 0
        iteration_small = 0
        for block in self.task4_blocks:
            if block[2] > 1650:
                self.pick_block(block)
                time.sleep(1)

                print(iteration)
                
                self.task4_coordinates_large = (390, 0, self.depth_values_large[placed_large])
                if (iteration == 0 or iteration == 1) :
                    self.task4_coordinates_large = (375, 0, self.depth_values_large[placed_large])

                if (iteration == 4) :
                    self.task4_coordinates_large = (380, 0, self.depth_values_large[placed_large])

                if (iteration == 5) :
                    self.task4_coordinates_large = (390, 0, self.depth_values_large[placed_large])
                
                self.rxarm.set_positions(self.placing_waypoints)
                time.sleep(1)
                self.place_block(self.task4_coordinates_large)
                time.sleep(1)
                self.rxarm.set_positions(self.placing_waypoints)
                time.sleep(1)
                self.rxarm.set_positions((0, 0, 0, 0, 0))
                placed_large += 1
                iteration = iteration + 1
            else:
                self.pick_block(block)
                time.sleep(1)
                self.task4_coordinates_small = (-350, -25, self.depth_values_small[placed_small])
                
                if(iteration_small == 0):
                    self.task4_coordinates_small = (-345, -25, self.depth_values_small[placed_small])

                if(iteration_small == 1 or iteration_small == 2):
                    self.task4_coordinates_small = (-350, -25, self.depth_values_small[placed_small])

                if(iteration_small == 3):
                    self.task4_coordinates_small = (-345, -25, self.depth_values_small[placed_small])
                
                if(iteration_small == 4):
                    self.task4_coordinates_small = (-360, -25, self.depth_values_small[placed_small])
                
                self.place_block(self.task4_coordinates_small)

                self.rxarm.set_positions((0, 0, 0, 0, 0))
                placed_small += 1
                iteration_small = iteration_small + 1
        # see if any blocks are not around the place coordinates
       #  self.sorted_dict = self.camera.sort_colored_dict_by_area()
       #  self.task4_blocks = []
       #  for color in self.colors:
       #      try:
       #          self.sorted_dict[color]
       #      except KeyError:
       #          continue
       #      for block in self.sorted_dict[color]:
       #          coordinates = self.sorted_dict[color][block]['coords']
       #          orientation = self.sorted_dict[color][block]['orientation']
       #          area = self.sorted_dict[color][block]['area']
       #          radial_distance = np.sqrt(coordinates[0]**2 + coordinates[1]**2)
       #          self.task4_blocks.append((coordinates, orientation, area, radial_distance, color))
       #  self.task4_blocks = sorted(self.task4_blocks, key=lambda x: x[3])
       #  for block in self.task4_blocks:
       #      x = block[0][0]
       #      y = block[0][1]
       #      x_range_small = (-300, -200)
       #      x_range_large = (200, 300)
       #      y_range = (-50, 50)
       #      if x < x_range_small[0] or x > x_range_small[1] or y < y_range[0] or y > y_range[1]:
       #          self.pick_block(block)
       #          time.sleep(1)
       #          self.task4_coordinates_small = (-250, -25, 20 + 19*placed_small)
       #          self.place_block(self.task4_coordinates_small)
       #          placed_small += 1
       #      if x < x_range_large[0] or x > x_range_large[1] or y < y_range[0] or y > y_range[1]:
       #          self.pick_block(block)
       #          time.sleep(1)
       #          self.task4_coordinates_large = (250, -25, 20 + 39*placed_large)
       #          self.place_block(self.task4_coordinates_large)
       #          placed_large += 1
       

    def task_bonus(self):
        # pick the block from a fixed place
        self.task_bonus_pick_coordinates = (250, -25, 20)
        self.task_bonus_pick_coordinates_waypoint = (250, -25, 150)
        # stack the blocks as high as possible
        self.task_bonus_coordinates = (0, 300, 20)
        self.task_bonus_coordinates_waypoint = (0, 300, 250)
        # sort the dict again but based on radial distance, try to pick the blocks
        # who are within the circle of radius 100 to 300 in ascending order of radial distance
        # and place them on top of each other
        self.depths = [30, 69, 107, 149, 188, 223, 246, 283, 324, 357, 400, 460]
        self.depths_new = [27, 64, 97, 135, 174, 209, 243, 280, 325, 357, 420, 460]
        joint_angles_pick = self.rxarm.get_joint_angles(self.task_bonus_pick_coordinates, np.pi/2, 0)
        joint_angles_pick_waypoint = self.rxarm.get_joint_angles(self.task_bonus_pick_coordinates_waypoint, np.pi/2, 0)
        joint_angles_waypoint = self.rxarm.get_joint_angles(self.task_bonus_coordinates_waypoint, np.pi/2, 0)
        for i in range(5):
            self.rxarm.set_positions(joint_angles_pick_waypoint)
            time.sleep(2)
            # go to the pick coordinates
            self.rxarm.set_positions(joint_angles_pick)
            time.sleep(2)
            # pick the block
            self.rxarm.gripper.grasp()
            time.sleep(2)
            # go to the place coordinates waypoint
            self.rxarm.set_positions(joint_angles_waypoint)
            time.sleep(2)
        #     # go to the place coordinates
            self.place_position = (self.task_bonus_coordinates[0], self.task_bonus_coordinates[1], self.depths[i])
            joint_angles_place = self.rxarm.get_joint_angles(self.place_position, np.pi/2, 0)
            self.rxarm.set_positions(joint_angles_place)
            time.sleep(2)
        #     # release the block
            self.rxarm.gripper.release()
            time.sleep(2)
        #     # go to the place coordinates waypoint
            self.rxarm.set_positions(joint_angles_waypoint)
            time.sleep(2)
        # # after 5 blocks for 5 to 10 blocks use another waypoint
        # #joint_angles_waypoint = (0,-23.2,-19.76,43.37,-2.64)
        # #joint_angles_waypoint = [angle*np.pi/180 for angle in joint_angles_waypoint]
        joint_angles_waypoint = (0,-20,-40,46,0)
        joint_angles_waypoint = [angle*np.pi/180 for angle in joint_angles_waypoint]
        # joint_angles_6_waypoint = (-42.5, -7, 39, -30, 0)
        # # joint_angles_6_waypoint = [angle*np.pi/180 for angle in joint_angles_6_waypoint]
        # # joint_angles_6 = (0, -29, 62.4, -32.9, 0)
        # # joint_angles_6 = [angle*np.pi/180 for angle in joint_angles_6]
        # # self.rxarm.set_positions(joint_angles_pick_waypoint)
        # # time.sleep(2)
        # # self.rxarm.set_positions(joint_angles_pick)
        # # time.sleep(2)
        # # self.rxarm.gripper.grasp()
        # # time.sleep(2)
        # # self.rxarm.set_positions(joint_angles_6_waypoint)
        # # time.sleep(2)
        # # self.rxarm.set_positions(joint_angles_6)
        # # time.sleep(2)
        # # self.rxarm.gripper.release()
        # # time.sleep(2)
        # #changing for loop for 11
        for i in range(5, 11):
            self.rxarm.set_positions(joint_angles_pick_waypoint)
            time.sleep(2)
            # go to the pick coordinates
            self.rxarm.set_positions(joint_angles_pick)
            time.sleep(2)
            # pick the block
            self.rxarm.gripper.grasp()
            time.sleep(2)
            # go to the place coordinates waypoint
            self.rxarm.set_positions(joint_angles_waypoint)
            time.sleep(2)
            # go to the place coordinates
            self.place_position = (self.task_bonus_coordinates[0], self.task_bonus_coordinates[1] - 10, self.depths[i])
            joint_angles_place = self.rxarm.get_joint_angles(self.place_position, np.pi/2, 0)
            self.rxarm.set_positions(joint_angles_place)
            time.sleep(2)
            # release the block
            self.rxarm.gripper.release()
            time.sleep(2)
            # go to the place coordinates waypoint
            self.rxarm.set_positions(joint_angles_waypoint)
            time.sleep(2)
        #adding joint positions to place the 11th block
        #add the waypoint for 11th
        self.rxarm.set_positions(joint_angles_waypoint)
        self.stack2_coordinates = (475, -35, 20)
        joint_angles_after_pick_waypoint = (-45, 0, 0, 0, 0)
        joint_angles_after_place_waypoint = (-90, 0, 0, 0, 0)
        joint_angles_after_pick_waypoint = [angle*np.pi/180 for angle in joint_angles_after_pick_waypoint]
        joint_angles_after_place_waypoint = [angle*np.pi/180 for angle in joint_angles_after_place_waypoint]
        # TODO: waypoint for 11 before and afte
        # waypoint for 1 of stack 2 before
        #changed to 5 iterations
        for i in range(1, 5):
            self.rxarm.set_positions(joint_angles_pick_waypoint)
            time.sleep(2)
            # go to the pick coordinates
            self.rxarm.set_positions(joint_angles_pick)
            time.sleep(2)
            # pick the block
            self.rxarm.gripper.grasp()
            time.sleep(2)
            self.rxarm.set_positions(joint_angles_after_pick_waypoint)
            time.sleep(2)
            # go to the place coordinates
            self.place_position = (self.stack2_coordinates[0], self.stack2_coordinates[1], self.depths_new[i])
            joint_angles_place = self.rxarm.get_joint_angles(self.place_position, np.pi/2, 0)
            self.rxarm.set_positions(joint_angles_place)
            time.sleep(2)
            # release the block
            self.rxarm.gripper.release()
            time.sleep(2)
        #     # go to the place coordinates waypoint
        self.rxarm.set_positions(joint_angles_after_place_waypoint)
        time.sleep(2)
        # pick the blocks from the stack and place them on top of the previous stack
        self.waypoint_pick_stack2_before = (-90, 4.6, 64.5, -60.6, -2.29)
        self.waypoint_pick_stack2_before = [angle*np.pi/180 for angle in self.waypoint_pick_stack2_before]
        self.rxarm.set_positions(self.waypoint_pick_stack2_before)
        time.sleep(2)
        self.waypoint_pick_stack2_coords = (490, -35, self.depths_new[1] - 5)
        self.waypoint_pick_stack2 = self.rxarm.get_joint_angles(self.waypoint_pick_stack2_coords, np.pi/2, 0)
        self.rxarm.set_positions(self.waypoint_pick_stack2, moving_time=6, accel_time = 2)
        time.sleep(8)
        self.rxarm.gripper.grasp()
        time.sleep(2)
        self.waypoint_pick_stack2_after_coords = (450, -25, self.depths_new[1] + 150)
        self.waypoint_pick_stack2_after = self.rxarm.get_joint_angles(self.waypoint_pick_stack2_after_coords, np.pi/2, 0)
        self.waypoint_pick_stack2_after = (self.waypoint_pick_stack2_after[0], self.waypoint_pick_stack2_after[1], self.waypoint_pick_stack2_after[2], self.waypoint_pick_stack2_after[3] - 10*np.pi/180, self.waypoint_pick_stack2_after[4])
        # self.waypoint_pick_stack2_after = [angle*np.pi/180 for angle in self.waypoint_pick_stack2_after]
        self.rxarm.set_positions(self.waypoint_pick_stack2_after, moving_time = 4, accel_time=2)
        time.sleep(5)
        self.rxarm.set_positions((self.waypoint_pick_stack2_after[0], 0, 0, -10*np.pi/180, 0), moving_time=4, accel_time=2)
        time.sleep(5)
        self.rxarm.set_positions((-50*np.pi/180, 0, 0, -10*np.pi/180, 0), moving_time=4, accel_time=2)
        time.sleep(5)
        # self.place_stack2_coords = (0, 300, self.depths[11])
        # self.place_stack2 = self.rxarm.get_joint_angles(self.place_stack2_coords, np.pi/2, 0)
        # self.place_stack2 = (self.place_stack2[0], self.place_stack2[1], self.place_stack2[2], -10*np.pi/180, self.place_stack2[4])
        # self.rxarm.set_positions(self.place_stack2, moving_time=6, accel_time=2)
        # time.sleep(8)
        self.place_stack2_joint_angles = (0, -15.2, -39.3, 40, -0.8)
        self.place_stack2_joint_angles = [angle*np.pi/180 for angle in self.place_stack2_joint_angles]
        self.rxarm.set_positions(self.place_stack2_joint_angles, moving_time=6, accel_time=2)
        time.sleep(8)        
        self.place_stack2_joint_angles = (0, -20.6, -27.3, 43.8, -0.8)
        self.place_stack2_joint_angles = [angle*np.pi/180 for angle in self.place_stack2_joint_angles]
        self.rxarm.set_positions(self.place_stack2_joint_angles, moving_time=6, accel_time=2)
        time.sleep(8)




    
    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"
        self.current_state = "execute"
        # assert len(self.waypoints) == len(self.gripper_pos)
        for _ in range(4):
            for i in range(len(self.waypoints)):
                self.rxarm.set_positions(self.waypoints[i])
                time.sleep(0.5)
                # i# f self.gripper_pos[i] == 0:
                #    self.rxarm.gripper.release()
                #else:
                #    self.rxarm.gripper.grasp()
        self.next_state = "idle"
        
    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""

        ## Store homography matrix
        ############################################### BEGIN
        msg = self.camera.tag_detections
        self.camera.homography_matrix = self.camera.compute_homography(msg)
        # print("homography matrix:\n", self.camera.homography_matrix)
        ############################################### END

        ## Pixel coordinates of the corners of AprilTags (with homography transformation)
        ############################################### BEGIN
        self.pixelcoordinate = []
        for i in range(4):
            for j in range(4):
                self.pixelcoordinate.append(msg.detections[i].corners[j].x)
                self.pixelcoordinate.append(msg.detections[i].corners[j].y)
        self.pixelcoordinate = np.array([self.pixelcoordinate])
        self.pixelcoordinate = self.pixelcoordinate.reshape(16, 2)
        # print("Before Homography: ", self.pixelcoordinate)
        ############################################### END
        
        ## World coordinates of the corners of AprilTags
        ############################################### BEGIN
        self.world_centers = np.array([[-250, -25], [250, -25], [250, 275], [-250, 275]])
        self.world_corners = np.zeros((16, 3))
        for i, center in enumerate(self.world_centers):
            x, y = center
            self.world_corners[i*4 + 0] = [x - 12.5, y - 12.5, 0]  # Bottom-left corner
            self.world_corners[i*4 + 1] = [x + 12.5, y - 12.5, 0]  # Bottom-right corner
            self.world_corners[i*4 + 2] = [x + 12.5, y + 12.5, 0]  # Top-right corner
            self.world_corners[i*4 + 3] = [x - 12.5, y + 12.5, 0]  # Top-left corner
        ############################################### END
        
        ## Carrying out solvePnP process
        ############################################### BEGIN
        object_points = self.world_corners.astype(np.float32)
        image_points = self.pixelcoordinate.astype(np.float32)
        # print("object_points:\n", object_points)
        # print("image_points:\n", image_points)
        intrinsic_matrix = np.array([[921.291487, 0, 650.25501], [0, 924.10414, 354.34914], [0, 0, 1]])
        # Distortion
        dist_coeffs = self.camera.dist_coeffs

        # Calculate rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, intrinsic_matrix, dist_coeffs)
        # if success:
        #     # Print rotation_vector
        #     print("Rotation Vector:\n", rotation_vector)

        #     # Print translation_vector
        #     print("Translation Vector:\n", translation_vector)
        # else:
        #     print("solvePnP failed to find a solution")
        ############################################### END
        
        ## Generating extrinsic matrix and pass it to a variable
        ######################################################## BEGIN
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Compose the extrinsic matrix (3x4 matrix)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_vector))
        last_row = np.array([[0, 0, 0, 1]])
        extrinsic_matrix = np.vstack((extrinsic_matrix, last_row))
        # Store the extrinsic matrix as needed
        self.camera.extrinsic_matrix = extrinsic_matrix
        print("EXT from solvePnP: ", self.camera.extrinsic_matrix)
        # print("extrinsic matrix:\n", self.camera.extrinsic_matrix)
        ######################################################## END

        self.camera.cameraCalibrated = True
        self.status_message = "Calibration - Completed Calibration"

    def reach_point(self, point, theta4):
        """!
        @brief      Move the rxarm to a specific point and orientation

        @param      point   The point
        @param      theta4  The theta 4
        """
        joint_angles = self.rxarm.get_joint_angles(point, theta4)
        self.rxarm.set_positions(joint_angles)      
        time.sleep(3)

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        time.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)
