#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import warnings

INTRINSIC_MATRIX = np.array([[921.2914875, 0, 650.25501], [0, 924.10414, 354.34914], [0, 0, 1]])
INV_INTRINSIC_MATRIX = np.linalg.inv(INTRINSIC_MATRIX)

class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)
        self.homography_matrix = None
        self.Inverse_homography_matrix = None

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        # self.intrinsic_matrix = np.array([[921.2914875, 0, 650.25501], [0, 924.10414, 354.34914], [0, 0, 1]])
        self.intrinsic_matrix = np.array([[900.543212890625, 0, 655.990478515625], [0, 900.89501953125, 353.4480285644531], [0, 0, 1]])
        self.extrinsic_matrix = np.eye(4)
        self.dist_coeffs = np.array([0.13974332809448242, -0.45853713154792786, -0.0008287496748380363, 0.00018046400509774685, 0.40496668219566345])
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275], [-250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])
        self.last_world_coordinates = np.array([0, 0, 0])
        self.colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        # self.colored_contours = {}
        # self.colored_coords = {}
        # self.colored_orientations = {}
        self.colored_dict = {}
        for color in self.colors:
            self.colored_dict[color] = {}

    def get_colored_dict(self):
        return self.colored_dict

    def get_depth_by_coordinates(self, x, y):
        """
        Get the depth of the block by the coordinates of the block in the image frame
        """
        return self.DepthFrameRaw[y, x]

    def compute_homography(self, msg):
        """!
        @brief      Compute a homography matrix from src to dst

        @param      src   The source points
        @param      dst   The destination points

        @return     The homography matrix
        """
        src = np.array([[msg.detections[0].centre.x, msg.detections[0].centre.y], 
                       [msg.detections[1].centre.x, msg.detections[1].centre.y],
                       [msg.detections[2].centre.x, msg.detections[2].centre.y],
                       [msg.detections[3].centre.x, msg.detections[3].centre.y]])
        # print("******SRC shape is : ", src.shape)
        # scale = 1.05
        xscale = 1280/1000
        yscale = 720/650
        dst = np.array([[250*xscale, 500*yscale], [750*xscale, 500*yscale], [750*xscale, 200*yscale], [250*xscale, 200*yscale]])
        # dst = np.array([[640 - 250*scale, 360 + 175*scale], [640 + 250*scale, 360 + 175*scale], [640 + 250*scale, 360 - 125 * scale], [640 - 250*scale, 360 - 125*scale]])
        # print("************DEBUG************")
        # print("src: ", src)
        # print("dst: ", dst)
        H = cv2.findHomography(src, dst)[0]
        # print("Shape of H : ", H.shape)
        # print("************DEBUG************")
        print("homography matrix", H)
        return H
        '''
        deltaX = int(1280 * 0.055)
        deltaY = int(720 * 0.055)
        dst = np.array([deltaX, 720 - deltaY, 1280 - deltaX, 720 - deltaY, 1280 - deltaX, deltaY, deltaX, deltaY]).reshape(4, 2)
        H = cv2.findHomography(src, dst)[0]
        return H
        '''
    

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)                
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def segment(self, image):
        """
        Segments the red blocks in the image
        INPUT:
        image - the image
        color - the color to segment
        OUTPUT:
        red_segmented - the segmented image with only the red blocks
        """
        # convert image to hsv values
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_dict = {}
        # convert to rgb values
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        '''
        #lower_red = np.array([0, 50, 50])
        lower_red = np.array([0, 0, 0])
        upper_red = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        segmented_mask_red = cv2.inRange(image, lower_red, upper_red)
        segmented_mask_red2 = cv2.inRange(image, lower_red2, upper_red2)
        segmented_mask_red = cv2.bitwise_or(segmented_mask_red, segmented_mask_red2)
        color_dict['red'] = segmented_mask_red
        
        #lower_orange = np.array([10, 80, 50])
        #upper_orange = np.array([20, 255, 255])
        lower_orange = np.array([10, 0, 0])
        upper_orange = np.array([20, 255, 255])
        segmented_mask_orange = cv2.inRange(image, lower_orange, upper_orange)
        color_dict['orange'] = segmented_mask_orange

        lower_yellow = np.array([21, 80, 150])
        #lower_yellow = np.array([20, 80, 150])
        upper_yellow = np.array([30, 255, 255])
        segmented_mask_yellow = cv2.inRange(image, lower_yellow, upper_yellow)
        color_dict['yellow'] = segmented_mask_yellow

        lower_green = np.array([30, 100, 50])
        upper_green = np.array([85, 255, 255])
        segmented_mask_green = cv2.inRange(image, lower_green, upper_green)
        color_dict['green'] = segmented_mask_green

        lower_blue = np.array([90, 100, 100])
        upper_blue = np.array([110, 255, 255])
        segmented_mask_blue = cv2.inRange(image, lower_blue, upper_blue)
        color_dict['blue'] = segmented_mask_blue

        lower_purple = np.array([110, 50, 55])
        upper_purple = np.array([170, 255, 255])
        segmented_mask_purple = cv2.inRange(image, lower_purple, upper_purple)
        color_dict['purple'] = segmented_mask_purple
        '''
        lower_red = np.array([0, 0, 0])
        upper_red = np.array([2, 255, 255])
        lower_red2 = np.array([170, 0, 0])
        upper_red2 = np.array([180, 255, 255])
        segmented_mask_red = cv2.inRange(image, lower_red, upper_red)
        segmented_mask_red2 = cv2.inRange(image, lower_red2, upper_red2)
        segmented_mask_red = cv2.bitwise_or(segmented_mask_red, segmented_mask_red2)
        color_dict['red'] = segmented_mask_red

        #lower_orange = np.array([10, 80, 50])
        #upper_orange = np.array([20, 255, 255])
        lower_orange = np.array([3, 80, 5])
        upper_orange = np.array([20, 255, 255])
        segmented_mask_orange = cv2.inRange(image, lower_orange, upper_orange)
        color_dict['orange'] = segmented_mask_orange

        lower_yellow = np.array([21, 80, 150])
        #lower_yellow = np.array([20, 80, 150])
        upper_yellow = np.array([30, 255, 255])
        segmented_mask_yellow = cv2.inRange(image, lower_yellow, upper_yellow)
        color_dict['yellow'] = segmented_mask_yellow

        lower_green = np.array([30, 100, 50])
        upper_green = np.array([85, 255, 255])
        segmented_mask_green = cv2.inRange(image, lower_green, upper_green)
        color_dict['green'] = segmented_mask_green

        lower_blue = np.array([86, 80, 80])
        upper_blue = np.array([110, 255, 255])
        segmented_mask_blue = cv2.inRange(image, lower_blue, upper_blue)
        color_dict['blue'] = segmented_mask_blue

        lower_purple = np.array([111, 50, 55])
        upper_purple = np.array([170, 255, 255])
        segmented_mask_purple = cv2.inRange(image, lower_purple, upper_purple)
        color_dict['purple'] = segmented_mask_purple
        #'pink':
        #lower = ([300, 60, 39])
        #     upper = ([350, 100, 100])
        return color_dict


    def mouse_to_world(self, x, y, z):
        world_coordinates = []
        mouse_coordinates_original = np.array([[x], [y], [1]])
        if self.cameraCalibrated == True:
            self.Inverse_homography_matrix = np.linalg.inv(self.homography_matrix)
            mouse_coordinates = np.dot(self.Inverse_homography_matrix, mouse_coordinates_original)
            mouse_coordinates = mouse_coordinates / mouse_coordinates[2]
        else:
            mouse_coordinates = mouse_coordinates_original
        z = self.DepthFrameRaw[int(mouse_coordinates[1, 0])][int(mouse_coordinates[0, 0])]               
        camera_coordinates = z * np.matmul(INV_INTRINSIC_MATRIX, mouse_coordinates)
        # print("Current camera coordinates", camera_coordinates)
        camera_coordinates = np.vstack((camera_coordinates, [1]))
        tag_extrinsic_matrix = self.extrinsic_matrix
        inv_tag_extrinsic_matrix = np.linalg.inv(tag_extrinsic_matrix)
        world_coordinates = np.matmul(inv_tag_extrinsic_matrix, camera_coordinates)
        z_calibration = (1194 * world_coordinates[0] + 2498 * world_coordinates[1] + 1560276)/149902
        world_coordinates = np.array([world_coordinates[0], world_coordinates[1], world_coordinates[2] - z_calibration])
        return world_coordinates
        


    def remove_noise(self,image):
        """
        Removes noise from a segmented image using a median filter
        INPUT:
        image - the segmented image
        OUTPUT:
        image - the segmented image with noise removed
        """
        blurred = cv2.medianBlur(image, 5)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        return thresh

    def find_contours(self, image):
        """
        Finds the contours of a segmented image
        INPUT:
        image - the segmented image
        OUTPUT:
        contours - the contours of the segmented image
        """
        # contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.findContours can return two or three values depending on the version of OpenCV. Use unpacking to support both cases.
        contours_info = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Unpack the contours from contours_info based on its length
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        # Filter just square contours or near square contours
        square_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if abs(w - h) < 60:  # Check if the bounding rectangle is square-like
                area = cv2.contourArea(contour)
                if area > 100:  # Use a threshold to filter out very small contours
                    square_contours.append(contour)
        # return just square contours or near square contours
        return square_contours
    
    def find_depth_contours(self, image, z):
        """
        Finds the contours of a segmented image
        INPUT:
        image - the segmented image
        OUTPUT:
        contours - the contours of the segmented image
        """
        # filter the depth image to include only blocks withing a certain depth
        image = cv2.inRange(image, z - 50, z + 50)
    
        contours_info = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Unpack the contours from contours_info based on its length
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        
        # Filter just square contours or near square contours
        square_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if abs(w - h) < 18:  # Check if the bounding rectangle is square-like
                area = cv2.contourArea(contour)
                if area > 100:  # Use a threshold to filter out very small contours
                    square_contours.append(contour)
        # return just square contours or near square contours
        return square_contours

    def get_correct_centroids_from_depth(self, x, y, z):
        """
        Get the correct centroids of the block from the depth
        """
        depth_image = self.DepthFrameRaw
        # find contours in the depth image
        contours = self.find_depth_contours(depth_image, z)
        # find the centroid of the blocks
        coords = self.calculate_coords(contours)
        # compare these coords with the coords given and return the correct coords
        # that are closest to the given coords
        for coord in coords:
            if abs(coord[0] - x) < 50 and abs(coord[1] - y) < 50:
                # print("Corrected coordinates vs given coordinates: ", coord, (x, y))
                return coord
        return x, y

        

    def calculate_coords(self,contours):
        """
        Calculates the coordinates of the centroids of the blobs in the image
        INPUT:
        contours - the contours of the image
        OUTPUT:
        coords - a list of tuples, where each tuple is the (x, y) coordinates of a block of that color
        """
        coords = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                coords.append((cX, cY))
        return coords

    def calculate_orientations(self, segmented, coords, contours):
        """
        Use cv2.minAreaRect() to find the orientation of the blocks
        INPUT:
        segmented - the segmented image
        coords - the coordinates of the centroids of the blobs in the image
        OUTPUT:
        orientations - a list of the orientations of the blocks
        """
        orientations = []
        for contour in contours:
            angle = cv2.minAreaRect(contour)[2]
            orientations.append(angle)
        return orientations


    def block_detector_loop(self, image):
        # image = self.VideoFrame.copy()
        # convert image to bgr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        segmented_mask = self.segment(image)
        self.colored_dict = {}
        for color in self.colors:
            self.colored_dict[color] = {}
            id = 0
            segmented_image = cv2.bitwise_and(image, image, mask=segmented_mask[color])
            noise_removed_image = self.remove_noise(segmented_image)
            noise_removed_image = cv2.cvtColor(noise_removed_image, cv2.COLOR_BGR2GRAY)
            contours = self.find_contours(noise_removed_image)
            min_area = 750
            max_area = 5000
            for contour in contours:
                self.colored_dict[color][id] = {}
                area = cv2.contourArea(contour)
                if area > min_area and area < max_area:
                    self.colored_dict[color][id]['contour'] = contour
                    self.colored_dict[color][id]['area'] = area
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cZ = int(self.get_depth_by_coordinates(cX, cY))
                        # cX_depth, cY_depth = self.get_correct_centroids_from_depth(cX, cY, cZ)
                        cX_world, cY_world, cZ_world = self.mouse_to_world(cX, cY, cZ)
                        self.colored_dict[color][id]['coords'] = (cX_world, cY_world, cZ_world)
                        self.colored_dict[color][id]['image_coords'] = (cX, cY, cZ)
                        self.colored_dict[color][id]['orientation'] = cv2.minAreaRect(contour)[2]
                    id += 1
            # self.colored_contours[color] = filtered_contours
            # coords = self.calculate_coords(contours)
            # self.colored_coords[color] = coords
            # orientations = self.calculate_orientations(segmented_image, coords, contours)
            # self.colored_orientations[color] = orientations
        
        block_coordinates = []
        for color in self.colors:
            # draw the contours on the original image 
            # and label the area of the contours and the color and orientation of the block
            # above the contour
            if len(self.colored_dict[color]) == 0:
                continue
            for id in self.colored_dict[color]:
                try:
                    self.colored_dict[color][id]
                except:
                    continue
                if len(self.colored_dict[color][id]) == 0:
                    continue
                # print(self.colored_dict[color][id])
                contour = self.colored_dict[color][id]['contour']
                area = self.colored_dict[color][id]['area']
                cX, cY, _ = self.colored_dict[color][id]['image_coords']
                cX_world, cY_world, cZ_world = self.colored_dict[color][id]['coords']
                orientation = self.colored_dict[color][id]['orientation']

                # UNCOMMENT LATER

                if(area < 1500):
                    size = "Small"
                if(area > 1500):
                    size = "large"

                cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
                # cv2.putText(image, f"Color: {color}", (cX, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # cv2.putText(image, f"{color}", (cX, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # cv2.putText(image, size, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # cv2.putText(image, f"Area: {int(area)}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # cv2.putText(image, f"Orientation: {int(orientation)}", (cX, cY + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # cv2.putText(image, f"Coordinates: [{int(cX_world)}, {int(cY_world)}, {int(cZ_world)}]", (cX, cY + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Heat map
                cv2.putText(image, f"{int(cX_world)}, {int(cY_world)}, {int(cZ_world)}", (cX-35, cY ), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
                block_coordinates.append((int(cX_world), int(cY_world), int(cZ_world)))
                
        # sort the block coordinates by x coordinates and then by y coordinates
        block_coordinates.sort(key=lambda x: (x[0], x[1]))
        templist = []
        sortedlist = []
        
        for coordinate in block_coordinates:
            # store the coordinates with similar x coordinates within -453 to -463 in templist
            if coordinate[0] > -470 and coordinate[0] < -450:
                templist.append(coordinate)
        # sort the templist by y coordinates
        templist.sort(key=lambda x: x[1])
        # append the sorted templist to the sortedlist
        sortedlist.extend(templist)
        # clear the templist
        templist.clear()
        
        for coordinate in block_coordinates:
            # store the coordinates with similar x coordinates within -403 to -413 in templist
            if coordinate[0] > -360 and coordinate[0] < -340:
                templist.append(coordinate)
        # sort the templist by y coordinates
        templist.sort(key=lambda x: x[1])
        # append the sorted templist to the sortedlist
        sortedlist.extend(templist)
        # clear the templist
        templist.clear()
        
        for coordinate in block_coordinates:
            # store the coordinates with similar x coordinates within -353 to -363 in templist
            if coordinate[0] > -260 and coordinate[0] < -240:
                templist.append(coordinate)
        # sort the templist by y coordinates
        templist.sort(key=lambda x: x[1])
        # append the sorted templist to the sortedlist
        sortedlist.extend(templist)
        # clear the templist
        templist.clear()
        
        for coordinate in block_coordinates:
            # store the coordinates with similar x coordinates within -303 to -313 in templist
            if coordinate[0] > -155 and coordinate[0] < -145:
                templist.append(coordinate)
        # sort the templist by y coordinates
        templist.sort(key=lambda x: x[1])
        # append the sorted templist to the sortedlist
        sortedlist.extend(templist)
        # clear the templist
        templist.clear()
        
        for coordinate in block_coordinates:
            # store the coordinates with similar x coordinates within -303 to -313 in templist
            if coordinate[0] > -55 and coordinate[0] < -48:
                templist.append(coordinate)
        # sort the templist by y coordinates
        templist.sort(key=lambda x: x[1])
        # append the sorted templist to the sortedlist
        sortedlist.extend(templist)
        # clear the templist
        templist.clear()
        
        for coordinate in block_coordinates:
            # store the coordinates with similar x coordinates within -303 to -313 in templist
            if coordinate[0] > 43 and coordinate[0] < 50:
                templist.append(coordinate)
        # sort the templist by y coordinates
        templist.sort(key=lambda x: x[1])
        # append the sorted templist to the sortedlist
        sortedlist.extend(templist)
        # clear the templist
        templist.clear()
        
        for coordinate in block_coordinates:
            # store the coordinates with similar x coordinates within -303 to -313 in templist
            if coordinate[0] > 140 and coordinate[0] < 150:
                templist.append(coordinate)
        # sort the templist by y coordinates
        templist.sort(key=lambda x: x[1])
        # append the sorted templist to the sortedlist
        sortedlist.extend(templist)
        # clear the templist
        templist.clear()
        
        for coordinate in block_coordinates:
            # store the coordinates with similar x coordinates within -303 to -313 in templist
            if coordinate[0] > 240 and coordinate[0] < 250:
                templist.append(coordinate)
        # sort the templist by y coordinates
        templist.sort(key=lambda x: x[1])
        # append the sorted templist to the sortedlist
        sortedlist.extend(templist)
        # clear the templist
        templist.clear()
        
        for coordinate in block_coordinates:
            # store the coordinates with similar x coordinates within -303 to -313 in templist
            if coordinate[0] > 340 and coordinate[0] < 352:
                templist.append(coordinate)
        # sort the templist by y coordinates
        templist.sort(key=lambda x: x[1])
        # append the sorted templist to the sortedlist
        sortedlist.extend(templist)
        # clear the templist
        templist.clear()
        
        for coordinate in block_coordinates:
            # store the coordinates with similar x coordinates within -303 to -313 in templist
            if coordinate[0] > 439 and coordinate[0] < 460:
                templist.append(coordinate)
        # sort the templist by y coordinates
        templist.sort(key=lambda x: x[1])
        # append the sorted templist to the sortedlist
        sortedlist.extend(templist)
        
        filename = "block_coordinates.txt"
        # Use a try-except block to handle potential errors
        try:
            # Attempt to create (and open) the file in write mode
            with open(filename, 'w') as file:
                # Assuming you have a list of coordinates to write
                for cX_world, cY_world, cZ_world in sortedlist:
                    # Write each coordinate to the file
                    file.write(f"{int(cX_world)}, {int(cY_world)}, {int(cZ_world)}\n")
            print(f"File '{filename}' created successfully.")
        except IOError as e:
            # Handle the error (e.g., print an error message)
            print(f"Error: {e.strerror}")
        
        # convert image to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            

        # min_area = 1300
        # max_area = 2200
        # filted_contours = []
        # for contour in contours:
        #     area = cv2.contourArea(contour)
        #     print("Area of contour: ", area)
        #     if area > min_area and area < max_area:
        #         filted_contours.append(contour)
        # print("Number of contours: ", len(filted_contours))
        # plot the contours on the original image
        # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        coords = self.calculate_coords(contours)
        # coords = calculate_coords(filted_contours)
        orientations = self.calculate_orientations(segmented_image, coords, contours)
        # orientations = calculate_orientations(segmented_image, coords, filted_contours)
        return image

    def sort_colored_dict_by_area(self):
        # self.sorted_dict = {}
        # for color in self.colored_dict:
        #    try:
        #        self.sorted_dict[color] = dict(sorted(self.colored_dict[color].items(), key=lambda item: item[1]['area'], reverse=True))
        #    except KeyError:
        #        continue
        return self.colored_dict

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        # self.depth_block_contours = self.find_contours(self.DepthFrameRaw)
        pass


    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        # self.grid_x_points = np.arange(-450, 500, 50)
        # self.grid_y_points = np.arange(-175, 525, 50)
        # self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))

        if self.cameraCalibrated == True:
            board_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points)).T.reshape(-1, 2)
            board_points_3D = np.column_stack((board_points, 40*np.ones(board_points.shape[0])))
            # 266 x 4
            board_points_homogenous = np.column_stack((board_points_3D, np.ones(board_points.shape[0])))

            ''' camera intrinsic matrix & inverse from camera_info ros message '''
            # Inrinsic Matrix
            K = self.intrinsic_matrix
            # Distortion
            D = self.dist_coeffs
            # P 
            P = np.column_stack((K, [0.0,0.0,0.0]))
            #print(P.shape)

            """ PnP solved Extrinsisc Matrix """
            EXT = self.extrinsic_matrix

            """ Project gridpoints to image """
            pixel_locations = np.transpose(
                np.matmul(
                    P,
                    np.divide(np.matmul(EXT, np.transpose(board_points_homogenous)), EXT[2, 3])))
            
            warped_locations = (self.homography_matrix @ pixel_locations.T).T
            for item in warped_locations:
                item[0] = item[0] / item[2]
                item[1] = item[1] / item[2]
            
            pts = warped_locations[:, :2].reshape(-1, 1, 2).astype(np.float32)
            undistorted_pts = cv2.undistortPoints(pts, K, D, None, K)
            #rgb_image = cv2.undistort(rgb_image, K, distCoeffs=D)
            img = self.VideoFrame.copy()

            for element in undistorted_pts:
                img = cv2.circle(img, (int(element[0, 0]), int(element[0, 1])), 5,
                                (255, 0, 255), -1)

            self.GridFrame = img

        else:
            pass


        # if self.cameraCalibrated == True:
        #     X, Y = self.grid_points
        #     flattened_X = X.flatten()
        #     flattened_Y = Y.flatten()
            
        #     # Combine X and Y coordinates
        #     flattened_mesh_grid = np.row_stack((flattened_X, flattened_Y))
        #     Z = np.zeros(flattened_mesh_grid.shape[1])
        #     for i in range(len(Z)):
        #         Z[i] = (1194 * flattened_X[i] + 2498 * flattened_Y[i] + 1560276)/149902
        #     ones = np.ones(flattened_mesh_grid.shape[1])
        #     flattened_mesh_grid = np.row_stack((flattened_mesh_grid, Z, ones))

        #     # [X_c Y_c Z_c 1]^T = H @ [X_w Y_w Z_w 1]^T
        #     camera_coords = np.matmul(self.extrinsic_matrix, flattened_mesh_grid)

        #     # [X_c Y_c Z_c 1] -> [X_c Y_c Z_c]
        #     camera_coords = camera_coords[:3, :]

        #     # [X_c Y_c Z_c] -> [X_c/Z_c Y_c/Z_c 1]
        #     camera_coords = camera_coords / camera_coords[2]

        #     # [u v 1] = K @ [X_c/Z_c Y_c/Z_C 1]
        #     pix_coords = np.matmul(self.intrinsic_matrix, camera_coords)

        #     # Warpped Gridpoints = homography_matrix @ [u v 1]
        #     warp_coords = np.matmul(self.homography_matrix, pix_coords)

        #     # Normalize Warpped coordinates
        #     warp_coords = warp_coords / warp_coords[2]

        #     img = self.VideoFrame.copy()
            
        #     # Drawing the circles
        #     for i in range(warp_coords.shape[1]):
        #         center = (int(warp_coords[0, i]), int(warp_coords[1, i]))
        #         cv2.circle(img, center, 5, (255, 0, 255), -1)

        #     self.GridFrame = img
        # else:
        #     pass

     
    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy()
        # Write your code here
        # Assuming 'msg.detections' is a list of detected objects
        number_of_detections = len(msg.detections)

        for i in range(number_of_detections):
            # Extract the center coordinates of each detection
            centre_x = int(msg.detections[i].centre.x)
            centre_y = int(msg.detections[i].centre.y)
            # Shift the centers by homography matrix
            if self.cameraCalibrated == True:
                centre = np.array([[centre_x], [centre_y], [1]])
                modified_centre = np.dot(self.homography_matrix, centre)
                modified_centre = modified_centre / modified_centre[2, 0]
                centre_x = int(modified_centre[0, 0])
                centre_y = int(modified_centre[1, 0])
            # Draw a circle on each detected center
            centre = (centre_x, centre_y)
            cv2.circle(modified_image, centre, 5, (0, 0, 255), -1)
            # Put the ids of the tag on the top right corner of the boxes
            cv2.putText(modified_image, "ID: " + str(msg.detections[i].id), (centre_x + 30, centre_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Draw lines according to the contour of the ArpilTag
            corner = np.zeros((3, 4))
            for j in range(4):
                corner[0, j] = int(msg.detections[i].corners[j].x)
                corner[1, j] = int(msg.detections[i].corners[j].y)
                corner[2, j] = 1
            # Shift the corners by homography matrix
            if self.cameraCalibrated == True:
                corner = self.homography_matrix @ corner
                for k in range(4):
                    corner[:, k] = corner[:, k] / corner[2, k]

            cv2.line(modified_image, (int(corner[0, 0]), int(corner[1, 0])), (int(corner[0, 1]), int(corner[1, 1])), (0, 255, 0), 2)
            cv2.line(modified_image, (int(corner[0, 1]), int(corner[1, 1])), (int(corner[0, 2]), int(corner[1, 2])), (0, 255, 0), 2)
            cv2.line(modified_image, (int(corner[0, 2]), int(corner[1, 2])), (int(corner[0, 3]), int(corner[1, 3])), (0, 255, 0), 2)
            cv2.line(modified_image, (int(corner[0, 3]), int(corner[1, 3])), (int(corner[0, 0]), int(corner[1, 0])), (0, 255, 0), 2)
            
        self.TagImageFrame = modified_image


class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        if self.camera.cameraCalibrated == True:
            cv_image = cv2.warpPerspective(cv_image, self.camera.homography_matrix, (cv_image.shape[1], cv_image.shape[0]))
            cv2.imwrite("./segmented_image.png", cv_image)
            cv_image = self.camera.block_detector_loop(cv_image)
        self.camera.VideoFrame = cv_image
        

class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)
            


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        # if self.camera.cameraCalibrated == True:
        #     cv_depth = cv2.warpPerspective(cv_depth, self.camera.homography_matrix, (cv_depth.shape[1], cv_depth.shape[0]))
        self.camera.DepthFrameRaw = cv_depth
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)
                # print("The name of this file is: ", __name__)

                if __name__ == '__main__':
                # if __name__ == 'camera':
                    print("I am about to show the image window")
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()