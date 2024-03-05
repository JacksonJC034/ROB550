# code for block detection


import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../launch/segmented_image.png')

def detect_blocks(image):
    """
    Detects the red, orange, yellow, green, blue, purple, and pink blocks and 
    returns their coordinates in the image.
    STEPS:
    1. Segment the image based on color or depth value
    2. Remove noise from segmented images
    3. Find contours of the segments
    4. Calculate the moments for the contours to find the centroids of
    the blobs in the image
    5. Use cv2.minAreaRect() to find orientation of block
    6. Use centroid, orientation, depth value, inverse intrinsic matrix &
    inverse extrinsic matrix to find locations in workspace
    INPUT:
    image - the image to detect the blocks in
    OUTPUT:
    block_coords - a dictionary with the following keys: 'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink'
                   Each key maps to a list of tuples, where each tuple is the (x, y) coordinates of a block of that color
    """
    # segment the image based on color
    red_segmented = segment(image, color='red')
    orange_segmented = segment(image, color='orange')
    yellow_segmented = segment(image, color='yellow')
    green_segmented = segment(image, color='green')
    blue_segmented = segment(image, color='blue')
    purple_segmented = segment(image, color='purple')
    pink_segmented = segment(image, color='pink')

    # remove noise from segmented images
    red_segmented = remove_noise(red_segmented)
    orange_segmented = remove_noise(orange_segmented)
    yellow_segmented = remove_noise(yellow_segmented)
    green_segmented = remove_noise(green_segmented)
    blue_segmented = remove_noise(blue_segmented)
    purple_segmented = remove_noise(purple_segmented)
    pink_segmented = remove_noise(pink_segmented)

    # find contours of the segments
    red_contours = find_contours(red_segmented)
    orange_contours = find_contours(orange_segmented)
    yellow_contours = find_contours(yellow_segmented)
    green_contours = find_contours(green_segmented)
    blue_contours = find_contours(blue_segmented)
    purple_contours = find_contours(purple_segmented)
    pink_contours = find_contours(pink_segmented)

    # calculate the moments for the contours to find the centroids of the blobs in the image
    red_coords = calculate_coords(red_contours)
    orange_coords = calculate_coords(orange_contours)
    yellow_coords = calculate_coords(yellow_contours)
    green_coords = calculate_coords(green_contours)
    blue_coords = calculate_coords(blue_contours)
    purple_coords = calculate_coords(purple_contours)
    pink_coords = calculate_coords(pink_contours)

    # calculate the orientation of the blocks
    red_orientations = calculate_orientations(red_segmented, red_coords)
    orange_orientations = calculate_orientations(orange_segmented, orange_coords)
    yellow_orientations = calculate_orientations(yellow_segmented, yellow_coords)
    green_orientations = calculate_orientations(green_segmented, green_coords)
    blue_orientations = calculate_orientations(blue_segmented, blue_coords)
    purple_orientations = calculate_orientations(purple_segmented, purple_coords)
    pink_orientations = calculate_orientations(pink_segmented, pink_coords)

    


def remove_noise(image):
    """
    Removes noise from a segmented image using a median filter
    INPUT:
    image - the segmented image
    OUTPUT:
    image - the segmented image with noise removed
    """
    return cv2.medianBlur(image, 5)

def find_contours(image):
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
        if abs(w - h) < 18:  # Check if the bounding rectangle is square-like
            area = cv2.contourArea(contour)
            if area > 100:  # Use a threshold to filter out very small contours
                square_contours.append(contour)
    # return just square contours or near square contours
    return square_contours

def segment(image):
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
        # lower and upper values of red in hsv
    # red: done
    # blue: done
    # orange: wrong
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([9, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    segmented_mask_red = cv2.inRange(image, lower_red, upper_red)
    segmented_mask_red2 = cv2.inRange(image, lower_red2, upper_red2)
    segmented_mask_red = cv2.bitwise_or(segmented_mask_red, segmented_mask_red2)
    color_dict['red'] = segmented_mask_red
    lower_orange = np.array([9, 80, 50])
    upper_orange = np.array([20, 255, 255])
    segmented_mask_orange = cv2.inRange(image, lower_orange, upper_orange)
    color_dict['orange'] = segmented_mask_orange
    lower_yellow = np.array([20, 80, 150])
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
    lower_purple = np.array([110, 30, 55])
    upper_purple = np.array([170, 255, 255])
    segmented_mask_purple = cv2.inRange(image, lower_purple, upper_purple)
    color_dict['purple'] = segmented_mask_purple
    
    #'pink':
    #lower = ([300, 60, 39])
    #     upper = ([350, 100, 100])
    return color_dict


def calculate_coords(contours):
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

def calculate_orientations(segmented, coords, contours):
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

def calculate_workspace_coords(coords, orientations, depth_values, inverse_intrinsic_matrix, inverse_extrinsic_matrix):
    """
    Use centroid, orientation, depth value, inverse intrinsic matrix & inverse extrinsic matrix to find locations in workspace
    INPUT:
    coords - the coordinates of the centroids of the blobs in the image
    orientations - a list of the orientations of the blocks
    depth_values - a list of the depth values of the blocks
    inverse_intrinsic_matrix - the inverse intrinsic matrix
    inverse_extrinsic_matrix - the inverse extrinsic matrix
    OUTPUT:
    workspace_coords - a list of the coordinates of the centroids of the blobs in the workspace
    """
    workspace_coords = []
    for i in range(len(coords)):
        x, y = coords[i]
        depth = depth_values[i]
        orientation = orientations[i]
        # calculate the location in the workspace
        workspace_coords.append((x, y, depth))
    return workspace_coords


if __name__=="__main__":
    # print(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    segmented_mask = segment(image)
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    colored_contours = {}
    colored_coords = {}
    colored_orientations = {}
    for color in colors:
        segmented_image = cv2.bitwise_and(image, image, mask=segmented_mask[color])
        noise_removed_image = remove_noise(segmented_image)
        noise_removed_image = cv2.cvtColor(noise_removed_image, cv2.COLOR_BGR2GRAY)
        contours = find_contours(noise_removed_image)
        min_area = 750
        max_area = 3000
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area and area < max_area:
                filtered_contours.append(contour)
        colored_contours[color] = filtered_contours
        coords = calculate_coords(contours)
        colored_coords[color] = coords
        orientations = calculate_orientations(segmented_image, coords, contours)
        colored_orientations[color] = orientations
    
    for color in colors:
        # draw the contours on the original image 
        # and label the area of the contours and the color and orientation of the block
        # above the contour
        for contour in colored_contours[color]:
            area = cv2.contourArea(contour)
            print("Area of contour: ", area)
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(image, color, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(image, str(area), (cX, cY+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(image, str(colored_orientations[color][0]), (cX, cY+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    plt.imshow(image)
    plt.show()
    coords = calculate_coords(contours)
    # coords = calculate_coords(filted_contours)
    #print(coords)
    orientations = calculate_orientations(segmented_image, coords, contours)
    # orientations = calculate_orientations(segmented_image, coords, filted_contours)
    #print(orientations)




