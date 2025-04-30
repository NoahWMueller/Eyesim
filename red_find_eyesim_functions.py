#!/usr/bin/env python

# IMPORTS ------------------------------------------------------------------------------------------------------------

from random import randint
import time
import cv2
import math
from eye import *
import numpy as np

# GLOBAL VARIABLES ---------------------------------------------------------------------------------------------------

CAMWIDTH = 160
CAMHEIGHT = 120
DESIRED_CAMHEIGHT = 120 # TODO change back

# EYESIM FUNCTIONS --------------------------------------------------------------------------------------------------

def eyesim_set_robot_speed(direction):
    speed = 100
    VWSetSpeed(0,speed*direction)
    time.sleep(0.1) # wait for a bit to let the robot move
    VWSetSpeed(0,0) # stop from moving

def eyesim_get_observation():
    img = CAMGet() # get image from camera
    processed_img = image_processing(img)
    return processed_img

def eyesim_get_position():
    distance = find_center()
    if distance >= 80:
        distance -= 80
    elif distance != -1 and distance < 80:
        distance = 80 - distance
    return distance

def eyesim_reset():
    # stop robot movement
    # set robot and can location randomly and make sure they arent in same spot
    VWSetSpeed(0,0)
    S4_pos_x = randint(200, 1800)
    S4_pos_y = randint(200, 1800)

    CAN_pos_x = randint(200, 1800)
    CAN_pos_y = randint(200, 1800)
    
    while get_distance(S4_pos_x,S4_pos_y,CAN_pos_x,CAN_pos_y) < 500 and find_center() != -1:
        CAN_pos_x = randint(200, 1800)
        CAN_pos_y = randint(200, 1800)

    angle = math.atan2(CAN_pos_y-S4_pos_y,CAN_pos_x-S4_pos_x) # angle of the line between the two points
    angle = round(math.degrees(angle)) # convert to degrees
    if angle < 0:
        angle = 360 + angle
    angle_variation = randint(-30,30) # random angle variation
    SIMSetRobot(1,S4_pos_x,S4_pos_y,10,-angle+angle_variation)
    SIMSetObject(2,CAN_pos_x,CAN_pos_y,0,0)
    return

def get_distance(x1,y1,x2,y2):
    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distance


# IMAGE PROCESSING -------------------------------------------------------------------------------------------------------

# Function to process the image from the camera
def image_processing(image):
    # Convert the image to a numpy array and shape it to the set dimensions
    decoded_array = np.asarray(image, dtype=np.uint8)
    image_reshaped = decoded_array.reshape((CAMHEIGHT, CAMWIDTH, 3))

    # Image cropping to desired height
    middle = CAMHEIGHT//2
    lower = middle - DESIRED_CAMHEIGHT//2
    upper = middle + DESIRED_CAMHEIGHT//2
    image_reshaped = image_reshaped[lower:upper, :, :]

    # Image resizing to desired width and height

    cropped_image = cv2.resize(image_reshaped, (CAMWIDTH, DESIRED_CAMHEIGHT))
    return cropped_image

# Function to find the center of the red peak in the image
def find_center(): 
    # Get image data from the camera
    img = CAMGet()

    # convert to HSI and find index of red color peak
    [h, s, i] = IPCol2HSI(img)  
    index = colour_search(h, s, i)
    # process image
    procesesed_img = image_processing(img)
    display_img = procesesed_img.ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
    LCDImageStart(0,0,CAMWIDTH,DESIRED_CAMHEIGHT)
    LCDImage(display_img)

    # draw line where red is maximum
    LCDLine(index, 0, index, DESIRED_CAMHEIGHT-1, GREEN)

    return index

# COLOUR DETECTION -------------------------------------------------------------------------------------------------------

def colour_search(h, s, i):
    histogram = [0] * CAMWIDTH  # Initialize a histogram array for each column (0 to 159)

    # Loop over each column of the image
    index, max = -1, 0
    for x in range(CAMWIDTH):
        count = 0  # Reset the count of red pixels for each column

        # Loop over each row of the column
        for y in range(DESIRED_CAMHEIGHT):
            pos = y * CAMWIDTH + x  # Calculate the position in the 1D array
            
            # Check if the pixel matches the criteria to be considered red
            if (0 <= h[pos] < 50 or 359 > h[pos] > 345) and (i[pos] > 60 or i[pos] < -100)  and (s[pos] > 50 or s[pos] < -100):
                count += 1
                
        # Store the count of red pixels in the histogram array for this column
        histogram[x] = count
        
        # find the highest count and filter out any small patches of red
        if count > max and count > 2:
            max = count
            index = x

    return index
    
# MAIN -------------------------------------------------------------------------------------------------------

def main():
    CAMInit(QQVGA)

    while True:
        
        find_center()
        
        LCDMenu("RESET", "DISTANCE", "-", "STOP")

        key = KEYRead()

        if key == KEY1:
            eyesim_reset()

        elif key == KEY2:
            print(f"find_center = {find_center()}")
            print(f"eyesim_get_position = {eyesim_get_position()}")
            
        elif key == KEY4:
            break


main()
