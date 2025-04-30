#!/usr/bin/env python

# IMPORTS ------------------------------------------------------------------------------------------------------------

from random import randint
import time
import cv2
from eye import *
import numpy as np

# GLOBAL VARIABLES ---------------------------------------------------------------------------------------------------

CAMWIDTH = 160
CAMHEIGHT = 120
DESIRED_CAMHEIGHT = 60

# Define the world dimensions with required angle
coordinates = [
    (2781,533,1),(848,571,30),(581,705,50),(400,886,70),(295,1105,82),
    (248,2086,91),(267,3486,109),(343,3762,124),(505,4010,141),(743,4229,154),
    (1000,4371,170),(2324,4438,-180),(4190,4381,-151),(4543,4143,-120),(4714,3819,-91),
    (4733,3495,-67),(4610,3190,-45),(3971,2524,-42),(3067,1648,-36),(2733,1419,-10),
    (2305,1381,21),(1895,1581,54),(1676,1905,81),(1619,2390,92),(1638,2952,116),
    (1752,3210,132),(1981,3438,153),(2343,3600,178),(2762,3581,-153),(3648,2829,-133),
    (4657,1686,-104),(4714,1257,-70),(4552,867,-42),(4248,629,-14)
]

# EYESIM FUNCTIONS --------------------------------------------------------------------------------------------------

# Function to set the speed of the robot based on the action taken
def eyesim_set_robot_speed(linear, angular): 
    # Set the speed of the robot based on the action taken
    speed = 25
    VWSetSpeed(round(speed*linear),round(speed*angular)) # Set the speed of the robot
    time.sleep(0.25) # Wait for 0.25 seconds to allow the robot to move
    VWSetSpeed(0,0) # Stop the robot

# Function to get the image from the camera and process it
def eyesim_get_observation(): 
    # Get image from camera
    img = CAMGet() 

    # Process image
    processed_img = image_processing(img) 

    # Optional: Display the processed image on the LCD screen
    display_img = processed_img.ctypes.data_as(ctypes.POINTER(ctypes.c_byte))

    LCDImage(img)

    return img

# Function to get the distance to the red peak
def eyesim_get_position(): 
    [x,y,z,phi] = SIMGetRobot(1)
    print(x,y,z,phi)
    point = (int(x), int(y))
    result = 0
    for i in range(0, len(coordinates)-2, 2):
        polygon = np.array([
            coordinates[i][:1],
            coordinates[i+1][:1],
            coordinates[i+3][:1],
            coordinates[i+2][:1]
        ], np.int32)

        # Reshape the polygon points
        polygon = polygon.reshape((-1, 1, 2))

        # Check if the point is inside the polygon
        result = cv2.pointPolygonTest(polygon, point, False)

        # If the point is inside the polygon return
        if result > 0:
            print(f"Point {point} is inside the polygon {i}")
            break
    return result


# Function to reset the robot and can positions in the simulation
def eyesim_reset(): 
    # Stop robot movement
    VWSetSpeed(0,0)

    # Pick random position along the road to start
    random = randint(0,len(coordinates)-1)

    # Position the robot in the simulation
    x,y,phi = coordinates[random]
    SIMSetRobot(1,x,y,10,phi+180) # Add 180 degrees to the angle to flip robot into correct direction

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
    
# MAIN -------------------------------------------------------------------------------------------------------

def main():
    CAMInit(QQVGA)

    while True:
        
        LCDMenu("RESET", "DISTANCE", "-", "STOP")

        key = KEYRead()

        if key == KEY1:
            eyesim_reset()
            eyesim_get_observation()

        elif key == KEY2:
            print(f"eyesim_get_position = {eyesim_get_position()}")
            
        elif key == KEY4:
            break


main()
