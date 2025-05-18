#!/usr/bin/env python

# IMPORTS ------------------------------------------------------------------------------------------------------------

from random import randint
import cv2
from eye import *
import numpy as np

# GLOBAL VARIABLES ---------------------------------------------------------------------------------------------------

CAMWIDTH = 160
CAMHEIGHT = 120
DESIRED_CAMHEIGHT = 60

# define the world coordinates for the left lane
flipped_left_lane_world_coordinates = [
    (3990,400),(4000,733),(943,410),(962,733),(590,543),
    (771,790),(352,714),(600,943),(190,1010),(505,1057),
    (114,1229),(438,1248),(114,3381),(429,3381),(181,3705),
    (467,3590),(305,3981),(571,3829),(552,4257),(771,4019),
    (819,4438),(981,4171),(1171,4562),(1190,4248),(4029,4581),
    (4019,4267),(4524,4362),(4333,4133),(4800,4000),(4514,3895),
    (4876,3619),(4562,3610),(4790,3248),(4524,3390),(4581,2943),
    (4362,3162),(3257,1686),(3057,1905),(2981,1390),(2743,1610),
    (2476,1229),(2476,1514),(1943,1352),(2114,1619),(1590,1743),
    (1876,1867),(1486,2067),(1810,2105),(1476,2867),(1800,2857),
    (1581,3171),(1838,3029),(1743,3448),(2000,3238),(2124,3695),
    (2267,3400),(2590,3771),(2600,3448),(3095,3581),(2895,3343),
    (4695,1933),(4448,1724),(4876,1457),(4562,1400),(4762,924),
    (4486,1086),(4486,600),(4267,819),(3990,400),(4000,733)
]

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


# Current polygon where the robot was placed
current_polygon = randint(0,len(coordinates))


# EYESIM FUNCTIONS --------------------------------------------------------------------------------------------------

# Function to get the image from the camera and process it
def eyesim_get_observation(): 
    # Get image from camera
    img = CAMGet() 

    # Process image
    processed_img = image_processing(img) 

    # Optional: Display the processed image on the LCD screen
    display_img = processed_img.ctypes.data_as(ctypes.POINTER(ctypes.c_byte))

    LCDImage(display_img)

    return processed_img

# Function to get the distance to the red peak
def eyesim_get_position(): 
    [x,y,_,_] = SIMGetRobot(1)
    point = (x.value, y.value)
    result1,result2 = -1,-1
    
    polygon = np.array([
        flipped_left_lane_world_coordinates[current_polygon*2],
        flipped_left_lane_world_coordinates[current_polygon*2+1],
        flipped_left_lane_world_coordinates[(current_polygon*2+3)%68],
        flipped_left_lane_world_coordinates[(current_polygon*2+2)%68]
    ], np.int32)

    next_polygon = current_polygon + 1

    polygon_2 = np.array([
        flipped_left_lane_world_coordinates[next_polygon*2],
        flipped_left_lane_world_coordinates[next_polygon*2+1],
        flipped_left_lane_world_coordinates[(next_polygon*2+3)%68],
        flipped_left_lane_world_coordinates[(next_polygon*2+2)%68]
    ], np.int32)

    # Reshape the polygon points
    polygon = polygon.reshape((-1, 1, 2))

    # Reshape the polygon points
    polygon_2 = polygon_2.reshape((-1, 1, 2))

    # Check if the point is inside the polygon
    result1 = cv2.pointPolygonTest(polygon, point, False)

    # Check if the point is inside the polygon
    result2 = cv2.pointPolygonTest(polygon_2, point, False)

    # If the point is inside the polygon return
    return result1,result2

# Function to reset the robot and can positions in the simulation
def eyesim_reset(): 
    global current_polygon
    # Stop robot movement
    VWSetSpeed(0,0)

    current_polygon = current_polygon%len(coordinates)-1

    
    # # Position the robot in the simulation
    x,y,phi = coordinates[current_polygon]

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
        LCDImageStart(0,0,CAMWIDTH,DESIRED_CAMHEIGHT)
        LCDMenu("RESET", "POSITION", "-", "STOP")

        key = KEYRead()

        if key == KEY1:
            eyesim_reset()
            eyesim_get_observation()

        elif key == KEY2:
            print(f"eyesim_get_position = {eyesim_get_position()}")
            
        elif key == KEY4:
            break


main()
