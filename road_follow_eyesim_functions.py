#!/usr/bin/env python

# IMPORTS ------------------------------------------------------------------------------------------------------------

from random import randint
import cv2
from eye import *
import numpy as np

# GLOBAL VARIABLES ---------------------------------------------------------------------------------------------------

CAMWIDTH = 160
CAMHEIGHT = 120
DESIRED_CAMHEIGHT = 120

# Left lane coordinates for the simulation environment
left_lane = [
    (3990,400),(4000,733),(3562,400),(3571,733),(3171,400),
    (3171,733),(2838,400),(2848,733),(2448,400),(2448,733),
    (2095,410),(2095,733),(1676,400),(1657,743),(1267,410),
    (1238,733),(943,410),(962,733),(590,543),(771,790),
    (352,714),(600,943),(190,1010),(505,1057),(114,1229),
    (438,1248),(114,1619),(438,1619),(114,1971),(438,2000),
    (114,2362),(438,2371),(114,2705),(438,2733),(114,3086),
    (438,3086),(114,3381),(429,3381),(181,3705),(467,3590),
    (305,3981),(571,3829),(552,4257),(771,4019),(819,4438),
    (981,4171),(1171,4562),(1190,4248),(1619,4562),(1619,4248),
    (2000,4571),(2010,4248),(2410,4581),(2410,4267),(2762,4590),
    (2762,4267),(3190,4581),(3190,4257),(3619,4581),(3619,4257),
    (4029,4581),(4019,4267),(4524,4362),(4333,4133),(4800,4000),
    (4514,3895),(4876,3619),(4562,3610),(4790,3248),(4524,3390),
    (4581,2943),(4362,3162),(4333,2705),(4105,2905),(4124,2552),
    (3933,2733),(3895,2314),(3667,2524),(3676,2086),(3438,2305),
    (3438,1867),(3219,2076),(3257,1686),(3057,1905),(2981,1390),
    (2743,1610),(2476,1229),(2476,1514),(1943,1352),(2114,1619),
    (1590,1743),(1876,1867),(1486,2067),(1810,2105),(1476,2476),
    (1800,2476),(1476,2867),(1800,2857),(1581,3171),(1838,3029),
    (1743,3448),(2000,3238),(2124,3695),(2267,3400),(2590,3771),
    (2600,3448),(3095,3581),(2895,3343),(3371,3314),(3143,3086),
    (3705,2981),(3457,2762),(3933,2733),(3667,2524),(4124,2524),
    (3905,2305),(4390,2257),(4152,2048),(4695,1933),(4448,1724),
    (4876,1457),(4562,1400),(4762,924),(4486,1086),(4486,600),
    (4267,819),(3990,400),(4000,733)
]

# Centroids of the polygons for the simulation environment
centroids = [
    (3829,533,7),(3410,533,8),(3048,533,9),(2686,533,8),(2314,533,10),
    (1924,533,8),(1505,533,9),(1143,533,11),(848,571,30),(581,705,50),
    (400,886,70),(295,1105,82),(248,1381,97),(248,1762,97),(248,2133,97),
    (248,2505,98),(248,2857,97),(248,3200,97),(267,3486,109),(343,3762,124),
    (505,4010,141),(743,4229,154),(1000,4371,170),(1362,4429,-175),(1781,4438,-173),
    (2171,4448,-174),(2552,4457,-172),(2933,4457,-172),(3362,4448,-174),(3781,4448,-174),
    (4190,4381,-151),(4543,4143,-120),(4714,3819,-91),(4733,3495,-67),(4610,3190,-45),
    (4400,2924,-33),(4171,2714,-27),(3952,2533,-34),(3714,2305,-35),(3495,2076,-32),
    (3286,1876,-32),(3067,1648,-36),(2733,1419,-10),(2305,1381,21),(1895,1581,54),
    (1676,1905,81),(1619,2238,95),(1610,2629,97),(1638,2952,116),(1752,3210,132),
    (1981,3438,153),(2343,3600,178),(2762,3581,-153),(3124,3381,-126),(3410,3086,-129),
    (3695,2790,-123),(3914,2562,-124),(4143,2324,-127),(4419,2038,-126),(4657,1686,-104),
    (4714,1257,-70),(4552,867,-42),(4248,629,-14)
]


# Current polygon where the robot was placed
current_centroid = randint(0,len(centroids))
current_polygon = np.array([])
next_polygon = np.array([])

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

def update_polygon(current_centroid):
    # Update the current and next polygon based on the current centroid
    current_polygon = np.array([
        left_lane[current_centroid*2],
        left_lane[current_centroid*2+1],
        left_lane[(current_centroid*2+3)],
        left_lane[(current_centroid*2+2)]
    ], np.int32)

    next_centroid = (current_centroid + 1)% len(centroids)

    next_polygon = np.array([
        left_lane[next_centroid*2],
        left_lane[next_centroid*2+1],
        left_lane[(next_centroid*2+3)],
        left_lane[(next_centroid*2+2)]
    ], np.int32)

    return current_polygon, next_polygon

# Function to get the distance to the red peak
def eyesim_get_position(): 
    [x,y,_,_] = SIMGetRobot(1)
    point = (x.value, y.value)
    result1,result2 = -1,-1
    
    current_polygon, next_polygon = update_polygon(current_centroid)

    # Reshape the polygon points
    current_polygon = current_polygon.reshape((-1, 1, 2))

    # Reshape the polygon points
    next_polygon = next_polygon.reshape((-1, 1, 2))

    # Check if the point is inside the polygon
    result1 = cv2.pointPolygonTest(current_polygon, point, False)

    # Check if the point is inside the polygon
    result2 = cv2.pointPolygonTest(next_polygon, point, False)

    # If the point is inside the polygon return
    return result1,result2

# Function to reset the robot and can positions in the simulation
def eyesim_reset(): 
    global current_centroid
    # Stop robot movement
    VWSetSpeed(0,0)

    current_centroid = current_centroid%len(centroids)-1

    
    # # Position the robot in the simulation
    x,y,phi = centroids[current_centroid]

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
