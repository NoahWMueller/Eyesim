#!/usr/bin/env python3
from eye import *
import numpy as np
import cv2

CAMHEIGHT = QVGA_Y
CAMWIDTH = QVGA_X
DESIRED_HEIGHT = CAMHEIGHT
threshold = 60
CAMInit(QVGA)
LCDImageStart(0,0,CAMWIDTH,DESIRED_HEIGHT)

""" LINEAR 
THRESHOLD = 80 for stop sign
THRESHOLD = 110 for speed signs
DESIRED_HEIGHT = CAMHEIGHT
"""
"""ANGULAR
THRESHOLD = 60
DESIRED_HEIGHT = CAMHEIGHT // 2
black walls
"""

def process_camera_image():
    # Get the grayscale image
    gray_image_raw = CAMGetGray()
    
    # Convert the image to a NumPy array
    gray_image_np = np.asarray(gray_image_raw, dtype=np.uint8).reshape((CAMHEIGHT, CAMWIDTH))

    # Apply thresholding to convert to a binary 
    binary_image = np.where(gray_image_np > threshold, 255, 0).astype(np.uint8)

    # Crop the image to the desired height
    cropped_image = binary_image[DESIRED_HEIGHT:, :]
    
    # Convert the cropped image to a ctypes pointer for LCD display
    c_type_pointer = cropped_image.ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
    
    # Display the final binary image
    LCDImageBinary(c_type_pointer)

while True:
    LCDMenu("TOP", "BOTTOM", "THRESHOLD", "END")
    key = KEYRead()
    process_camera_image()
    if key == KEY1:
        DESIRED_HEIGHT = CAMHEIGHT
        LCDImageStart(0,0,CAMWIDTH,DESIRED_HEIGHT)
        LCDClear()
    elif key == KEY2:
        DESIRED_HEIGHT = (CAMHEIGHT // 2)
        LCDImageStart(0,0,CAMWIDTH,DESIRED_HEIGHT)
        LCDClear()
    elif key == KEY3:
        while True:
            LCDMenu("UP", "DOWN", "SET", "EXIT")
            print(threshold)
            key = KEYRead()
            if key == KEY1:
                threshold = min(255, threshold + 1)
            elif key == KEY2:
                threshold = max(0, threshold - 1)
            elif key == KEY4:
                break
            process_camera_image()
    elif key == KEY4:
        break