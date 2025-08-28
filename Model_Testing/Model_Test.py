#!/usr/bin/env python

import os
from eye import *
import numpy as np
import cv2
import ctypes
from stable_baselines3 import PPO

Linear_model_file = 'Linear_model.zip'
Angular_model_file = 'Angular_model.zip'

# Constants for camera settings
CAM_SETTING = QQVGA
CAMWIDTH = QQVGA_X
CAMHEIGHT = QQVGA_Y

def test_models(Linear = True, Angular = True):
    if Linear:
        if os.path.exists(Linear_model_file): 
            print(f"Loading {Linear_model_file}")
            Linear_model = PPO.load(Linear_model_file)
        else: 
            print(f"File {Linear_model_file} does not exist, cannot proceed with testing.")
            return
    if Angular: 
        if os.path.exists(Angular_model_file): 
            print(f"loading {Angular_model_file}")
            Angular_model = PPO.load(Angular_model_file)
        else: 
            print(f"File {Angular_model_file} does not exist, cannot proceed with testing.")
            return

    # Continue testing the loaded model until the user decides to stop
    while True:
        LCDMenu("-", "-", "-", "Stop")
        LCDSetPrintf(0,60,"Testing Models")
        key = KEYRead()

        obs = eyesim_get_observation()
        
        # Predict the action using the loaded model
        if Linear: 
            Linear_action, _ = Linear_model.predict(obs)
            Linear_action = Linear_action[0]
        else: Linear_action = 0.5
        if Angular: 
            Angular_action, _ = Angular_model.predict(obs)
            Angular_action = Angular_action[0]
        else: Angular_action = 0

        base_linear = 200
        base_angular = 100

        LCDSetPrintf(4,60,f"Linear: {round(Linear_action*base_linear)}    ")
        LCDSetPrintf(6,60,f"Angular: {round(Angular_action*base_angular)}    ")

        VWSetSpeed(round(Linear_action*base_linear), round(Angular_action*base_angular))
        
        # End testing if user presses the stop key
        if key == KEY4: 
            VWSetSpeed(0,0)
            LCDSetPrintf(0,60,"Testing Stopped")
            LCDSetPrintf(4,60,"Linear: 0    ")
            LCDSetPrintf(6,60,"Angular: 0    ")
            break

def eyesim_get_observation(): 
    # Get image from camera
    img = CAMGet() 

    # Process image
    processed_img = image_processing(img) 

    # Optional: Display the processed image on the LCD screen
    display_img = processed_img.ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
    LCDImage(display_img)

    return processed_img

# Function to process the image from the camera
def image_processing(image):
    # Convert the image to a numpy array and shape it to the set dimensions
    decoded_array = np.asarray(image, dtype=np.uint8)
    image_reshaped = decoded_array.reshape((CAMHEIGHT, CAMWIDTH, 3))

    # Image cropping to desired height
    middle = CAMHEIGHT//2
    lower = middle - CAMHEIGHT//2
    upper = middle + CAMHEIGHT//2
    image_reshaped = image_reshaped[lower:upper, :, :]

    # Image resizing to desired width and height
    cropped_image = cv2.resize(image_reshaped, (CAMWIDTH, CAMHEIGHT))

    return cropped_image

def main():
    # Initialize the camera with QQVGA resolution (160x120)
    CAMInit(CAM_SETTING) 
    LCDImageStart(0,0,CAMWIDTH,CAMHEIGHT)

    LCDSetPrintf(0,60,"Program Start")
    Linear = False
    LCDSetPrintf(2,60,"Linear Model OFF")
    while True:
        LCDMenu("Test", "Toggle Linear", "-", "Quit")
        key = KEYRead()
        # Test the models
        if key == KEY1: 
            test_models(Linear, Angular = True)

        if key == KEY2: 
            if Linear: 
                Linear = False
                LCDSetPrintf(2,60,"Linear Model OFF")
            else: 
                Linear = True
                LCDSetPrintf(2,60,"Linear Model ON ")
                
        # Quit program
        elif key == KEY4:
            break
        
main()