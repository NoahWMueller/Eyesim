# IMPORTS ------------------------------------------------------------------------------------------------------------

import os
import re
import cv2
import ast
from eye import *
import numpy as np

# GLOBAL VARIABLES ---------------------------------------------------------------------------------------------------

# Constants for camera settings
CAMWIDTH = 160
CAMHEIGHT = 120

# FILE HANDLING -------------------------------------------------------------------------------------------------------

# Function to find the latest model in the models directory
def find_latest_model(models_dir):
    previous_models = os.listdir(models_dir)

    # Filter only files matching pattern like model_123.zip
    model_files = [f for f in previous_models if re.match(r"model_\d+\.zip", f)]

    # Extract the number and find the model with the highest number
    def extract_model_number(filename):
        match = re.search(r"model_(\d+)\.zip", filename)
        return int(match.group(1)) if match else -1

    # Sort models by number
    model_files.sort(key=extract_model_number)

    # Get the most recent model
    most_recent_model = model_files[-1] if model_files else "None"

    if most_recent_model == "None":
        print("No pre-trained model found.")
        return None
    
    iteration = (int(most_recent_model.split("_")[1].split(".")[0]) + 1) if most_recent_model else 0

    # If no pre-trained model is found, print a message and return
    if iteration == 0:
        print("No pre-trained model found.")
        return None
    
    return most_recent_model, iteration

# Function to load map points from a file
def load_map_points(file_path):
    map_points = []
    with open(file_path, 'r') as file:
        for line in file:
            # Convert string "(x, y, phi)" safely into a tuple
            map_points.append(ast.literal_eval(line.strip()))
    return map_points

# IMAGE PROCESSING -------------------------------------------------------------------------------------------------------

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

# EYESIM ------------------------------------------------------------------------------------------------------------------

StopSign1 = [0, 0, 0]  # Placeholder for StopSign1 coordinates
StopSign2 = [0, 0, 0]  # Placeholder for StopSign2 coordinates
SpeedLimit10Sign1 = [0, 0, 0]  # Placeholder for SpeedLimit10Sign1 coordinates
SpeedLimit10Sign2 = [0, 0, 0]  # Placeholder for SpeedLimit10Sign2 coordinates
SpeedLimitSign1 = [0, 0, 0]  # Placeholder for SpeedLimitSign1 coordinates
SpeedLimitSign2 = [0, 0, 0]  # Placeholder for SpeedLimitSign2 coordinates

# Function to check if the objects are in the correct position and set them if not
def object_check():
    if SIMGetObject(2)[0].value != StopSign1[0] or SIMGetObject(2)[1].value != StopSign1[1] or SIMGetObject(2)[2].value != StopSign1[2]: 
        SIMSetObject(2, StopSign1[0], StopSign1[1], 10, StopSign1[2]+90)
    if SIMGetObject(3)[0].value != StopSign2[0] or SIMGetObject(3)[1].value != StopSign2[1] or SIMGetObject(3)[2].value != StopSign2[2]: 
        SIMSetObject(3, StopSign2[0], StopSign2[1], 10, StopSign2[2]+90)
    if SIMGetObject(4)[0].value != SpeedLimit10Sign1[0] or SIMGetObject(4)[1].value != SpeedLimit10Sign1[1] or SIMGetObject(4)[2].value != SpeedLimit10Sign1[2]: 
        SIMSetObject(4, SpeedLimit10Sign1[0], SpeedLimit10Sign1[1], 10, SpeedLimit10Sign1[2])
    if SIMGetObject(5)[0].value != SpeedLimit10Sign2[0] or SIMGetObject(5)[1].value != SpeedLimit10Sign2[1] or SIMGetObject(5)[2].value != SpeedLimit10Sign2[2]: 
        SIMSetObject(5, SpeedLimit10Sign2[0], SpeedLimit10Sign2[1], 10, SpeedLimit10Sign2[2])
    if SIMGetObject(6)[0].value != SpeedLimitSign1[0] or SIMGetObject(6)[1].value != SpeedLimitSign1[1] or SIMGetObject(6)[2].value != SpeedLimitSign1[2]: 
        SIMSetObject(6, SpeedLimitSign1[0], SpeedLimitSign1[1], 10, SpeedLimitSign1[2])
    if SIMGetObject(7)[0].value != SpeedLimitSign2[0] or SIMGetObject(7)[1].value != SpeedLimitSign2[1] or SIMGetObject(7)[2].value != SpeedLimitSign2[2]: 
        SIMSetObject(7, SpeedLimitSign2[0], SpeedLimitSign2[1], 10, SpeedLimitSign2[2])