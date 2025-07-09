# IMPORTS ------------------------------------------------------------------------------------------------------------

import os
import re
import cv2
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