#!/usr/bin/env python

# IMPORTS ------------------------------------------------------------------------------------------------------------

import os
from pydoc import cli
import cv2
import math
import time
from eye import *
import numpy as np
import gymnasium as gym
from random import randint
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# GLOBAL VARIABLES ---------------------------------------------------------------------------------------------------

# Constants for camera settings
CAMWIDTH = 160
CAMHEIGHT = 120
DESIRED_CAMHEIGHT = 60

# Algorithm used for training
algorithm = "PPO" 

# Current version of the code for saving models and logs
version = 1.2 

# Directory paths for saving models and logs
models_dir = f"models/{algorithm}/{version}"
logdir = f"logs/{algorithm}/{version}"

# ENVIRONMENT --------------------------------------------------------------------------------------------------------

# Custom environment for the robot simulation using OpenAI Gymnasium
class EyeSimEnv(gym.Env):
    
    def __init__(self):
        super(EyeSimEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,)) # Float action space for robot speed, range from -2 to 2
        self.observation_space = spaces.Box(low=0, high=255, shape=(DESIRED_CAMHEIGHT,CAMWIDTH,3), dtype=np.uint8) # Image observation space, 3 channels (RGB), 60x160 pixels

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        eyesim_reset()
        observation = eyesim_get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # Distance to red peak before action
        init_position = eyesim_get_position() 
        
        # Set robot speed based on action
        eyesim_set_robot_speed(round(float(action[0]),2)) 
        
        # Distance to red peak after action
        after_position = eyesim_get_position() 

        # Read image from camera
        observation = eyesim_get_observation() 

        # Calculate reward based on position or sensor readings
        reward = self.calculate_reward(init_position,after_position)

        # Determine if the episode is done
        done = self.is_done(after_position)

        # Truncated is not used in this case, but included for compatibility with gym API
        truncated = False
        info = {}

        return observation, reward, done, truncated, info

    def calculate_reward(self, init_position, after_position):

        # scaling factor to increase reward as the robot gets closer to the red peak and increase penalty as it moves away
        scaling_factor = (after_position / 80)

        if after_position == -1: # if the robot is lost
            reward = -10.0
        elif after_position < init_position and after_position != -1 or init_position == -1 and after_position != -1: # if the robot is moving towards the red peak
            reward = 1.0 * 5*(1-scaling_factor)
        elif after_position == 0 and after_position != -1: # if the robot is at the red peak
            reward = 10.0
        elif after_position > init_position and init_position != -1: # if the robot is moving away from the red peak
            reward = -1.0 * 5 * scaling_factor
        else: reward = 0

        return reward

    def is_done(self, position):
        # Check if the robot is lost or if it is at the red peak
        if position == -1: 
            return True
        elif position == 0: 
            return True
        else: 
            return False

# EYESIM FUNCTIONS --------------------------------------------------------------------------------------------------

# Function to set the speed of the robot based on the action taken
def eyesim_set_robot_speed(direction): 
    # Set the speed of the robot based on the action taken
    speed = 25
    VWSetSpeed(0,round(speed*direction)) # Set the speed of the robot
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
    LCDImage(display_img)

    return processed_img

# Function to get the distance to the red peak
def eyesim_get_position(): 
    # Get the distance to the red peak using the find_center function
    index = find_center()

    # Adjust index to be between 0 and 80 
    # With 0 being the red peak and 80 being the edge of the camera view and -1 being lost
    if index >= 80:
        index -= 80
    elif index != -1 and index < 80:
        index = 80 - index
    return index

# Function to set the can position randomly
def rand_can_pos(): 
    CAN_pos_x = randint(200, 1800)
    CAN_pos_y = randint(200, 1800)
    return CAN_pos_x, CAN_pos_y

# Function to reset the robot and can positions in the simulation
def eyesim_reset(): 
    # stop robot movement
    VWSetSpeed(0,0)

    # Set the robot and can positions randomly
    S4_pos_x = randint(200, 1800)
    S4_pos_y = randint(200, 1800)

    CAN_pos_x = randint(200, 1800)
    CAN_pos_y = randint(200, 1800)
    
    # Ensure the can is not too close to the robot if so reposition it
    while get_distance(S4_pos_x,S4_pos_y,CAN_pos_x,CAN_pos_y) < 500:
        CAN_pos_x, CAN_pos_y = rand_can_pos()
    
    # Find the angle between the robot and the can
    angle = math.atan2(CAN_pos_y-S4_pos_y,CAN_pos_x-S4_pos_x)
    angle = round(math.degrees(angle))

    # Adjust the angle to be between 0 and 360 degrees
    if angle < 0:
        angle = 360 + angle

    # Add a random angle variation to the robot's angle in range 
    angle_variation = 0
    while angle_variation < 5 and angle_variation > -5: # Ensure the angle variation is not too small
        angle_variation = randint(-30,30)
    
    # Adjustment for simulation functions
    angle = -angle + angle_variation

    # Position the robot and can in the simulation
    SIMSetRobot(2,S4_pos_x,S4_pos_y,10,angle)
    SIMSetObject(1,CAN_pos_x,CAN_pos_y,0,0)
    return

# Function to calculate the distance between two points
def get_distance(x1,y1,x2,y2): 
    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distance

# INITIALIZE ----------------------------------------------------------------------------------------------------------

# Register the environment with gymnasium and create an instance of it
gym.register(
    id="gymnasium_env/EyeSimEnv",
    entry_point=EyeSimEnv,
)
env = gym.make("gymnasium_env/EyeSimEnv")

# train and test parameters
learning_rate=0.0001
n_steps=128
batch_size=64
n_epochs=10
gamma=0.99
ent_coef=0.01
clip_range=0.2

# TEST ----------------------------------------------------------------------------------------------------------------

# Function to test the environment and the robot's performance
def test(): 
    
    # Reset the environment and check if it is valid
    env.reset()
    check_env(env)

    # Test the environment by taking random actions
    episodes = 100
    for ep in range(episodes):
        action = env.action_space.sample()
        obs, reward, done, _, _= env.step(action)
        print(f"Episode: {ep+1}, Action: {action}, Reward: {reward}")

# TRAIN ---------------------------------------------------------------------------------------------------------------

# Function to train the robot behaviour using an reinforcement learning algorithm
def train(): 

    # Check if the models directory exists, if not create it
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Check if the logs directory exists, if not create it
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Reset the environment and check if it is valid
    env.reset()
    check_env(env)

    # Define the PPO model with the specified parameters
    model = PPO("CnnPolicy", env=env, verbose=1, tensorboard_log=logdir, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma, ent_coef=ent_coef, clip_range=clip_range)

    # Train the model for 10 iterations, each with 10,000 timesteps
    for i in range(1,10):
        model.learn(total_timesteps=10000, progress_bar = True, reset_num_timesteps=False)
        model.save(f"{models_dir}/{i}")

# LOAD ---------------------------------------------------------------------------------------------------------------- 

# Function to load a pre-trained model and test it
def load(): 
    # Reset the environment and check if it is valid
    env.reset()
    check_env(env)

    # Load the pre-trained model
    trained_model = 9
    model_path = f"{models_dir}/{trained_model}.zip"
    model = PPO.load(model_path,env=env)

    # Test the loaded model by taking actions based on the model's predictions
    episodes = 10
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            print(f"Episode: {ep+1}, Action: {action}, Reward: {reward}")
    
# LOAD AND TRAIN --------------------------------------------------------------------------------------------------------

# Function to load a pre-trained model and continue training it
def load_train(): 
    # Reset the environment and check if it is valid
    env.reset()
    check_env(env)
    
    # Load the pre-trained model
    trained_model = 9
    model_path = f"{models_dir}/{trained_model}.zip"
    model = PPO.load("CNNPolicy", model_path, env=env, verbose = 1, tensorboard_log=logdir, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma, ent_coef=ent_coef, clip_range=clip_range)
    
    iterations = 10
    # Continue training the model
    for i in range(trained_model + 1, trained_model + 1 + iterations):
        model.learn(total_timesteps=10000, progress_bar = True, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{i}")
        
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
    
    # Process image
    procesesed_img = image_processing(img)

    # convert to HSI and find index of red color peak
    [h, s, i] = IPCol2HSI(procesesed_img)  
    index = colour_search(h, s, i)

    return index

# COLOUR DETECTION -------------------------------------------------------------------------------------------------------

# Function to search for the red color in the image
def colour_search(h, s, i): 
    # Initialize a histogram array for each column (0 to 159)
    histogram = [0] * CAMWIDTH  

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
        if count > max and count > 5:
            max = count
            index = x

    return index
        
# MAIN -------------------------------------------------------------------------------------------------------

def main():
    # Initialize the camera with QQVGA resolution (160x120)
    CAMInit(QQVGA) 
    LCDImageStart(0,0,CAMWIDTH,DESIRED_CAMHEIGHT)

    while True:
        LCDMenu("TRAIN", "LOAD", "TEST", "STOP")

        key = KEYRead()
        if key == KEY1: # Train the model
            train()

        elif key == KEY2: # Load a pre-trained model and test it
            load()

        elif key == KEY3: # Test the environment and the robot's performance
            test()
            
        elif key == KEY4: # Stop the program
            break

main()
