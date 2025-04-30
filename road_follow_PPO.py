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
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# GLOBAL VARIABLES ---------------------------------------------------------------------------------------------------

# Constants for camera settings
CAMWIDTH = 160
CAMHEIGHT = 120
DESIRED_CAMHEIGHT = 60

# Algorithm used for training
algorithm = "PPO" 

# Current version of the code for saving models and logs
version = 1.3

# Directory paths for saving models and logs
models_dir = f"models/{version}"
logdir = f"logs/{version}"

# Define the world coordinates for the flipped centroids
flipped_centroids_world_coordinates = [
    (2781,533),(848,571),(581,705),(400,886),(295,1105),
    (248,2086),(267,3486),(343,3762),(505,4010),(743,4229),
    (1000,4371),(2324,4438),(4190,4381),(4543,4143),(4714,3819),
    (4733,3495),(4610,3190),(3971,2524),(3067,1648),(2733,1419),
    (2305,1381),(1895,1581),(1676,1905),(1619,2390),(1638,2952),
    (1752,3210),(1981,3438),(2343,3600),(2762,3581),(3648,2829),
    (4657,1686),(4714,1257),(4552,867),(4248,629)
]

# ENVIRONMENT --------------------------------------------------------------------------------------------------------

# Custom environment for the robot simulation using OpenAI Gymnasium
class EyeSimEnv(gym.Env):
    
    def __init__(self):
        super(EyeSimEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,)) # Float action space for robot angular speed, range from -1 to 1 TODO include second value for linear speed
        self.observation_space = spaces.Box(low=0, high=255, shape=(DESIRED_CAMHEIGHT,CAMWIDTH,3), dtype=np.uint8) # Image observation space, 3 channels (RGB), 60x160 pixels

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        eyesim_reset()
        observation = eyesim_get_observation()
        info = {}
        return observation, info

    def step(self, action):
        linear, angular = 0, 0 # TODO update linear and angular speed based on action

        # Determines if robot is inside left lane or has gotten lost
        position = eyesim_get_position() 
        
        # Set robot linear and angular speed based on action
        eyesim_set_robot_speed(linear, angular) 

        # Read image from camera
        observation = eyesim_get_observation()

        # Calculate reward based on position
        reward = self.calculate_reward(position)

        # Determine if the episode is done
        done = self.is_done(position)

        # Truncated is not used in this case, but included for compatibility with gym API
        truncated = False
        info = {}

        return observation, reward, done, truncated, info

    def calculate_reward(self, position):
        return False

    def is_done(self, position):
        return False

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
    LCDImage(display_img)

    return processed_img

# Function to get the distance to the red peak
def eyesim_get_position(): 
    # TODO determine if robot is inside new polygon or is outside the allowed space
    return


# Function to reset the robot and can positions in the simulation
def eyesim_reset(): 
    # Stop robot movement
    VWSetSpeed(0,0)
    # TODO randomize position based on centroid positions
    # TODO set angle so robot is always facing in the correct direction

    # Position the robot and can in the simulation
    SIMSetRobot(2,0,0,10,0)
    return

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

    # Define the PPO model with the specified parameters
    model = PPO("CnnPolicy", env=env, verbose=1, tensorboard_log=logdir, n_steps=n_steps, learning_rate=learning_rate)

    # Train the model for 10 iterations, each with 10,000 timesteps
    for i in range(1,10):
        model.learn(total_timesteps=10000, progress_bar = True, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
        model.save(f"{models_dir}/{i}")

# LOAD ---------------------------------------------------------------------------------------------------------------- 

# Function to load a pre-trained model and test it
def load(): 

    # Load the pre-trained model
    trained_model = 6
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

    # Load the pre-trained model
    trained_model = 9
    model_path = f"{models_dir}/{trained_model}.zip"
    model = PPO.load(model_path, env)
    
    iterations = 10
    # Continue training the model
    for i in range(trained_model + 1, trained_model + 1 + iterations):
        model.learn(total_timesteps=10000, progress_bar = True, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
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
