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
version = 1.1

# Directory paths for saving models and logs
models_dir = f"models/Carolo/{version}"
logdir = f"logs/Carolo/{version}"

# polygon positions
current_polygon = np.array([])
current_centroid = 0
next_polygon = np.array([])
next_centroid = 0

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


# ENVIRONMENT --------------------------------------------------------------------------------------------------------

# Custom environment for the robot simulation using OpenAI Gymnasium
class EyeSimEnv(gym.Env):
    
    def __init__(self):
        super(EyeSimEnv, self).__init__()
        # Define the lower and upper bounds
        low = np.array([-1.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32) # Float action space for robot angular speed, range from -1 to 1 TODO include second value for linear speed
        self.observation_space = spaces.Box(low=0, high=255, shape=(DESIRED_CAMHEIGHT,CAMWIDTH,3), dtype=np.uint8) # Image observation space, 3 channels (RGB), 60x160 pixels

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        eyesim_reset()
        observation = eyesim_get_observation()
        info = {}
        return observation, info

    def step(self, action):
        angular, linear = action[0], action[1] # linear and angular action

        # Determines if robot is inside left lane or has gotten lost
        result1, result2 = eyesim_get_position() 
        
        # Set robot linear and angular speed based on action
        eyesim_set_robot_speed(linear, angular) 

        # Read image from camera
        observation = eyesim_get_observation()

        # Calculate reward based on position
        reward = self.calculate_reward(result1, result2)

        # Determine if the episode is done
        done = self.is_done(result1,result2)

        # Truncated is not used in this case, but included for compatibility with gym API
        truncated = False
        info = {}

        return observation, reward, done, truncated, info

    def calculate_reward(self, result1, result2):
        global current_centroid
        if result2 > 0:
            # update previous polygon to current polygon and repeat 
            current_centroid += 1
            current_centroid %= 34
            update_polygon()
            return 1.0
        if result1 < 0 and result2 < 0: 
            return -1.0
        else:
            return 0.0

    def is_done(self, result1, result2):
        # Determine if the robot left all allowable polygons
        if result1 == -1 and result2 == -1: return True
        else: return False

# EYESIM FUNCTIONS --------------------------------------------------------------------------------------------------

# Function to set the speed of the robot based on the action taken
def eyesim_set_robot_speed(linear, angular): 
    # Set the speed of the robot based on the action taken
    linear_speed = 100
    angular_speed = 50
    VWSetSpeed(round(linear_speed*linear),round(angular_speed*angular)) # Set the speed of the robot
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

def update_polygon():
    global current_polygon, next_polygon, next_centroid
    current_polygon = np.array([
        flipped_left_lane_world_coordinates[current_centroid*2],
        flipped_left_lane_world_coordinates[current_centroid*2+1],
        flipped_left_lane_world_coordinates[(current_centroid*2+3)],
        flipped_left_lane_world_coordinates[(current_centroid*2+2)]
    ], np.int32)

    next_centroid = (current_centroid + 1)% 34

    next_polygon = np.array([
        flipped_left_lane_world_coordinates[next_centroid*2],
        flipped_left_lane_world_coordinates[next_centroid*2+1],
        flipped_left_lane_world_coordinates[(next_centroid*2+3)],
        flipped_left_lane_world_coordinates[(next_centroid*2+2)]
    ], np.int32)


# Function to get the distance to the red peak
def eyesim_get_position(): 
    global current_polygon, next_polygon
    [x,y,_,_] = SIMGetRobot(1)
    point = (x.value, y.value)
    
    update_polygon()

    # Reshape the polygon points
    current_polygon = current_polygon.reshape((-1, 1, 2))

    # Reshape the polygon points
    next_polygon = next_polygon.reshape((-1, 1, 2))

    # Check if the point is inside the polygon
    current_result = cv2.pointPolygonTest(current_polygon, point, False)

    # Check if the point is inside the polygon
    next_result = cv2.pointPolygonTest(next_polygon, point, False)

    # If the point is inside the polygon return
    return current_result, next_result

# Function to reset the robot and can positions in the simulation
def eyesim_reset(): 
    global current_centroid
    # Stop robot movement
    VWSetSpeed(0,0)

    # # Pick random position along the road to start
    random = randint(0,len(coordinates)-1)
    current_centroid = random

    # # Position the robot in the simulation
    x,y,phi = coordinates[random]
    SIMSetRobot(1,x,y,10,phi+180) # Add 180 degrees to the angle to flip robot into correct direction
# INITIALIZE ----------------------------------------------------------------------------------------------------------

# Register the environment with gymnasium and create an instance of it
gym.register(
    id="gymnasium_env/EyeSimCaroloEnv",
    entry_point=EyeSimEnv,
)
env = gym.make("gymnasium_env/EyeSimCaroloEnv")

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
    while True:
        LCDMenu("-", "-", "-", "STOP")
        action = env.action_space.sample()
        obs, reward, done, _, _= env.step(action)
        print(f"Action: {action}")

        key = KEYRead()
        if key == KEY4: # Train the model
            break

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
    done = False
    obs, _ = env.reset()
    while True:
        if done:
            obs, _ = env.reset()
            done = False
        LCDMenu("-", "-", "-", "STOP")
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(f"Reward: {reward}, Done: {done}")

        key = KEYRead()
        if key == KEY4: # Train the model
            break
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
        LCDMenu("-", "-", "-", "STOP")
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
