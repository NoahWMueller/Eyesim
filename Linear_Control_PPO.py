#!/usr/bin/env python

# TO-DO --------------------------------------------------------------------------------------------------------------

# create 2 extra tracks, one for 10 limit, one for stop sign, assuming is for 30 speed limit
# maybe only 2 distant tracks, change between 30 and 10 speed limit sign on same track
# picks one of 3/2 tracks at random to travel, random sign position along the track except for stop sign which randomizes robot position
# update speed reward function
# find values for TBDs

# IMPORTS ------------------------------------------------------------------------------------------------------------

import os
import re
import cv2
import time
from eye import *
import numpy as np
import gymnasium as gym
from random import randint
from stable_baselines3 import PPO

# GLOBAL VARIABLES ---------------------------------------------------------------------------------------------------

# Constants for camera settings
CAMWIDTH = 160
CAMHEIGHT = 120

# Current version of the code for saving models and logs
version = 1.6

# Directory paths for saving models and logs
models_dir = f"models/Carolo/{version}"
logdir = f"logs/Carolo/{version}"

# Algorithm used for training
algorithm = "PPO" 
# Policy network used for training
policy_network = "CnnPolicy" 

# Training parameters
learning_rate=0.0001
n_steps=2048

# To be determined values
TBD = 0

# Starting positions for the robot on the tracks
start_positions = [TBD,TBD] 

# GYMNASIUM ENVIRONMENT --------------------------------------------------------------------------------------------------------

# Custom environment for the robot simulation using OpenAI Gymnasium
class EyeSimEnv(gym.Env):
    
    def __init__(self):
        super(EyeSimEnv, self).__init__()
        # Float action space for robot linear speed, range from 0.0 to 1.0
        self.action_space = gym.spaces.Box(low=np.array([0.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32), dtype=np.float32)

        # Image observation space, 3 channels (RGB), 120x160 pixels
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(CAMHEIGHT,CAMWIDTH,3), dtype=np.uint8) 
        
        # Initialize class variables
        self.stop_reached = False
        self.stop_time = time.time()
        self.speed_limit = 1.0 # Base speed limit
        self.track = randint(1,3) # Randomly select a track for the robot to follow
        self.speedlimit10_position = randint(TBD, TBD) # Randomly select a position for the 10 limit sign
        self.speedlimit30_position = randint(TBD, TBD) # Randomly select a position for the 30 limit sign
        self.stop_sign_robot_position = randint(TBD, TBD) # Randomly select a position for the robot on stop sign track

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.eyesim_reset()
        observation = self.eyesim_get_observation()
        info = {}
        return observation, info

    def step(self, action):
        linear_speed = action[0] # linear and angular action
        
        position = self.eyesim_get_position()
        # Set robot linear and angular speed based on action
        self.eyesim_set_robot_speed(linear_speed) 

        # Read image from camera
        observation = self.eyesim_get_observation()

        # Calculate reward based on position
        reward = self.calculate_speed_reward(linear_speed) # Calculate the speed reward based on the speed

        # Determine if the episode is done
        done = self.is_done(position)

        # Truncated is not used in this case, but included for compatibility with gym API
        truncated = False

        # Create info dictionary to store additional information
        info = {"Reward": reward, 
                "Stop_Reached": self.stop_reached}

        return observation, reward, done, truncated, info
    
    def calculate_speed_reward(self, linear_speed):
        # if self.track == 1:
        # do track one stuff

        # elif self.track == 2:
        # do track two stuff

        # elif self.track == 3:
        # do track three stuff
        return 0.0
        """
        # If the robot has reached a stop, check if it has been stopped for more than 3 seconds before resetting the stop flag
        if self.stop_reached and time.time() - self.stop_time >= 3.0: 
            self.stop_reached = False
            self.stop_time = 0.0

        # Speed limit logic
        if self.speed_limit_centroids[0] <= self.current_centroid <= self.speed_limit_centroids[1]:
            self.speed_limit = 0.75
        elif self.current_centroid == self.stop_centroid:
            if not self.stop_reached:
                self.speed_limit = 0.0
                if linear_speed == 0.0:
                    self.stop_reached = True
                    self.stop_time = time.time()
            else:
                self.speed_limit = 0.5
        else:
            self.speed_limit = 0.5

        # Smooth reward function: reward is higher when speed is close to target
        speed_error = abs(linear_speed - self.speed_limit)
        penalty = 0.0
        if speed_error > 0.05: # If the speed is lower than the speed limit with tolerance
            penalty = -speed_error # Penalty is proportional to the speed error
        elif speed_error > 0.2: # If the speed is higher than the speed limit with tolerance
            penalty = -speed_error

        return 0.2 + penalty # Return the speed reward, which is 0.2 plus the penaltys
        """

    def is_done(self, position):
        # If the robot has reached the end of the track, return True
        if position >= 4700: return True
        return False

    # INCLUDED EYESIM HELPER FUNCTIONS --------------------------------------------------------------------------------------------------

    def eyesim_get_position(self): 
        x,_,_,_ = SIMGetRobot(1)
        return x.value

    # Function to set the speed of the robot based on the action taken
    def eyesim_set_robot_speed(self, linear): 
        # Set the speed of the robot based on the action taken
        linear_speed = 200
        VWSetSpeed(round(linear_speed*linear),0) # Set the speed of the robot
        time.sleep(0.1) # Sleep for a short time to allow the robot to move
        VWSetSpeed(0,0)

    # Function to get the image from the camera and process it
    def eyesim_get_observation(self): 
        # Get image from camera
        img = CAMGet() 
    
        # Process image
        processed_img = image_processing(img) 

        # Optional: Display the processed image on the LCD screen
        display_img = processed_img.ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
        LCDImage(display_img)

        return processed_img

    # Function to reset the robot and can positions in the simulation
    def eyesim_reset(self): 
        # Stop robot movement
        VWSetSpeed(0,0)

        # Randomly select a track for the robot to follow
        self.track = randint(1,3) 
        self.speedlimit10_position = randint(TBD, TBD) # Randomly select a position for the sign
        self.speedlimit30_position = randint(TBD, TBD) # Randomly select a position for the sign
        self.stop_sign_robot_position = randint(TBD, TBD) # Randomly select a position for the stop sign

        # Position the robot in the simulation on a random track
        if self.track == 3:
            x,y = randint(self.stop_sign_robot_position, TBD),TBD
        else:
            x,y = 305,start_positions[randint(0,1)]

        # Place the signs and robot in the simulation
        self.place_signs() 
        SIMSetRobot(1,x,y,10,TBD)

# Function to check if the objects are in the correct position and set them if not
    def place_signs(self):
            SIMSetObject(2, TBD, TBD, 10, TBD) # stop sign
            SIMSetObject(3, self.speedlimit10_position, TBD, 10, TBD) # speed limit 10 sign
            SIMSetObject(4, self.speedlimit30_position, TBD, 10, TBD) # speed limit 30 sign

# INITIALIZE ----------------------------------------------------------------------------------------------------------------

# Register the environment with gymnasium and create an instance of it
env_id = "gymnasium_env/EyeSimCaroloEnv"

if env_id not in gym.registry:
    gym.register(
        id=env_id,
        entry_point="road_follow_PPO:EyeSimEnv",
    )

env = gym.make(env_id)

# TEST ----------------------------------------------------------------------------------------------------------------

# Function to test the environment and the robot's performance
def test(): 
    env.reset()
    while True:
        action = env.action_space.sample()
        _, reward, done, _, info= env.step(action)
        print(f"Reward: {reward}, Action: {action}, Done: {done}, Current_Centroid: {info['Current_Centroid']}, Current_Lane: {info['Current_Lane']}")
    
        if done: # If the episode is done, reset the environment
            env.reset()

        # Stop the random actions
        LCDMenu("-", "-", "-", "STOP")
        key = KEYRead()
        if key == KEY4:
            VWSetSpeed(0,0)
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
    model = PPO(policy_network, env=env, verbose=1, tensorboard_log=logdir, learning_rate=learning_rate, n_steps=n_steps)

    # Train the model for 100,000 steps
    model.learn(total_timesteps=100*n_steps, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
    model.save(f"{models_dir}/model_0")

# LOAD ---------------------------------------------------------------------------------------------------------------- 

# Function to load a pre-trained model and test it
def load_test(): 
    # Find the most recent model
    result = find_latest_model()
    if result is None:
        return
    most_recent_model, _ = result

    print(f"Loading model: {most_recent_model}")

    # Load the pre-trained model
    trained_model = most_recent_model
    model_path = f"{models_dir}/{trained_model}"
    model = PPO.load(model_path, env=env)

    # Test the loaded model by taking actions based on the model's predictions
    done = False
    obs, _ = env.reset()

    # Continue testing the loaded model until the user decides to stop
    while True:
        if done:
            obs, _ = env.reset()
            done = False
        action, _ = model.predict(obs)
        obs, reward, done, _, info= env.step(action)

        LCDMenu("-", "-", "-", "STOP")
        key = KEYRead()
        if key == KEY4: # Train the model
            break
    
# LOAD AND TRAIN --------------------------------------------------------------------------------------------------------
    
# Function to load a pre-trained model and continue training it
def load_train(): 

    # Find the most recent model
    result = find_latest_model()
    if result is None:
        return
    most_recent_model, iteration = result

    print(f"Loading model: {most_recent_model} for further training with iteration {iteration}")

    # Load the pre-trained model
    model_path = f"{models_dir}/{most_recent_model}"
    model = PPO.load(model_path, env=env)

    # Continue training the model v
    while True:
        LCDMenu("TRAIN", "-", "-", "BACK")
        key = KEYRead()
        if key == KEY1:  # Train the model
            model.learn(total_timesteps=50*n_steps, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
            new_model = f"model_{iteration}"
            model.save(f"{models_dir}/{new_model}")
            iteration += 1
        elif key == KEY4:
            break

# FILE HANDLING -------------------------------------------------------------------------------------------------------

# Function to find the latest model in the models directory
def find_latest_model():
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
    iteration = (int(most_recent_model.split("_")[1].split(".")[0]) + 1) if most_recent_model else 0

    # If no pre-trained model is found, print a message and return
    if iteration == 0 or most_recent_model == "None":
        print("No pre-trained model found. Please train a model first.")
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

# MAIN -------------------------------------------------------------------------------------------------------

def main():
    # Initialize the camera with QQVGA resolution (160x120)
    CAMInit(QQVGA) 
    LCDImageStart(0,0,CAMWIDTH,CAMHEIGHT)

    while True:
        LCDMenu("TRAIN", "TEST", "LOAD", "STOP")
        print("1")

        key = KEYRead()
        if key == KEY1: # Train the model
            train()

        elif key == KEY2: # Load a pre-trained model and test it
            while True:
                LCDMenu("TEST_ENV", "OBJECT_POS", "TEST_RESET", "BACK")
                key = KEYRead()
                if key == KEY1: # Load a pre-trained model and test it
                    test()
                elif key == KEY2: # Load a pre-trained model and continue training it
                    for i in range(2,8):
                        print((SIMGetObject(i)))
                if key == KEY3: # Load a pre-trained model and test it
                    env.reset()
                elif key == KEY4: # Stop the program
                    break 

        elif key == KEY3: # Test the environment and the robot's performance
            while True:
                LCDMenu("LOAD_TEST", "LOAD_TRAIN", "-", "BACK")
                key = KEYRead()
                if key == KEY1: # Load a pre-trained model and test it
                    load_test()
                elif key == KEY2: # Load a pre-trained model and continue training it
                    load_train()
                elif key == KEY4: # Stop the program
                    break
            
        elif key == KEY4: # Stop the program
            break

main()

