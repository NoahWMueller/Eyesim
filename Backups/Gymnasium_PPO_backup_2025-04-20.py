#!/usr/bin/env python

# IMPORTS ------------------------------------------------------------------------------------------------------------

import os
from random import randint
from sys import version
from turtle import mode
import cv2
import math
from eye import *
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import time

# GLOBAL VARIABLES ---------------------------------------------------------------------------------------------------

CAMWIDTH = 160
CAMHEIGHT = 120
DESIRED_CAMHEIGHT = 60

models_dir = "models/PPO"
logdir = "logs"

# ENVIRONMENT --------------------------------------------------------------------------------------------------------

class EyeSimEnv(gym.Env):
    
    def __init__(self):
        super(EyeSimEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,)) # turn left or right
        self.observation_space = spaces.Box(low=0, high=255, shape=(3,DESIRED_CAMHEIGHT,CAMWIDTH), dtype=np.uint8) # takes camera image

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        eyesim_reset()
        observation = eyesim_get_observation()
        info = {}
        return observation, info

    def step(self, action):
        
        init_position = eyesim_get_position() # distance to red peak
        
        eyesim_set_robot_speed(action[0]) # action[0] is angular motor speeds.

        observation = eyesim_get_observation() # camera image

        after_position = eyesim_get_position() # distance to red peak

        # Calculate reward based on position or sensor readings
        reward = self._calculate_reward(init_position,after_position)
        # Determine if the episode is done
        done = self._is_done(after_position)
        info = {}
        return observation, reward, done, False, info

    def _calculate_reward(self, init_position, after_position):
        scaling_factor = 5* (after_position / 80) # scaling factor to make the reward more manageable
        if after_position < init_position and after_position != -1: # if the robot is moving towards the red peak
            reward = 1.0 * scaling_factor
        elif after_position < 1 and after_position != -1: # if the robot is at the red peak
            reward = 10.0
        elif after_position == -1: # if the robot is lost
            reward = -10.0
        elif after_position > init_position: # if the robot is moving away from the red peak
            reward = -1.0 * scaling_factor
        else: reward = 0
        return reward

    def _is_done(self, position):
        if position < 1: return True
        else: return False

gym.register(
    id="gymnasium_env/EyeSimEnv",
    entry_point=EyeSimEnv,
)

# EYESIM FUNCTIONS --------------------------------------------------------------------------------------------------

def eyesim_set_robot_speed(direction):
    speed = 100
    VWSetSpeed(0,speed*direction)
    time.sleep(0.1) # wait for a bit to let the robot move
    VWSetSpeed(0,0) # stop from moving

def eyesim_get_observation():
    img = CAMGet() # get image from camera
    processed_img = image_processing(img)
    transposed_img = np.transpose(processed_img, (2,0,1)) # process image
    return transposed_img

def eyesim_get_position():
    distance = find_center()
    if distance >= 80:
        distance -= 80
    return distance

def eyesim_reset():
    # stop robot movement
    # set robot and can location randomly and make sure they arent in same spot
    VWSetSpeed(0,0)
    S4_pos_x = randint(200, 1800)
    S4_pos_y = randint(200, 1800)

    CAN_pos_x = randint(200, 1800)
    CAN_pos_y = randint(200, 1800)
    
    while get_distance(S4_pos_x,S4_pos_y,CAN_pos_x,CAN_pos_y) < 500 and find_center() != -1:
        CAN_pos_x = randint(200, 1800)
        CAN_pos_y = randint(200, 1800)

    angle = math.atan2(CAN_pos_y-S4_pos_y,CAN_pos_x-S4_pos_x) # angle of the line between the two points
    angle = round(math.degrees(angle)) # convert to degrees
    if angle < 0:
        angle = 360 + angle
    angle_variation = randint(-30,30) # random angle variation
    SIMSetRobot(2,S4_pos_x,S4_pos_y,10,-angle+angle_variation)
    SIMSetObject(1,CAN_pos_x,CAN_pos_y,0,0)
    return

def get_distance(x1,y1,x2,y2):
    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distance

# TRAIN ---------------------------------------------------------------------------------------------------------------

version = 1.0

def train():

    # Check if the models directory exists, if not create it
    models_dir = models_dir + "-" + str(version)
    logdir = logdir + "-" + str(version)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = gym.make("gymnasium_env/EyeSimEnv")
    env.reset()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, learning_rate=0.0001, n_steps=128, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01, clip_range=0.2)

    for i in range(1,10):
        model.learn(total_timesteps=10000, progress_bar = True, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/PPO-{version}-Eyesim-{i}")

# LOAD ---------------------------------------------------------------------------------------------------------------- 

def load():
    env = gym.make("gymnasium_env/EyeSimEnv-v0")
    env.reset()
    i = 1
    model_path = f"{models_dir}/PPO-{version}-Eyesim-{i}.zip"

    model = PPO.load(model_path,env=env)

    episodes = 10

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            print(f"Episode: {ep+1}, Action: {action}, Reward: {reward}")
    
# LOAD AND TRAIN --------------------------------------------------------------------------------------------------------

def load_train():
    env = gym.make("gymnasium_env/EyeSimEnv-v0")
    env.reset()
    start = 9
    model_path = f"{models_dir}/PPO-Eyesim-{start}.zip"

    model = PPO.load(model_path,env=env)
    
    for i in range(10,50):
        model.learn(total_timesteps=10000, progress_bar = True, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/PPO-Eyesim-{i}")
        env.reset()
        
# IMAGE PROCESSING -------------------------------------------------------------------------------------------------------

# converts image to new size
def image_processing(image):
    decoded_array = np.asarray(image, dtype=np.uint8)
    image_reshaped = decoded_array.reshape((CAMHEIGHT, CAMWIDTH, 3))

    # image cropping to desired height
    middle = CAMHEIGHT//2
    lower = middle - DESIRED_CAMHEIGHT//2
    upper = middle + DESIRED_CAMHEIGHT//2

    image_reshaped = image_reshaped[lower:upper, :, :]

    cropped_image = cv2.resize(image_reshaped, (CAMWIDTH, DESIRED_CAMHEIGHT))
    return cropped_image

# function to find and draw center of red object in the image
def find_center():
    # Get image data from the camera
    img = CAMGet()
    
    # process image
    procesesed_img = image_processing(img)
    display_img = procesesed_img.ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
    # LCDImageStart(0,0,CAMWIDTH,DESIRED_CAMHEIGHT)
    # LCDImage(display_img)

    # draw centered line
    # LCDLine(int(0.5*CAMWIDTH), 0, int(0.5*CAMWIDTH), DESIRED_CAMHEIGHT-1, BLUE)

    # convert to HSI and find red
    [h, s, i] = IPCol2HSI(display_img)  # Convert the image to HSI format
    
    index = colour_search(h, s, i)

    # draw line where red is maximum
    # LCDLine(index, 0, index, DESIRED_CAMHEIGHT-1, GREEN)

    return index

# COLOUR DETECTION -------------------------------------------------------------------------------------------------------

def colour_search(h, s, i):
    histogram = [0] * CAMWIDTH  # Initialize a histogram array for each column (0 to 159)

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
    CAMInit(QQVGA)

    while True:
        # Get image data from the camera
        img = CAMGet()
        procesesed_img = image_processing(img)
        display_img = procesesed_img.ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
        LCDImageStart(0,0,CAMWIDTH,DESIRED_CAMHEIGHT)
        LCDImage(display_img)
        
        LCDMenu("TRAIN", "LOAD", "L-T", "STOP")

        key = KEYRead()

        if key == KEY1:
            train()

        elif key == KEY2:
            load()

        elif key == KEY3:
            load_train()
            
        elif key == KEY4:
            break

main()
