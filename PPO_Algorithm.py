#!/usr/bin/env python

# IMPORTS ------------------------------------------------------------------------------------------------------------

import os
import re
import time
from eye import *
import gymnasium as gym
from random import randint
from stable_baselines3 import PPO
from Gymnasium_Env import EyeSimEnv

# GLOBAL VARIABLES ---------------------------------------------------------------------------------------------------

# Current version of the code for saving models and logs
version = 1.6

# Directory paths for saving models and logs
models_dir = f"models/Carolo/{version}"
log_dir = f"logs/Carolo/{version}"

# Algorithm used for training
algorithm = "PPO" 
policy_network = "CnnPolicy" # Policy network used for training

# Training parameters
learning_rate=0.0001
n_steps=2048

left_lane_30speedlimit = [1,28]
right_lane_30speedlimit = [35,62]
left_lane_stop = 55
right_lane_stop = 6

# INITIALIZE ----------------------------------------------------------------------------------------------------------------

# Register the environment with gymnasium and create an instance of it
env_id = "gymnasium_env/EyeSimEnv"

if env_id not in gym.registry:
    gym.register(
        id=env_id,
        entry_point=f"road_follow_PPO:{EyeSimEnv}",
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

    angular_model_dir = f"{models_dir}/angular"
    linear_model_dir = f"{models_dir}/linear"
    angular_log_dir = f"{log_dir}/angular"
    linear_log_dir = f"{log_dir}/linear"

    while True:
        LCDMenu("ANGULAR", "LINEAR", "BOTH", "BACK")
        key = KEYRead()

        env = None
        model_save_dir = "None"
        model_log_dir = "None"

        if key == KEY1: # Train the angular model
            env = EyeSimEnv(control_mode='angular')
            model_save_dir = f"{angular_model_dir}/angular_model_0"
            model_log_dir = angular_log_dir

        if key == KEY2: # Train the linear model
            env = EyeSimEnv(control_mode='linear')
            model_save_dir = f"{linear_model_dir}/linear_model_0"
            model_log_dir = linear_log_dir

        if key == KEY3: # Train the both model
            env = EyeSimEnv(control_mode='both')
            model_save_dir = f"{models_dir}/model_0"
            model_log_dir = log_dir
            
        elif key == KEY4:
            break
        
        if env:# Define the PPO model with the specified parameters
            print("Model save directory:", model_save_dir, "Model log directory:", model_log_dir)
            # model = PPO(policy_network, env=env, verbose=1, tensorboard_log=model_log_dir, learning_rate=learning_rate, n_steps=n_steps)
            # model.learn(total_timesteps=100*n_steps, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
            # model.save(model_save_dir)

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

    # Initialize variables for speed and stop logic
    current_speed = 0
    current_reward = 0.0
    iteration = 0
    stop_reached = False
    stop_time = 0.0
    speed_limit = 0.0
    total_iterations = 5

    # Continue testing the loaded model until the user decides to stop
    while True:
        if done:
            obs, _ = env.reset()
            done = False
        action, _ = model.predict(obs)
        obs, reward, done, _, info= env.step(action)
        
        # Display the current action, reward, done status, and additional info
        print(f"Reward:{reward}, Action: {action}, Done: {done}, Info: {info}")
        
        # Retrieve the current centroid from the info dictionary
        current_centroid = info["Current_Centroid"]
        current_lane = info["Current_Lane"]

        # Gather reward and speed information to average out
        if iteration < total_iterations:
            iteration += 1
            # Convert the speed to speed limit
            current_speed += action[1] * 100 
            current_reward += round(float(reward),2)

        else:
            # Clear the LCD area for speed display
            LCDArea(0,CAMHEIGHT, CAMWIDTH, CAMHEIGHT*2, BLACK,1) 
            
            # Check if the robot has reached a stop and if it has been stopped for more than 3 seconds
            if stop_reached and time.time() - stop_time >= 3.0: 
                stop_reached = False
                stop_time = 0.0
            
            # Speed limit logic
            if current_lane == "left_lane":
                stop_centroids = left_lane_stop
                speed_limit_centroids = left_lane_30speedlimit
            else:
                stop_centroids = right_lane_stop
                speed_limit_centroids = right_lane_30speedlimit

            if speed_limit_centroids[0] <= current_centroid <= speed_limit_centroids[1]:
                speed_limit = 0.75
            elif current_centroid == stop_centroids:
                if not stop_reached:
                    speed_limit = 0.0
                    if action[1] == 0.0:
                        stop_reached = True
                        stop_time = time.time()
                else:
                    speed_limit = 0.5
            else:
                speed_limit = 0.5
            
            # Display the speed limit and average speed on the LCD
            LCDSetColor(RED,BLACK)
            LCDSetPrintf(10,0, "Speed Limit: %d", int(speed_limit*100))
            LCDSetColor(WHITE,BLACK)
            LCDSetPrintf(12,0, "Average Speed: %d", int(current_speed/total_iterations))
            LCDSetColor(WHITE,BLACK)
            LCDSetPrintf(14,0, "Average Reward: %.2f", round((current_reward/total_iterations),2))
            
            # Reset the speed count and current speed for the next iteration
            iteration = 0
            current_speed = 0
            current_reward = 0

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

# MAIN -------------------------------------------------------------------------------------------------------

def main():
    # Initialize the camera with QQVGA resolution (160x120)
    CAMInit(QQVGA) 
    LCDImageStart(0,0,CAMWIDTH,CAMHEIGHT)

    while True:
        LCDMenu("TRAIN", "TEST", "LOAD", "STOP")

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
