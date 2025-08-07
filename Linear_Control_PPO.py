#!/usr/bin/env python

# TO-DO --------------------------------------------------------------------------------------------------------------



# IMPORTS ------------------------------------------------------------------------------------------------------------

import time
from eye import *
import gymnasium as gym
from random import randint
from Helper_Functions import *
from stable_baselines3 import PPO

# GLOBAL VARIABLES ---------------------------------------------------------------------------------------------------

# Constants for camera settings
CAMWIDTH = 160
CAMHEIGHT = 120

# Current version of the code for saving models and logs
version = 1.6

# Directory paths for saving models and logs
models_dir = f"models/Carolo/{version}/Linear"
logdir = f"logs/Carolo/{version}/Linear"

# Algorithm used for training
algorithm = "PPO" 
# Policy network used for training
policy_network = "CnnPolicy" 

# Training parameters
learning_rate=0.0001
n_steps=2048

# Starting positions for the robot on the tracks
start_positions = [863,2535] 
stop_sign_position = 2230 # Positions for the stop sign on the tracks
basespeedlimit = 0.6
speedlimit10 = 0.5
speedlimit30 = 0.75

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
        self.completed_stop = False
        self.track = randint(1,3) # Randomly select a track for the robot to follow
        self.speedlimit10_position = randint(1000, 3500) # Randomly select a position for the 10 limit sign
        self.speedlimit30_position = randint(1000, 3500) # Randomly select a position for the 30 limit sign
        self.stop_sign_robot_position = randint(300, 1600) # Randomly select a position for the robot on stop sign track

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
        reward = self.calculate_speed_reward(linear_speed, position) # Calculate the speed reward based on the speed

        # Determine if the episode is done
        done = self.is_done(position)

        # Truncated is not used in this case, but included for compatibility with gym API
        truncated = False

        # Create info dictionary to store additional information
        info = {"Reward": reward, 
                "Stop_Reached": self.stop_reached}

        return observation, reward, done, truncated, info
    
    def calculate_speed_reward(self, linear_speed, position):
        penalty = 0.0 # Initialize penalty for speed limit violations
        buffer = 550
        if self.track == 1:
            if position >= self.speedlimit10_position - buffer and position <= self.speedlimit10_position: # ramping up to speed limit 10
                # Smooth reward function: reward is higher when speed is close to target
                linear_speed_limit = speedlimit10+(1-((self.speedlimit10_position-position)/buffer))*abs(speedlimit10-basespeedlimit)
                speed_error = abs(linear_speed - linear_speed_limit)
                if speed_error > 0.05: # If the speed is lower than the speed limit with tolerance
                    penalty = -speed_error

            elif position >= self.speedlimit10_position: # reached speed limit 10
                # Smooth reward function: reward is higher when speed is close to target
                speed_error = abs(linear_speed - speedlimit10)
                if speed_error > 0.05: # If the speed is lower than the speed limit with tolerance
                    penalty = -speed_error

            else: # prior to speed limit 10
                # Smooth reward function: reward is higher when speed is close to target
                speed_error = abs(linear_speed - basespeedlimit)
                if speed_error > 0.05: # If the speed is lower than the speed limit with tolerance
                    penalty = -speed_error

        if self.track == 2:
            if position >= self.speedlimit30_position - buffer and position <= self.speedlimit30_position: # ramping up to speed limit 30
                # Smooth reward function: reward is higher when speed is close to target
                linear_speed_limit = speedlimit30-(((self.speedlimit30_position-position)/buffer))*abs(speedlimit30-basespeedlimit)
                speed_error = abs(linear_speed - linear_speed_limit)
                if speed_error > 0.05: # If the speed is lower than the speed limit with tolerance
                    penalty = -speed_error

            if position >= self.speedlimit30_position - buffer: # reached speed limit 30
                # Smooth reward function: reward is higher when speed is close to target
                speed_error = abs(linear_speed - speedlimit30)
                if speed_error > 0.05: # If the speed is lower than the speed limit with tolerance
                    penalty = -speed_error

            else: # prior to speed limit 30
                # Smooth reward function: reward is higher when speed is close to target
                speed_error = abs(linear_speed - basespeedlimit)
                if speed_error > 0.05: # If the speed is lower than the speed limit with tolerance
                    penalty = -speed_error


        elif self.track == 3:
            if position >= (stop_sign_position - 100) and not self.completed_stop:
                
                # Check if robot has stopped
                if linear_speed == 0.0 and not self.stop_reached:
                        self.stop_reached = True
                        self.stop_time = time.time()

                elif self.stop_reached and linear_speed > 0.0 and not self.completed_stop:
                    # If the robot is moving after stopping, reset stop reached
                    self.stop_reached = False
                    self.stop_time = 0.0

                # If the robot has stopped, check if it's been stopped long enough
                elif self.stop_reached:
                    elapsed = time.time() - self.stop_time
                    if elapsed >= 3.0:
                        # Stop duration satisfied; remove speed limit penalty
                        self.stop_reached = False
                        self.stop_time = 0.0
                        self.completed_stop = True

                # If the robot is moving after completing the stop, reset the stop reached flag and stop time
                elif self.completed_stop:
                    self.stop_reached = False
                    self.stop_time = 0.0
                    
                else:
                    # Robot is approaching stop, enforce speed limit
                    speed_error = abs(linear_speed)
                    if speed_error > 0.05:
                        penalty = -speed_error  # Slightly fast: small penalty
            else:
                # Smooth reward function: reward is higher when speed is close to target
                speed_error = abs(linear_speed - basespeedlimit)
                if speed_error > 0.05: # If the speed is lower than the speed limit with tolerance
                    penalty = -speed_error

        return 0.2 + penalty # Return the speed reward, which is 0.2 plus the penaltys
        

    def is_done(self, position):
        # If the robot has reached the end of the track, return True
        if self.track < 3 and position >= 4700: return True
        elif self.track == 3 and position >= stop_sign_position: return True
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
        self.speedlimit10_position = randint(1000, 3500) # Randomly select a position for the sign
        self.speedlimit30_position = randint(1000, 3500) # Randomly select a position for the sign
        self.stop_sign_robot_position = randint(300, 1600) # Randomly select a position for the robot

        # Position the robot in the simulation on a random track
        if self.track == 3:
            x,y = self.stop_sign_robot_position,4442
        else:
            x,y = 300,start_positions[self.track-1]

        # Place the signs and robot in the simulation
        self.place_signs() 

        # Reset completed stop flags
        self.completed_stop = False
        self.stop_reached = False

        SIMSetRobot(1,x,y,10,0)

# Function to check if the objects are in the correct position and set them if not
    def place_signs(self):
            SIMSetObject(2, stop_sign_position, 4673, 10, 0) # stop sign
            SIMSetObject(4, self.speedlimit30_position, 2755, 10, 0) # speed limit 30 sign
            SIMSetObject(3, self.speedlimit10_position, 1100, 10, 0) # speed limit 10 sign

# INITIALIZE ----------------------------------------------------------------------------------------------------------------

# Register the environment with gymnasium and create an instance of it
env_id = "gymnasium_env/LinearEnv"

if env_id not in gym.registry:
    gym.register(
        id=env_id,
        entry_point="Linear_Control_PPO:EyeSimEnv",
    )

env = gym.make(env_id)

# TEST ----------------------------------------------------------------------------------------------------------------

# Function to test the environment and the robot's performance
def test(): 
    env.reset()
    while True:
        action = env.action_space.sample()
        _, reward, done, _, info= env.step(action)
        print(f"Reward: {reward}, Action: {action}, Done: {done}, Info: {info}")
    
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

    # Train the model
    model.learn(total_timesteps=100*n_steps, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
    model.save(f"{models_dir}/model_0")

# LOAD ---------------------------------------------------------------------------------------------------------------- 

# Function to load a pre-trained model and test it
def load_test(): 

    # Find the most recent model
    result = find_latest_model(models_dir)

    # If no model is found, return
    if result is None:
        print("No pre-trained model found.")
        return
    
    # If a model is found, select it
    else:
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
        LCDMenu("-", "-", "-", "STOP")
        key = KEYRead()
        
        # If the robot completes an episode, reset the environment
        if done:
            obs, _ = env.reset()
            done = False
        
        # Predict the action using the loaded model
        action, _ = model.predict(obs)
        obs, reward, done, _, _= env.step(action)
        print(f"Reward: {reward}, Action: {action}, Done: {done}")
        # End testing if user presses the stop key
        if key == KEY4: 
            break
    
# LOAD AND TRAIN --------------------------------------------------------------------------------------------------------
    
# Function to load a pre-trained model and continue training it
def load_train(): 

    # Find the most recent model
    result = find_latest_model(models_dir)

    # If no model is found, return
    if result is None:
        print("No pre-trained model found.")
        return
    
    # If a model is found, select it
    else:
        most_recent_model, iteration = result
        print(f"Loading model: {most_recent_model}")

    # Load the pre-trained model
    model_path = f"{models_dir}/{most_recent_model}"
    model = PPO.load(model_path, env=env)
    print(f"Loading model: {most_recent_model} for further training with model_{iteration}")

    # Continue training the model
    while True:
        LCDMenu("TRAIN", "-", "-", "BACK")
        key = KEYRead()

        # If the user presses the train key, continue training the model
        if key == KEY1:
            model.learn(total_timesteps=50*n_steps, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
            new_model = f"model_{iteration}"
            model.save(f"{models_dir}/{new_model}")
            iteration += 1

        # If the user presses the back key, stop training and return to the main menu
        elif key == KEY4:
            break

# MAIN -------------------------------------------------------------------------------------------------------

def main():
    # Initialize the camera with QQVGA resolution (160x120)
    CAMInit(QQVGA) 
    LCDImageStart(0,0,CAMWIDTH,CAMHEIGHT)

    while True:
        LCDMenu("TRAIN", "TEST", "LOAD", "STOP")
        key = KEYRead()

        # Train the model
        if key == KEY1: 
            train()

        # Testing Menu
        elif key == KEY2: 
            while True:
                LCDMenu("TEST_ENV", "OBJECT_POS", "TEST_RESET", "BACK")
                key = KEYRead()

                if key == KEY1: # Test the environment with random actions
                    test()
                elif key == KEY2: # Display the positions of the objects in the simulation
                    i = 2
                    while SIMGetObject(i)[0].value != 0:
                        [x,y,_,_] = SIMGetObject(i)
                        print(f"Object {i} position: {x.value}, {y.value}")
                        i+=1
                if key == KEY3: # Reset the environment
                    env.reset()
                elif key == KEY4: # Back to the main menu
                    break 
        
        # Load Menu
        elif key == KEY3: 
            while True:
                LCDMenu("LOAD_TEST", "LOAD_TRAIN", "-", "BACK")
                key = KEYRead()
                
                if key == KEY1: # Load a pre-trained model for testing
                    load_test()
                elif key == KEY2: # Load a pre-trained model to continue training
                    load_train()
                elif key == KEY4: # Back to the main menu
                    break
        
        # Stop the program
        elif key == KEY4:
            break

main()

