#!/usr/bin/env python

# IMPORTS ------------------------------------------------------------------------------------------------------------

import time
import math
import random
from eye import *
import gymnasium as gym
from random import randint
from Helper_Functions import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

# GLOBAL VARIABLES ---------------------------------------------------------------------------------------------------

# Constants for camera settings
CAM_SETTING = QVGA
CAMWIDTH = QVGA_X
CAMHEIGHT = QVGA_Y

# Directory paths for saving models and logs
models_dir = f"models/Linear"
logdir = f"logs/Linear"

# Check if the models directory exists, if not create it
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Check if the logs directory exists, if not create it
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Algorithm used for training
algorithm = "PPO" 
policy_network = "MultiInputPolicy" # Policy network used for training

# Training parameters
learning_rate = 0.0003
n_steps = 2048

# Starting positions for the robot on the tracks
# [track 1, track 2, track 3]
robot_y_position = [5250,6180,7103] 
robot_x_position = 400

# [lower x, upper x, stop sign x]
sign_x_positions = [1450, 3450, 2263]

# [track 1, track 2, track 3]
sign_y_positions = [5465, 6386, 7307]

# Max travel distance on tracks
max_distance = 4300

# Distance buffer for sign recognition
buffer = 300
stop_buffer = 100

# Assigned values for speed limits
speedlimit10 = 0.5
speedlimit30 = 0.7
maxspeedlimit = 1.5
minspeedlimit = 0.4

# Robot ID in the simulation
robot_id = 1

# LCD print position
LCD_Right_Print = 52


# GYMNASIUM ENVIRONMENT --------------------------------------------------------------------------------------------------------

# Custom environment for the robot simulation using OpenAI Gymnasium
class EyeSimEnv(gym.Env):
    
    def __init__(self):
        super(EyeSimEnv, self).__init__()
        self.history_length = 4
        
        # Float action space for robot linear speed
        self.action_space = gym.spaces.Box(low=np.array([-1.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32), dtype=np.float32)

        # Image observation space
        self.observation_space = gym.spaces.Dict({
            "image_history": gym.spaces.Box(low=0, high=255, shape=(self.history_length, CAMHEIGHT, CAMWIDTH), dtype=np.uint8),
            "speed_history": gym.spaces.Box(low=0.0, high=maxspeedlimit, shape=(self.history_length,), dtype=np.float32)
        })
        
        # Initialize track variables
        self.track = 1
        self.speedlimit10_position = randint(sign_x_positions[0], sign_x_positions[1]) # Randomly select a position for the 10 limit sign
        self.speedlimit30_position = randint(sign_x_positions[0], sign_x_positions[1]) # Randomly select a position for the 30 limit sign
        self.completed_stop = False
        self.speed_reached = False
        self.stop_position = 0.0
        self.base_speedlimit = round(random.uniform(minspeedlimit, maxspeedlimit), 1)
        self.threshold = 110
        
        # Additionally variable to stop robot movement when episode finishes
        self.timesteps = 0

        # Setting observation spaces
        self.image_history = np.zeros((self.history_length, CAMHEIGHT, CAMWIDTH), dtype=np.uint8)
        self.speed_history = np.zeros(self.history_length, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.eyesim_reset()

        image = self.eyesim_get_observation()
        self.image_history[:] = np.repeat(image[0:1, :, :], self.history_length, axis=0)
        # Initialize speed history with a single value
        self.speed_history[:] = np.full(self.history_length, self.base_speedlimit, dtype=np.float32)
        
        observation = {"image_history": self.image_history, "speed_history": self.speed_history}

        info = {}
        return observation, info

    def step(self, action):
        reward = 0.0

        # Track current episode steps
        self.timesteps += 1

        # Print action to LCD display
        LCDSetPrintf(1,LCD_Right_Print,f"Action: {action[0]:.2f}     ")

        # Determine new current speed of the robot based on action
        current_speed = np.clip(self.speed_history[-1] + (action[0]/10), 0.0, maxspeedlimit)

        # Set robot linear and angular speed based on action
        self.eyesim_set_robot_speed(float(current_speed)) 

        # Get current position of robot
        position = self.eyesim_get_position()

        # Obtain new image
        new_image = self.eyesim_get_observation()

        # Update observation histories
        self.image_history = np.roll(self.image_history, shift=-1, axis=0)
        self.image_history[-1] = new_image[0]
        self.speed_history = np.roll(self.speed_history, shift=-1, axis=0)
        self.speed_history[-1] = current_speed

        # Calculate reward
        reward = self.calculate_speed_reward(float(current_speed), position)

        # Determine if the episode is done
        done = self.is_done(position)
        truncated = done

        # Build observation stack
        observation = {"image_history": self.image_history, "speed_history": self.speed_history}
        
        # Create info dictionary to store additional information
        info = {}

        return observation, reward, done, truncated, info
    
    def calculate_speed_reward(self, linear_speed, position):
        LCDSetPrintf(2,LCD_Right_Print, f"Speed: {float(self.speed_history[-1]):.2f}    ")
        # Initialize score for speed control
        score = 0.0 

        # SPEED LIMIT 30 SIGN
        if self.track == 1:
            target_speed = speedlimit30
            LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {target_speed}    ")

            # Gradual change towards target before the sign 
            if self.speedlimit30_position - buffer <= position <= self.speedlimit30_position:
                ramp_factor = (position - (self.speedlimit30_position - buffer)) / buffer
                gradual_target = self.base_speedlimit + ramp_factor * (target_speed - self.base_speedlimit)

                LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {gradual_target:.2f}    ")
                score += reward_calculation(gradual_target, linear_speed)

            # After passing the sign, enforce speed limit 
            elif position > self.speedlimit30_position:
                if linear_speed - target_speed == linear_speed and not self.speed_reached:
                    self.speed_reached = True
                    score += 2.0  # big bonus for reaching the correct speed for the first time
                score += reward_calculation(target_speed, linear_speed)

            # Before speed change zone, stick to base speed 
            else:
                target_speed = self.base_speedlimit
                LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {target_speed}    ")
                score += reward_calculation(target_speed, linear_speed)

        # SPEED LIMIT 10 SIGN
        elif self.track == 2:
            target_speed = speedlimit10
            LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {target_speed}    ")

            # Gradual change towards target before the sign 
            if self.speedlimit10_position - buffer <= position <= self.speedlimit10_position:
                ramp_factor = (position - (self.speedlimit10_position - buffer)) / buffer
                gradual_target = self.base_speedlimit + ramp_factor * (target_speed - self.base_speedlimit)

                LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {gradual_target:.2f}    ")
                score += reward_calculation(gradual_target, linear_speed)

            # After passing the sign, enforce speed limit 
            elif position > self.speedlimit10_position:
                if linear_speed - target_speed == linear_speed and not self.speed_reached:
                    self.speed_reached = True
                    score += 2.0  # big bonus for reaching the correct speed for the first time
                score += reward_calculation(target_speed, linear_speed)
                
            # Before speed change zone, stick to base speed 
            else:
                target_speed = self.base_speedlimit
                LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {target_speed}    ")
                score += reward_calculation(target_speed, linear_speed)

        # STOP SIGN
        elif self.track == 3: 
            target_speed = 0.0
            LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {target_speed}    ")

            # Deceleration zone before full stop 
            if (sign_x_positions[2] - buffer) <= position < (sign_x_positions[2] - stop_buffer):
                ramp_factor = (position - (sign_x_positions[2] - buffer)) / (buffer - stop_buffer)
                gradual_target = self.base_speedlimit + ramp_factor * (target_speed - self.base_speedlimit)

                LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {gradual_target:.2f}    ")
                score += reward_calculation(gradual_target, linear_speed)

            # Stop enforcement zone
            elif (sign_x_positions[2] - stop_buffer) <= position <= sign_x_positions[2] and not self.completed_stop:

                if linear_speed == 0:
                    LCDSetPrintf(5, LCD_Right_Print, "Stop Completed        ")
                    score += 2.0
                    self.completed_stop = True
                    self.stop_position = position
                    # sleep for two seconds to immitate full stop
                    time.sleep(2)

                # Encourage slowing down toward stop
                score += reward_calculation(target_speed, linear_speed)

            elif self.completed_stop:
                target_speed = self.base_speedlimit
                ramp_factor = (position - self.stop_position) / buffer
                gradual_target = ramp_factor * (target_speed)

                LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {gradual_target:.2f}    ")
                score += reward_calculation(gradual_target, linear_speed)
            else:
                target_speed = self.base_speedlimit
                LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {target_speed}    ")
                score += reward_calculation(target_speed, linear_speed)

        LCDSetPrintf(4,LCD_Right_Print,f"Score: {round(score,2):.2f}    ")
        return score # Return the speed score
        

    def is_done(self, position):
        # If the robot has reached the end of the track or completed an episode
        if self.track == 1 and position >= (self.speedlimit30_position + buffer):
            self.base_speedlimit = round(random.uniform(minspeedlimit, maxspeedlimit), 1) 
            self.speed_history[:] = np.full(self.history_length, self.base_speedlimit, dtype=np.float32)
            self.speed_reached = False
            LCDClear()
            return True
        
        elif self.track == 2 and position >= (self.speedlimit10_position + buffer):
            self.base_speedlimit = round(random.uniform(minspeedlimit, maxspeedlimit), 1) 
            self.speed_history[:] = np.full(self.history_length, self.base_speedlimit, dtype=np.float32)
            self.speed_reached = False
            LCDClear()
            return True
        
        elif self.track == 3 and ((position >= self.stop_position + buffer and self.completed_stop) or (position >= sign_x_positions[2] and not self.completed_stop)):
            self.base_speedlimit = round(random.uniform(minspeedlimit, maxspeedlimit), 1) 
            self.speed_history[:] = np.full(self.history_length, self.base_speedlimit, dtype=np.float32)
            self.completed_stop = False
            self.stop_position = 0.0
            LCDClear()
            return True
        
        elif self.timesteps == n_steps:
            self.timesteps = 0
            VWSetSpeed(0,0)
            LCDClear()
            return True
        
        return False

    # INCLUDED EYESIM HELPER FUNCTIONS --------------------------------------------------------------------------------------------------

    def eyesim_get_position(self): 
        x,_,_,_ = SIMGetRobot(robot_id)
        return x.value

    # Function to set the speed of the robot based on the action taken
    def eyesim_set_robot_speed(self, linear): 
        # Set the speed of the robot based on the action taken
        linear_speed = 200
        _,_,_,phi = SIMGetRobot(robot_id)
        angular_speed = 0
        if phi.value > 0: 
            phi.value = (-phi.value if phi.value < 180 else 360 - phi.value)
            angular_speed = phi.value
        VWSetSpeed(round(linear_speed*linear),angular_speed) # Set the speed of the robot

    # Function to get the image from the camera and process it
    def eyesim_get_observation(self): 
        # Get image from camera
        image = CAMGetGray()

        # Convert the image to a numpy array and reshape to observation space
        processed_image = np.asarray(image, dtype=np.uint8).reshape((1, CAMHEIGHT, CAMWIDTH))

        # Apply thresholding to convert to a binary 
        binary_image = np.where(processed_image > self.threshold, 255, 0).astype(np.uint8)   
        
        # Convert the cropped image to a ctypes pointer for LCD display
        LCDImageGray(binary_image.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)))
        
        return binary_image

    # Function to reset the robot and can positions in the simulation "C:\Users\noah\AppData\Local\Programs\EyeSim\EyeSim.exe"
    def eyesim_reset(self): 
        # Stop robot movement
        VWSetSpeed(0,0)

        # Randomly select a track for the robot to follow
        self.track = (self.track % 3) + 1
        if self.track == 3:
            self.threshold = 80
        else:
            self.threshold = 110
        self.speedlimit10_position = randint(sign_x_positions[0], sign_x_positions[1]) # Randomly select a position for the sign
        self.speedlimit30_position = randint(sign_x_positions[0], sign_x_positions[1]) # Randomly select a position for the sign

        # Position the robot in the simulation on a random track
        x,y = robot_x_position,robot_y_position[self.track-1]

        # Place the signs and robot in the simulation
        self.place_signs() 

        SIMSetRobot(robot_id,x,y,10,0)

# Function to check if the objects are in the correct position and set them if not
    def place_signs(self):
            SIMSetObject(robot_id+1, sign_x_positions[2], sign_y_positions[2], 10, -45) # stop sign
            SIMSetObject(robot_id+2, self.speedlimit30_position, sign_y_positions[0], 10, -45) # speed limit 10 sign
            SIMSetObject(robot_id+3, self.speedlimit10_position, sign_y_positions[1], 10, -45) # speed limit 30 sign

def reward_calculation(target_speed, current_speed):
    # Guassian score calculator
    sigma = 0.2
    y_offset = 0.335
    difference = (current_speed-target_speed)**2
    score = math.exp(-(difference/(2*sigma**2)))-y_offset
    return score


# CNN EXTRACTOR ---------------------------------------------------------------------------------------------------------

class EyeSimExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Shapes
        T = observation_space.spaces["image_history"].shape[0]
        H, W = observation_space.spaces["image_history"].shape[1:]
        
        # CNN for image history (process each frame independently, then flatten)
        self.cnn_his = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            n_flat_his = self.cnn_his(th.zeros(1, 1, H, W)).shape[1]
        n_flat_his *= T  # total features from all history frames

        # MLP for speed history
        speed_history_dim = observation_space.spaces["speed_history"].shape[0]
        self.speed_net = nn.Sequential(
            nn.Linear(speed_history_dim, 32),
            nn.ReLU()
        )

        # Final fusion
        self.fc = nn.Sequential(
            nn.Linear(n_flat_his + 32, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: dict) -> th.Tensor:
        # Image history: (B, T, H, W) -> (B*T, 1, H, W)
        B, T, H, W = obs["image_history"].shape
        img_his = obs["image_history"].unsqueeze(2).float()  # (B, T, 1, H, W)
        img_his = img_his.view(B * T, 1, H, W)
        his_feat = self.cnn_his(img_his)
        his_feat = his_feat.view(B, -1)  # (B, T*features_per_frame)

        # Speed history
        speed_feat = self.speed_net(obs["speed_history"].float())

        # Combine all features
        return self.fc(th.cat([his_feat, speed_feat], dim=1))

# Use custom extractor
policy_kwargs = dict(
    features_extractor_class=EyeSimExtractor,
    features_extractor_kwargs=dict(features_dim=256)  # output dim for policy net
)

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
        obs, reward, done, _, _ = env.step(action)
        print(f"Reward: {reward:.2f}, Action: {action[0]:.2f}, Done: {done}, Current Velocity: {obs['speed_history']}")
    
        if done: # If the episode is done, reset the environment
            env.reset()

        # Stop the random actions
        LCDMenu("-", "-", "-", "STOP")
        key = KEYRead()
        if key == KEY4:
            VWSetSpeed(0,0)
            LCDClear()
            break

# TRAIN ---------------------------------------------------------------------------------------------------------------

# Function to train the robot behaviour using an reinforcement learning algorithm
def train(): 

    # Define the PPO model with the specified parameters
    model = PPO(policy_network, env=env, verbose=1, tensorboard_log=logdir, n_steps=n_steps, policy_kwargs=policy_kwargs,
                learning_rate=learning_rate)
    
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    # eval_callback = EvalCallback(env, eval_freq=n_steps*20, callback_after_eval=stop_train_callback, verbose=1)


    training_count = 1
    # Continue training the model
    while True:
        LCDMenu("Train", "Set Training", "-", "Back")
        key = KEYRead()
        # If the user presses the train key, continue training the model
        if key == KEY1:
            for i in range (1,training_count+1):
                LCDSetPrintf(10,LCD_Right_Print,f"New Model = {i}               ")
                LCDSetPrintf(11,LCD_Right_Print,f"Remaining = {training_count-i}            ")
                model.learn(total_timesteps=n_steps*25, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
                new_model = f"linear_model_{i}"
                model.save(f"{models_dir}/{new_model}")
        if key == KEY2:
            print(f"Training count: {training_count}")
            LCDSetPrintf(12,LCD_Right_Print,f"Training count = {training_count}         ")
            while True:
                LCDMenu("Up", "Down", "-", "Back")
                LCDSetPrintf(12,LCD_Right_Print,f"Training count = {training_count}         ")
                key = KEYRead()
                if key == KEY1:
                    training_count += 1
                    print(f"Training count: {training_count}        ")
                elif key == KEY2:
                    if training_count > 1: training_count -= 1
                    print(f"Training count: {training_count}        ")
                elif key == KEY4:
                    break
        # If the user presses the back key, stop training and return to the main menu
        elif key == KEY4:
            LCDClear()
            break

# LOAD ---------------------------------------------------------------------------------------------------------------- 

# Function to load a pre-trained model and test it
def load_test(model): 

    print(f"Loading model: {model}")

    # Load the pre-trained model
    model_path = f"{models_dir}/{model}"
    model = PPO.load(model_path, env=env)

    # Test the loaded model by taking actions based on the model's predictions
    done = False
    obs, _ = env.reset()

    # Continue testing the loaded model until the user decides to stop
    while True:
        LCDMenu("-", "-", "-", "Stop")
        key = KEYRead()
        
        # If the robot completes an episode, reset the environment
        if done:
            obs, _ = env.reset()
            done = False
        
        # Predict the action using the loaded model
        action, _ = model.predict(obs)
        obs, reward, done, _, _= env.step(action)
        print(f"Reward: {reward:.2f}, Action: {action[0]:.2f}, Done: {done}, Current Velocity: {obs['speed_history']}")

        # End testing if user presses the stop key
        if key == KEY4: 
            LCDClear()
            break

# LOAD AND TRAIN --------------------------------------------------------------------------------------------------------
    
# Function to load a pre-trained model and continue training it
def load_train(model, iteration): 

    print(f"Loading model: {model}")

    # Load the pre-trained model
    model_path = f"{models_dir}/{model}"
    model = PPO.load(model_path, env=env)
    new_iteration = iteration
    training_count = 1
    # Continue training the model
    while True:
        LCDMenu("Train", "Set Training", "-", "Back")
        key = KEYRead()
        # If the user presses the train key, continue training the model
        if key == KEY1:
            for i in range (1,training_count):
                new_iteration += 1
                LCDSetPrintf(10,LCD_Right_Print,f"New Model = {new_iteration}               ")
                LCDSetPrintf(11,LCD_Right_Print,f"Remaining = {training_count-i}            ")
                model.learn(total_timesteps=n_steps*25, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
                new_model = f"linear_model_{new_iteration}"
                model.save(f"{models_dir}/{new_model}")
        if key == KEY2:
            print(f"Training count: {training_count}")
            LCDSetPrintf(12,LCD_Right_Print,f"Training count = {training_count}         ")
            while True:
                LCDMenu("Up", "Down", "-", "Back")
                LCDSetPrintf(12,LCD_Right_Print,f"Training count = {training_count}         ")
                key = KEYRead()
                if key == KEY1:
                    training_count += 1
                    print(f"Training count: {training_count}        ")
                elif key == KEY2:
                    if training_count > 1: training_count -= 1
                    print(f"Training count: {training_count}        ")
                elif key == KEY4:
                    break
        # If the user presses the back key, stop training and return to the main menu
        elif key == KEY4:
            LCDClear()
            break

# MAIN -------------------------------------------------------------------------------------------------------

def main():
    # Initialize the camera with QQVGA resolution (160x120)
    CAMInit(CAM_SETTING) 
    LCDImageStart(0,0,CAMWIDTH,CAMHEIGHT)
    LCDSetPrintf(0,LCD_Right_Print,"Linear Control          ")

    while True:
        LCDMenu("Train", "Test", "Load", "Quit")
        key = KEYRead()

        # Train the model
        if key == KEY1: 
            train()

        # Testing Menu
        elif key == KEY2: 
            while True:
                LCDMenu("Env", "check_env", "Reset", "Back")
                key = KEYRead()
                if key == KEY1: # Test the environment with random actions
                    test()
                elif key == KEY2: # Check the environment for any issues
                    check_env(env, warn=True)
                if key == KEY3: # Reset the environment
                    env.reset()
                elif key == KEY4: # Back to the main menu
                    break 
        
        # Load Menu
        elif key == KEY3: 
            model = None
            model_number = 0
            while True:
                LCDMenu("Test", "Train", "Select", "Back")
                key = KEYRead()
                LCDSetPrintf(9,LCD_Right_Print,f"Loaded Model = {model_number}       ")
                if key == KEY1: # Load a pre-trained model for testing
                    if model != "None": load_test(model)
                    else: print("Please select model before testing.")
                elif key == KEY2: # Load a pre-trained model to continue training
                    if model != "None": load_train(model, model_number)
                    else: print("Please select model before additional training.")
                elif key == KEY3:
                    model_list, most_recent_model = find_latest_model(models_dir)
                    if len(model_list) != 0: 
                        while(True):
                            LCDMenu("Up", "Down", "Latest", "Back")
                            key = KEYRead()
                            LCDSetPrintf(9,LCD_Right_Print,f"Selected Model = {model_number}        ")
                            if key == KEY1:
                                if model_number < len(model_list): 
                                    model_number +=1
                                    model = f"linear_model_{model_number}.zip"
                                LCDSetPrintf(9,LCD_Right_Print,f"Selected Model = {model_number}        ")
                            elif key == KEY2:
                                if model_number > 1: 
                                    model_number -= 1
                                    model = f"linear_model_{model_number}.zip"
                                LCDSetPrintf(9,LCD_Right_Print,f"Selected Model = {model_number}        ")
                            elif key == KEY3:
                                model_number = most_recent_model
                                model = f"linear_model_{model_number}.zip"
                                LCDSetPrintf(9,LCD_Right_Print,f"Selected Model = {model_number}        ")
                            elif key == KEY4:
                                break
                    else:
                        print("No models available, please train one to select.")

                elif key == KEY4: # Back to the main menu
                    VWSetSpeed(0,0)
                    break
        
        # Stop the program
        elif key == KEY4:
            break
            

main()

