#!/usr/bin/env python

# TO-DO --------------------------------------------------------------------------------------------------------------

"""

change reward system to score = exp(((-ln(0.01)/range)x)-max_reward

reduce to one observation stack and only current speed then update code and feature extraction

Eval callback saves best model based on mean reward ## look into this ##
eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir,
                             log_path=logdir, eval_freq=50_000, n_eval_episodes=5, deterministic=Tru

add features to LCD to show current model being trained, and how many left to do

slightly increase ramp range
"""

# IMPORTS ------------------------------------------------------------------------------------------------------------

import time

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
learning_rate = 0.0001
n_steps = 1024
batch_size = 128 
ent_coef = 0.05      
clip_range = 0.2
max_grad_norm = 0.5
use_sde = False
sde_sample_freq = -1

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
buffer = 350
stop_buffer = 100

# Assigned values for speed limits
basespeedlimit = 1.0
speedlimit10 = 0.5
speedlimit30 = 0.7
maxspeedlimit = 1.5

# Robot ID in the simulation
robot_id = 1


LCD_Right_Print = 60

# GYMNASIUM ENVIRONMENT --------------------------------------------------------------------------------------------------------

# Custom environment for the robot simulation using OpenAI Gymnasium
class EyeSimEnv(gym.Env):
    
    def __init__(self):
        super(EyeSimEnv, self).__init__()
        # Float action space for robot linear speed, range from 0.0 to 1.0
        self.action_space = gym.spaces.Box(low=np.array([-0.1], dtype=np.float32), high=np.array([0.1], dtype=np.float32), dtype=np.float32)

        self.history_length = 4

        # Image observation space, 1 channels (Gray), 120x160 pixels
        self.observation_space = gym.spaces.Dict({
            "image":     gym.spaces.Box(low=0, high=255, shape=(1, CAMHEIGHT, CAMWIDTH), dtype=np.uint8),
            "Current_Speed": gym.spaces.Box(low=0.0, high=maxspeedlimit, shape=(1,), dtype=np.float32),
            "image_his": gym.spaces.Box(low=0, high=255, shape=(self.history_length, CAMHEIGHT, CAMWIDTH), dtype=np.uint8),
            "speed_his": gym.spaces.Box(low=0.0, high=maxspeedlimit, shape=(self.history_length,), dtype=np.float32)
        })
        
        # Initialize class variables
        self.track = 1
        self.speedlimit10_position = randint(sign_x_positions[0], sign_x_positions[1]) # Randomly select a position for the 10 limit sign
        self.speedlimit30_position = randint(sign_x_positions[0], sign_x_positions[1]) # Randomly select a position for the 30 limit sign
        self.timesteps = 0
        self.speed_reached = False
        self.completed_stop = False
        self.stop_reached = False
        self.additional_buffer = 0.0
        self.stop_position = 0.0
        self.stop_time = 0.0

        self.image_his = np.zeros((self.history_length, CAMHEIGHT, CAMWIDTH), dtype=np.uint8)
        self.speed_his = np.zeros((self.history_length,), dtype=np.float32)
        self.Current_Speed = np.array([1.0], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.eyesim_reset()
        frame = self.eyesim_get_observation() 
        self.image_his[:] = np.repeat(frame[0:1, :, :], self.history_length, axis=0)
        self.speed_his.fill(self.Current_Speed[0])

        observation = {"image":frame,
                    "Current_Speed": self.Current_Speed,
                    "image_his": self.image_his,
                    "speed_his": self.speed_his
                    }
        info = {}
        return observation, info

    def step(self, action):
        reward = 0.0
        # Print action to LCD display
        LCDSetPrintf(1,LCD_Right_Print,f"Action: {action[0]:.2f}     ")

        self.timesteps += 1
        if self.timesteps == n_steps:
            self.timesteps = 0
            VWSetSpeed(0,0)
            
        # Determine new current speed of the robot based on action
        self.Current_Speed = np.array([np.clip(self.Current_Speed[0] + action[0], 0.0, maxspeedlimit)], dtype=np.float32)
        speed = self.Current_Speed[0].__float__()

        # Set robot linear and angular speed based on action
        self.eyesim_set_robot_speed(speed) 

        # Get current position of robot
        position = self.eyesim_get_position()

        # Obtain new image
        new_frame = self.eyesim_get_observation()
        # Shift history
        self.image_his = np.roll(self.image_his, shift=-1, axis=0)
        self.image_his[-1] = new_frame[0]                   # drop channel
        self.speed_his = np.roll(self.speed_his, shift=-1)
        self.speed_his[-1] = self.Current_Speed[0]

        # Calculate reward based on position
        reward = self.calculate_speed_reward(speed, position) # Calculate the speed reward based on the speed

        # Determine if the episode is donea
        done = self.is_done(position)
        truncated = False

        # Build observation
        observation = {
            "image": new_frame,                              # (1, H, W), uint8
            "Current_Speed": self.Current_Speed.astype(np.float32),
            "image_his": self.image_his,                    # (T, H, W), uint8
            "speed_his": self.speed_his.astype(np.float32)
        }
        
        # Create info dictionary to store additional information
        info = {"current velocity": self.Current_Speed[0]}


        return observation, reward, done, truncated, info
    
    def calculate_speed_reward(self, linear_speed, position):
        LCDSetPrintf(2,LCD_Right_Print, f"Speed: {self.Current_Speed[0]:.2f}    ")
        # Initialize score for speed control
        score = 0.0 

        # SPEED LIMIT 30 SIGN
        if self.track == 1:
            target_speed = speedlimit30
            LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {target_speed}    ")

            # --- Gradual ramp towards target before the sign ---
            if self.speedlimit30_position - buffer <= position <= self.speedlimit30_position:
                ramp_factor = (position - (self.speedlimit30_position - buffer)) / buffer
                gradual_target = basespeedlimit + ramp_factor * (target_speed - basespeedlimit)

                LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {gradual_target:.2f}    ")
                score = reward_calculation(0.5, gradual_target, linear_speed)

            # --- After passing the sign, enforce speed limit ---
            elif position > self.speedlimit30_position:
                threshold = 0.05  # allowable deviation
                if abs(linear_speed - target_speed) <= threshold:
                    if not self.speed_reached:
                        self.speed_reached = True
                        score += 2.0  # big bonus for reaching the limit
                    else:
                        score += 0.5  # small bonus for holding the speed
                else:
                    if self.speed_reached:
                        self.speed_reached = False
                        score -= 2.0  # penalty for drifting from limit
                    else:
                        score = reward_calculation(0.5, target_speed, linear_speed)

            # --- Before ramp zone, stick to base speed ---
            else:
                target_speed = basespeedlimit
                LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {target_speed}    ")
                score = reward_calculation(0.5, target_speed, linear_speed)

        # SPEED LIMIT 10 SIGN
        elif self.track == 2:
            target_speed = speedlimit10
            LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {target_speed}    ")

            # --- Gradual ramp towards target before the sign ---
            if self.speedlimit10_position - buffer <= position <= self.speedlimit10_position:
                ramp_factor = (position - (self.speedlimit10_position - buffer)) / buffer
                gradual_target = basespeedlimit + ramp_factor * (target_speed - basespeedlimit)

                LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {gradual_target:.2f}    ")
                score = reward_calculation(0.5, gradual_target, linear_speed)

            # --- After passing the sign, enforce speed limit ---
            elif position > self.speedlimit10_position:
                threshold = 0.05  # allowable deviation
                if abs(linear_speed - target_speed) <= threshold:
                    if not self.speed_reached:
                        self.speed_reached = True
                        score += 2.0  # big bonus for reaching the limit
                    else:
                        score += 0.5  # small bonus for holding the speed
                else:
                    if self.speed_reached:
                        self.speed_reached = False
                        score -= 2.0  # penalty for drifting from limit
                    else:
                        score = reward_calculation(0.5, target_speed, linear_speed)

            # --- Before ramp zone, stick to base speed ---
            else:
                target_speed = basespeedlimit
                LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {target_speed}    ")
                score = reward_calculation(0.5, target_speed, linear_speed)

        # STOP SIGN
        elif self.track == 3: 
            target_speed = 0.05
            LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {target_speed}    ")

            # --- Deceleration zone before full stop ---
            if (sign_x_positions[2] - buffer) <= position < (sign_x_positions[2] - stop_buffer):
                ramp_factor = (position - (sign_x_positions[2] - buffer)) / (buffer - stop_buffer)
                gradual_target = basespeedlimit + ramp_factor * (target_speed - basespeedlimit)

                LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {gradual_target:.2f}    ")
                score = reward_calculation(0.5, gradual_target, linear_speed)

            # --- Stop enforcement zone (must hold for 2s) ---
            elif (sign_x_positions[2] - stop_buffer) <= position <= sign_x_positions[2] + self.additional_buffer:
                target_speed = 0.0
                LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {target_speed}    ")

                if not self.completed_stop:
                    # Check if agent has reached near-zero speed
                    if abs(linear_speed - target_speed) < 0.05:
                        if not self.stop_reached:
                            self.stop_reached = True
                            self.stop_time = time.time()
                            score += 2.0  # bonus for stopping
                        else:
                            # Holding at stop
                            elapsed = time.time() - self.stop_time
                            LCDSetPrintf(5, LCD_Right_Print, f"Stop Time: {elapsed:.2f}s   ")

                            if elapsed >= 2.0:
                                # Full stop completed
                                self.stop_reached = False
                                self.completed_stop = True
                                self.stop_position = position
                                self.additional_buffer = (self.stop_position - sign_x_positions[2] + buffer)
                                score += 5.0
                                LCDSetPrintf(5, LCD_Right_Print, "Stop Completed        ")
                            else:
                                score += 0.5  # reward holding each step
                    else:
                        # Broke stop hold → reset
                        if self.stop_reached:
                            self.stop_reached = False
                            self.stop_time = 0.0
                            score -= 2.0
                        else:
                            # Encourage slowing down toward stop
                            score = reward_calculation(0.5, target_speed, linear_speed)

                # --- After full stop completed, resume ---
                else:
                    target_speed = basespeedlimit
                    ramp_factor = (position - self.stop_position) / buffer
                    gradual_target = 0.0 + ramp_factor * (target_speed - 0.0)

                    LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {gradual_target:.2f}    ")
                    score = reward_calculation(0.5, gradual_target, linear_speed)

            # --- Before decel zone & after stop sign → normal driving ---
            elif self.completed_stop == False and position > sign_x_positions[2]:
                score += -5.0 # Heavy penalty for not stopping
            
            else:
                target_speed = basespeedlimit
                LCDSetPrintf(3, LCD_Right_Print, f"Speedlimit: {target_speed}    ")
                score = reward_calculation(0.5, target_speed, linear_speed)



        LCDSetPrintf(4,LCD_Right_Print,f"Score: {round(score,2):.2f}    ")
        return score # Return the speed score
        

    def is_done(self, position):
        # If the robot has reached the end of the track, return True

        if self.track == 1 and position >= (self.speedlimit30_position + buffer):
            self.Current_Speed = np.array([basespeedlimit], dtype=np.float32) 
            return True
        if self.track == 2 and position >= (self.speedlimit10_position + buffer):
            self.Current_Speed = np.array([basespeedlimit], dtype=np.float32) 
            return True
        elif self.track == 3 and position >= (sign_x_positions[2] + buffer + self.additional_buffer) or self.track == 3 and position >= sign_x_positions[2] and self.completed_stop == False: 
            self.Current_Speed = np.array([basespeedlimit], dtype=np.float32) 
            # Reset completed stop flags
            self.completed_stop = False
            self.stop_reached = False
            self.additional_buffer = 0.0
            self.stop_position = 0.0
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
        img = CAMGetGray() 
    
        # Process image
        processed_img = image_processing(img) 

        # Optional: Display the processed image on the LCD screen
        display_img = processed_img.ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
        LCDImageGray(display_img)

        return processed_img

    # Function to reset the robot and can positions in the simulation "C:\Users\noah\AppData\Local\Programs\EyeSim\EyeSim.exe"
    def eyesim_reset(self): 
        # Stop robot movement
        VWSetSpeed(0,0)

        # Randomly select a track for the robot to follow
        self.track = (self.track % 3) + 1 # Cycle through tracks 1, 2, and 3
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

def reward_calculation(max_reward, target_speed, current_speed):
    range = 0
    if current_speed > target_speed: range = maxspeedlimit-target_speed
    if current_speed < target_speed: range = target_speed
    if current_speed == target_speed: return max_reward
    score = max_reward - 2*(abs(target_speed-current_speed)/range)*max_reward
    return score

# CNN EXTRACTOR ---------------------------------------------------------------------------------------------------------

class EyeSimExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Shapes
        C, H, W = observation_space.spaces["image"].shape
        T = observation_space.spaces["image_his"].shape[0]

        # CNN for current image
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute CNN output size
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, C, H, W)).shape[1]

        # CNN for image history (process each frame independently, then flatten)
        self.cnn_his = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            n_flat_his = self.cnn_his(th.zeros(1, 1, H, W)).shape[1]
        n_flat_his *= T  # total features from all history frames

        # MLP for speed
        self.speed_net = nn.Sequential(
            nn.Linear(1 + T, 32),  # Current_Speed + speed_his
            nn.ReLU()
        )

        # Final fusion
        self.fc = nn.Sequential(
            nn.Linear(n_flat + n_flat_his + 32, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: dict) -> th.Tensor:
        # Current image: (B, C, H, W)
        img = obs["image"].float()  # make sure it's float
        img_feat = self.cnn(img)

        # Image history: (B, T, H, W) -> (B*T, 1, H, W)
        B, T, H, W = obs["image_his"].shape
        img_his = obs["image_his"].unsqueeze(2).float()  # (B, T, 1, H, W)
        img_his = img_his.view(B * T, 1, H, W)
        his_feat = self.cnn_his(img_his)
        his_feat = his_feat.view(B, -1)  # (B, T*features_per_frame)

        # Speed
        speed_feat = self.speed_net(th.cat([obs["Current_Speed"].float(), obs["speed_his"].float()], dim=1))

        # Combine all features
        return self.fc(th.cat([img_feat, his_feat, speed_feat], dim=1))

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
        obs, reward, done, _, _= env.step(action)
        print(f"Reward: {reward:.2f}, Action: {action[0]:.2f}, Done: {done}, Current Velocity: {obs['Current_Speed'][0]:.2f}, speed_history: {obs['speed_his'][0]}")
    
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

    # Define the PPO model with the specified parameters
    model = PPO(policy_network, env=env, verbose=1, tensorboard_log=logdir, n_steps=n_steps, policy_kwargs=policy_kwargs,
                learning_rate=learning_rate, batch_size=batch_size, ent_coef=ent_coef, 
                clip_range=clip_range, max_grad_norm=max_grad_norm, use_sde=use_sde, sde_sample_freq=sde_sample_freq)

    # Train the model
    for i in range(1,21): # Train the model
        model.learn(total_timesteps=n_steps*25, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
        model.save(f"{models_dir}/linear_model_{i}")



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
        print(f"Reward: {reward}, Action: {action}")

        # End testing if user presses the stop key
        if key == KEY4: 
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
            for i in range (0,training_count):
                new_iteration += 1
                LCDSetPrintf(10,LCD_Right_Print,f"New Model = {new_iteration}")
                LCDSetPrintf(11,LCD_Right_Print,f"Remaining = {training_count-i}")
                model.learn(total_timesteps=51200, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
                new_model = f"linear_model_{new_iteration}"
                model.save(f"{models_dir}/{new_model}")
        if key == KEY2:
            print(f"Training count: {training_count}")
            LCDSetPrintf(12,LCD_Right_Print,f"Training count = {training_count}")
            while True:
                LCDMenu("Up", "Down", "-", "Back")
                LCDSetPrintf(12,LCD_Right_Print,f"Training count = {training_count}")
                key = KEYRead()
                if key == KEY1:
                    training_count += 1
                    print(f"Training count: {training_count}")
                elif key == KEY2:
                    if training_count > 1: training_count -= 1
                    print(f"Training count: {training_count}")
                elif key == KEY4:
                    break
        # If the user presses the back key, stop training and return to the main menu
        elif key == KEY4:
            break

# MAIN -------------------------------------------------------------------------------------------------------

def main():
    # Initialize the camera with QQVGA resolution (160x120)
    CAMInit(CAM_SETTING) 
    LCDImageStart(0,0,CAMWIDTH,CAMHEIGHT)
    LCDSetPrintf(0,LCD_Right_Print,"Linear Control")

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
            model = "None"
            model_number = 0
            while True:
                LCDMenu("Test", "Train", "Select", "Back")
                key = KEYRead()
                LCDSetPrintf(9,LCD_Right_Print,f"Loaded Model = {model_number} ")
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
                            LCDSetPrintf(9,LCD_Right_Print,f"Loaded Model = {model_number}")
                            if key == KEY1:
                                if model_number < len(model_list): 
                                    model_number +=1
                                    model = f"linear_model_{model_number}.zip"
                                print(f"Selected model: {model_number}")
                            elif key == KEY2:
                                if model_number > 1: 
                                    model_number -= 1
                                    model = f"linear_model_{model_number}.zip"
                                print(f"Selected model: {model_number}")
                            elif key == KEY3:
                                model_number = most_recent_model
                                model = f"linear_model_{model_number}.zip"
                                print(f"Selected model: {model_number}")
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

