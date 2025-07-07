#!/usr/bin/env python

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
log_dir = f"logs/Carolo/{version}"

# Algorithm used for training
algorithm = "PPO" 
policy_network = "CnnPolicy" # Policy network used for training

# Training parameters
learning_rate=0.0001
n_steps=2048

# Left lane coordinates for the simulation environment
left_lane = [
    (3990,400),(4000,733),(3562,400),(3571,733),
    (3171,400),(3171,733),(2838,400),(2848,733),
    (2448,400),(2448,733),(2095,410),(2095,733),
    (1676,400),(1657,743),(1267,410),(1238,733),
    (943,410),(962,733),(590,543),(771,790),
    (352,714),(600,943),(190,1010),(505,1057),
    (114,1229),(438,1248),(114,1619),(438,1619),
    (114,1971),(438,2000),(114,2362),(438,2371),
    (114,2705),(438,2733),(114,3086),(438,3086),
    (114,3381),(429,3381),(181,3705),(467,3590),
    (305,3981),(571,3829),(552,4257),(771,4019),
    (819,4438),(981,4171),(1171,4562),(1190,4248),
    (1619,4562),(1619,4248),(2000,4571),(2010,4248),
    (2410,4581),(2410,4267),(2762,4590),(2762,4267),
    (3190,4581),(3190,4257),(3619,4581),(3619,4257),
    (4029,4581),(4019,4267),(4524,4362),(4333,4133),
    (4800,4000),(4514,3895),(4876,3619),(4562,3610),
    (4790,3248),(4524,3390),(4581,2943),(4362,3162),
    (4333,2705),(4105,2905),(4124,2552),(3933,2733),
    (3895,2314),(3667,2524),(3676,2086),(3438,2305),
    (3438,1867),(3219,2076),(3257,1686),(3057,1905),
    (2981,1390),(2743,1610),(2476,1229),(2476,1514),
    (1943,1352),(2114,1619),(1590,1743),(1876,1867),
    (1486,2067),(1810,2105),(1476,2476),(1800,2476),
    (1476,2867),(1800,2857),(1581,3171),(1838,3029),
    (1743,3448),(2000,3238),(2124,3695),(2267,3400),
    (2590,3771),(2600,3448),(3095,3581),(2895,3343),
    (3371,3314),(3143,3086),(3610,3086),(3352,2867),
    (3705,2981),(3457,2762),(3933,2733),(3667,2524),
    (4124,2524),(3905,2305),(4390,2257),(4152,2048),
    (4695,1933),(4448,1724),(4876,1457),(4562,1400),
    (4762,924),(4486,1086),(4486,600),(4267,819),
    (3990,400),(4000,733)
]

# Centroids of the polygons for the simulation environment
left_centroids = [
    (3829,533,7),(3410,533,8),(3048,533,9),(2686,533,8),
    (2314,533,10),(1924,533,8),(1505,533,9),(1143,533,11),
    (848,571,30),(581,705,50),(400,886,70),(295,1105,82),
    (248,1381,97),(248,1762,97),(248,2133,97),(248,2505,98),
    (248,2857,97),(248,3200,97),(267,3486,109),(343,3762,124),
    (505,4010,141),(743,4229,154), (1000,4371,170),(1362,4429,-175),
    (1781,4438,-173),(2171,4448,-174),(2552,4457,-172),(2933,4457,-172),
    (3362,4448,-174),(3781,4448,-174),(4190,4381,-151),(4543,4143,-120),
    (4714,3819,-91),(4733,3495,-67),(4610,3190,-45),(4400,2924,-33),
    (4171,2714,-27),(3952,2533,-34),(3714,2305,-35),(3495,2076,-32),
    (3286,1876,-32),(3067,1648,-36),(2733,1419,-10),(2305,1381,21),
    (1895,1581,54),(1676,1905,81),(1619,2238,95),(1610,2629,97),
    (1638,2952,116),(1752,3210,132),(1981,3438,153),(2343,3600,178),
    (2762,3581,-153),(3124,3381,-126),(3371,3133,-125),(3552,2952,-109),
    (3695,2790,-123),(3914,2562,-124),(4143,2324,-127),(4419,2038,-126),
    (4657,1686,-104),(4714,1257,-70),(4552,867,-42),(4248,629,-14)
]

right_lane = [
    (3990,1048),(3990,733),(4133,1095),(4267,829),
    (4229,1210),(4486,1086),(4238,1381),(4562,1400),
    (4171,1571),(4448,1724),(3981,1743),(4219,1971),
    (3781,1971),(4010,2200),(3686,2076),(3905,2305),
    (3438,2305),(3667,2524),(3229,2552),(3457,2762),
    (2952,2848),(3200,3029),(2714,3076),(2895,3343),
    (2571,3124),(2600,3448),(2362,3095),(2267,3400),
    (2219,3000),(2000,3238),(2152,2905),(1838,3029),
    (2133,2857),(1800,2857),(2114,2505),(1800,2476),
    (2124,2114),(1810,2105),(2152,2038),(1876,1867),
    (2267,1924),(2114,1619),(2467,1867),(2476,1514),
    (2590,1886),(2743,1610),(2848,2181),(3057,1905),
    (3010,2305),(3219,2076),(3248,2533),(3438,2305),
    (3467,2733),(3667,2524),(3714,3000),(3933,2733),
    (3905,3162),(4105,2905),(4162,3410),(4362,3162),
    (4229,3514),(4524,3390),(4248,3629),(4562,3610),
    (4210,3781),(4514,3895),(4124,3867),(4333,4133),
    (3981,3943),(4019,4267),(3610,3952),(3619,4257),
    (3190,3952),(3190,4257),(2771,3952),(2762,4267),
    (2410,3952),(2410,4267),(2019,3943),(2010,4248),
    (1610,3943),(1619,4248),(1229,3933),(1190,4248),
    (1086,3876),(981,4171),(962,3771),(771,4019),
    (848,3638),(571,3829),(790,3486),(467,3590),
    (771,3371),(429,3381),(762,3095),(438,3086),
    (752,2733),(438,2733),(743,2381),(438,2371),
    (752,2019),(438,2000),(752,1638),(438,1619),
    (762,1305),(438,1248),(781,1210),(505,1057),
    (829,1162),(600,943),(905,1105),(771,790),
    (1000,1057),(962,733),(1229,1029),(1238,733),
    (1667,1029),(1657,743),(2095,1029),(2095,733),
    (2438,1038),(2448,733),(2848,1048),(2848,733),
    (3162,1048),(3171,733),(3571,1038),(3571,733),
    (4000,1057),(4000,733)
]

right_centroids = [
    (4076,943,171),(4257,1057,138),(4352,1257,110),(4333,1486,82),
    (4200,1714,55),(4000,1924,57),(3838,2105,63),(3676,2257,52),
    (3448,2486,59),(3219,2743,54),(2943,3029,53),(2705,3210,33),
    (2476,3238,3),(2248,3162,-17),(2086,3029,-34),(2019,2905,-42),
    (2000,2705,-79),(2000,2333,-82),(2019,2048,-87),(2114,1895,-122),
    (2324,1762,-154),(2552,1743,178),(2771,1886,139),(3000,2124,150),
    (3190,2305,143),(3419,2524,145),(3657,2743,143),(3876,2952,148),
    (4095,3152,141),(4295,3371,135),(4362,3524,114),(4362,3705,90),
    (4286,3886,63),(4124,4010,38),(3848,4067,9),(3448,4067,8),
    (3029,4076,7),(2629,4076,9),(2257,4067,7),(1857,4057,9),
    (1457,4057,8),(1143,4029,-2),(981,3943,-23),(829,3800,-29),
    (705,3629,-50),(657,3457,-55),(638,3257,-77),(638,2943,-78),
    (629,2590,-80),(629,2229,-81),(629,1857,-82),(629,1486,-82),
    (657,1219,-81),(705,1114,-99),(790,1029,-120),(914,952,-139),
    (1086,914,-167),(1410,905,-176),(1838,905,-175),(2238,905,-175),
    (2610,914,-174),(2981,914,-173),(3333,914,-173),(3743,914,-176)
]

# Sign simulator positions
StopSign1 = (3722,3129,315)
StopSign2 = (3679,1994,135)
SpeedLimit10Sign1 = (3182,4713,0)
SpeedLimit10Sign2 = (3178,1153,0)
SpeedLimitSign1 = (3178,285,180)
SpeedLimitSign2 = (3182,3825,180)

left_lane_30speedlimit = [1,28]
right_lane_30speedlimit = [35,62]
left_lane_stop = 55
right_lane_stop = 6

# GYMNASIUM ENVIRONMENT --------------------------------------------------------------------------------------------------------

# Custom environment for the robot simulation using OpenAI Gymnasium
class EyeSimEnv(gym.Env):

    def __init__(self, control_mode='angular', angular_model=None):
        super(EyeSimEnv, self).__init__()
        self.control_mode = control_mode

        if control_mode == 'linear':
            self.action_space = gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)
        elif control_mode == 'angular':
            self.action_space = gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(CAMHEIGHT,CAMWIDTH,3), dtype=np.uint8)

        # Initialize variables
        self.stop_reached = False
        self.stop_time = time.time()
        self.current_polygon = np.array([])
        self.current_centroid = 1
        self.next_polygon = np.array([])
        self.next_centroid = 2
        self.speed_limit = 1.0
        self.current_lane = left_lane
        self.current_centroids = left_centroids
        self.speed_limit_centroids = left_lane_30speedlimit
        self.stop_centroid = left_lane_stop
        self.angular_model = angular_model

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.eyesim_reset()
        observation = self.eyesim_get_observation()
        self.last_obs = observation.copy()  # Save for angular model to use
        info = {}
        return observation, info

    def step(self, action):
        object_check()

        if self.control_mode == 'linear':
            linear = action[0]
            angular = 0.0
            if self.angular_model:
                angular_action, _ = self.angular_model.predict(self.last_obs, deterministic=True)
                angular = angular_action[0]
        elif self.control_mode == 'angular':
            linear = 1
            angular = action[0]
        else:
            angular, linear = action[0], action[1]

        result1, result2 = self.eyesim_get_position()
        self.eyesim_set_robot_speed(linear, angular)

        observation = self.eyesim_get_observation()
        self.last_obs = observation.copy()  # Save for angular model to use
        drive_reward = self.calculate_drive_reward(result1, result2)
        speed_reward = self.calculate_speed_reward(linear) if self.control_mode != 'angular' else 0.0

        reward = drive_reward + speed_reward if drive_reward != -10.0 else drive_reward
        done = self.is_done(result1, result2)
        truncated = False

        info = {
            "Position_reward": drive_reward,
            "Speed_reward": speed_reward,
            "Stop_Reached": self.stop_reached,
            "Current_Centroid": self.current_centroid,
            "Current_Lane": "left_lane" if self.current_lane == left_lane else "right_lane"
        }

        return observation, reward, done, truncated, info

    def calculate_drive_reward(self, result1, result2):
        # If the robot is inside the next polygon, return a positive reward
        if result2 > 0:
            # update previous polygon to current polygon and repeat 
            self.current_centroid += 1
            self.current_centroid %= len(self.current_centroids) # No need to change since both centroid lists are the same length
            self.update_polygon()
            return 5.0
        # If the robot is inside neither  polygon, return a big negative reward
        if result1 < 0 and result2 < 0: 
            return -10.0
        # If the robot is inside the current polygon, return no reward
        else:
            return 0.0

    def calculate_speed_reward(self, linear):
        # Calculate the speed of the robot based on the angular and linear speed
        speed = linear

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
                if speed == 0.0:
                    self.stop_reached = True
                    self.stop_time = time.time()
            else:
                self.speed_limit = 0.5
        else:
            self.speed_limit = 0.5

        # Smooth reward function: reward is higher when speed is close to target
        speed_error = abs(speed - self.speed_limit)
        penalty = 0.0
        if speed_error > 0.05: # If the speed is lower than the speed limit with tolerance
            penalty = -speed_error # Penalty is proportional to the speed error
        elif speed_error > 0.2: # If the speed is higher than the speed limit with tolerance
            penalty = -speed_error

        return 0.2 + penalty # Return the speed reward, which is 0.2 plus the penaltys

    def is_done(self, result1, result2):
        # Determine if the robot left all allowable polygons
        if (result1 == -1 and result2 == -1) or self.lap_check(): return True
        else: return False

    # INCLUDED EYESIM HELPER FUNCTIONS --------------------------------------------------------------------------------------------------

    def lap_check(self):
        # If the robot has completed a lap, switch sides and reset the current centroid
        if self.current_centroid == 63:
            if self.current_lane == left_lane:
                self.current_lane = right_lane
                self.speed_limit_centroids = right_lane_30speedlimit
                self.current_centroids = right_centroids
                self.current_centroid = 0
                return True
            else:
                self.current_lane = left_lane
                self.speed_limit_centroids = left_lane_30speedlimit
                self.current_centroids = left_centroids
                self.current_centroid = 0
                return True
        return False

    # Function to set the speed of the robot based on the action taken
    def eyesim_set_robot_speed(self, linear, angular): 
        # Set the speed of the robot based on the action taken
        linear_speed = 300
        angular_speed = 50
        VWSetSpeed(round(linear_speed*linear),round(angular_speed*angular)) # Set the speed of the robot
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

    def update_polygon(self):
        # Update the current and next polygon based on the current centroid
        self.current_polygon = np.array([
            self.current_lane[self.current_centroid*2],
            self.current_lane[self.current_centroid*2+1],
            self.current_lane[(self.current_centroid*2+3)],
            self.current_lane[(self.current_centroid*2+2)]
        ], np.int32)

        self.next_centroid = (self.current_centroid + 1)% len(self.current_centroids)

        self.next_polygon = np.array([
            self.current_lane[self.next_centroid*2],
            self.current_lane[self.next_centroid*2+1],
            self.current_lane[(self.next_centroid*2+3)],
            self.current_lane[(self.next_centroid*2+2)]
        ], np.int32)

    # Function to get the distance to the red peak
    def eyesim_get_position(self): 
        [x,y,_,_] = SIMGetRobot(1)
        point = (x.value, y.value)
        
        self.update_polygon()

        # Reshape the polygon points
        self.current_polygon = self.current_polygon.reshape((-1, 1, 2))

        # Reshape the polygon points
        self.next_polygon = self.next_polygon.reshape((-1, 1, 2))

        # Check if the point is inside the polygon
        current_result = cv2.pointPolygonTest(self.current_polygon, point, False)

        # Check if the point is inside the polygon
        next_result = cv2.pointPolygonTest(self.next_polygon, point, False)

        # If the point is inside the polygon return
        return current_result, next_result

    # Function to reset the robot and can positions in the simulation
    def eyesim_reset(self): 
        # Stop robot movement
        VWSetSpeed(0,0)
        if self.current_centroid != 0:
            self.current_centroid -= 1
        # # Position the robot in the simulation
        x,y,phi = self.current_centroids[self.current_centroid]

        SIMSetRobot(1,x,y,10,phi+180) # Add 180 degrees to the angle to flip robot into correct direction

        self.update_polygon()

# ADDITIONAL HELPER FUNCTIONS -------------------------------------------------------------------------------------------------

# Function to check if the objects are in the correct position and set them if not
def object_check():
    if SIMGetObject(2)[0].value != StopSign1[0] or SIMGetObject(2)[1].value != StopSign1[1] or SIMGetObject(2)[2].value != StopSign1[2]: 
        SIMSetObject(2, StopSign1[0], StopSign1[1], 10, StopSign1[2]+90)
    if SIMGetObject(3)[0].value != StopSign2[0] or SIMGetObject(3)[1].value != StopSign2[1] or SIMGetObject(3)[2].value != StopSign2[2]: 
        SIMSetObject(3, StopSign2[0], StopSign2[1], 10, StopSign2[2]+90)
    if SIMGetObject(4)[0].value != SpeedLimit10Sign1[0] or SIMGetObject(4)[1].value != SpeedLimit10Sign1[1] or SIMGetObject(4)[2].value != SpeedLimit10Sign1[2]: 
        SIMSetObject(4, SpeedLimit10Sign1[0], SpeedLimit10Sign1[1], 10, SpeedLimit10Sign1[2])
    if SIMGetObject(5)[0].value != SpeedLimit10Sign2[0] or SIMGetObject(5)[1].value != SpeedLimit10Sign2[1] or SIMGetObject(5)[2].value != SpeedLimit10Sign2[2]: 
        SIMSetObject(5, SpeedLimit10Sign2[0], SpeedLimit10Sign2[1], 10, SpeedLimit10Sign2[2])
    if SIMGetObject(6)[0].value != SpeedLimitSign1[0] or SIMGetObject(6)[1].value != SpeedLimitSign1[1] or SIMGetObject(6)[2].value != SpeedLimitSign1[2]: 
        SIMSetObject(6, SpeedLimitSign1[0], SpeedLimitSign1[1], 10, SpeedLimitSign1[2])
    if SIMGetObject(7)[0].value != SpeedLimitSign2[0] or SIMGetObject(7)[1].value != SpeedLimitSign2[1] or SIMGetObject(7)[2].value != SpeedLimitSign2[2]: 
        SIMSetObject(7, SpeedLimitSign2[0], SpeedLimitSign2[1], 10, SpeedLimitSign2[2])

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

