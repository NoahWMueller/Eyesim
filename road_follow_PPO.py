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
from gymnasium import spaces
from stable_baselines3 import PPO

# GLOBAL VARIABLES ---------------------------------------------------------------------------------------------------

# Constants for camera settings
CAMWIDTH = 160
CAMHEIGHT = 120
DESIRED_CAMHEIGHT = 120

# Algorithm used for training
algorithm = "PPO" 

# Current version of the code for saving models and logs
version = 1.5

# Directory paths for saving models and logs
models_dir = f"models/Carolo/{version}"
logdir = f"logs/Carolo/{version}"

# Left lane coordinates for the simulation environment
left_lane = [
    (3990,400),(4000,733),(3562,400),(3571,733),(3171,400),
    (3171,733),(2838,400),(2848,733),(2448,400),(2448,733),
    (2095,410),(2095,733),(1676,400),(1657,743),(1267,410),
    (1238,733),(943,410),(962,733),(590,543),(771,790),
    (352,714),(600,943),(190,1010),(505,1057),(114,1229),
    (438,1248),(114,1619),(438,1619),(114,1971),(438,2000),
    (114,2362),(438,2371),(114,2705),(438,2733),(114,3086),
    (438,3086),(114,3381),(429,3381),(181,3705),(467,3590),
    (305,3981),(571,3829),(552,4257),(771,4019),(819,4438),
    (981,4171),(1171,4562),(1190,4248),(1619,4562),(1619,4248),
    (2000,4571),(2010,4248),(2410,4581),(2410,4267),(2762,4590),
    (2762,4267),(3190,4581),(3190,4257),(3619,4581),(3619,4257),
    (4029,4581),(4019,4267),(4524,4362),(4333,4133),(4800,4000),
    (4514,3895),(4876,3619),(4562,3610),(4790,3248),(4524,3390),
    (4581,2943),(4362,3162),(4333,2705),(4105,2905),(4124,2552),
    (3933,2733),(3895,2314),(3667,2524),(3676,2086),(3438,2305),
    (3438,1867),(3219,2076),(3257,1686),(3057,1905),(2981,1390),
    (2743,1610),(2476,1229),(2476,1514),(1943,1352),(2114,1619),
    (1590,1743),(1876,1867),(1486,2067),(1810,2105),(1476,2476),
    (1800,2476),(1476,2867),(1800,2857),(1581,3171),(1838,3029),
    (1743,3448),(2000,3238),(2124,3695),(2267,3400),(2590,3771),
    (2600,3448),(3095,3581),(2895,3343),(3371,3314),(3143,3086),
    (3610,3086),(3352,2867),(3705,2981),(3457,2762),(3933,2733),
    (3667,2524),(4124,2524),(3905,2305),(4390,2257),(4152,2048),
    (4695,1933),(4448,1724),(4876,1457),(4562,1400),(4762,924),
    (4486,1086),(4486,600),(4267,819),(3990,400),(4000,733)

]

# Centroids of the polygons for the simulation environment
centroids = [
    (3829,533,7),(3410,533,8),(3048,533,9),(2686,533,8),(2314,533,10),
    (1924,533,8),(1505,533,9),(1143,533,11),(848,571,30),(581,705,50),
    (400,886,70),(295,1105,82),(248,1381,97),(248,1762,97),(248,2133,97),
    (248,2505,98),(248,2857,97),(248,3200,97),(267,3486,109),(343,3762,124),
    (505,4010,141),(743,4229,154),(1000,4371,170),(1362,4429,-175),(1781,4438,-173),
    (2171,4448,-174),(2552,4457,-172),(2933,4457,-172),(3362,4448,-174),(3781,4448,-174),
    (4190,4381,-151),(4543,4143,-120),(4714,3819,-91),(4733,3495,-67),(4610,3190,-45),
    (4400,2924,-33),(4171,2714,-27),(3952,2533,-34),(3714,2305,-35),(3495,2076,-32),
    (3286,1876,-32),(3067,1648,-36),(2733,1419,-10),(2305,1381,21),(1895,1581,54),
    (1676,1905,81),(1619,2238,95),(1610,2629,97),(1638,2952,116),(1752,3210,132),
    (1981,3438,153),(2343,3600,178),(2762,3581,-153),(3124,3381,-126),(3371,3133,-125),
    (3552,2952,-109),(3695,2790,-123),(3914,2562,-124),(4143,2324,-127),(4419,2038,-126),
    (4657,1686,-104),(4714,1257,-70),(4552,867,-42),(4248,629,-14)
]

stop_centroid = 55

# GYMNASIUNM ENVIRONMENT --------------------------------------------------------------------------------------------------------

# Custom environment for the robot simulation using OpenAI Gymnasium
class EyeSimEnv(gym.Env):
    
    def __init__(self):
        super(EyeSimEnv, self).__init__()
        # Define the lower and upper bounds
        low = np.array([-1.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32) # Float action space for robot angular speed, range from -1 to 1
        self.observation_space = spaces.Box(low=0, high=255, shape=(DESIRED_CAMHEIGHT,CAMWIDTH,3), dtype=np.uint8) # Image observation space, 3 channels (RGB), 60x160 pixels
        
        # Initialize variables
        self.stop_reached = False
        self.stop_time = time.time()
        self.current_polygon = np.array([])
        self.current_centroid = randint(0, len(centroids)-1)
        self.next_polygon = np.array([])
        self.next_centroid = 0
        self.speed_limit = 1.0  # Speed limit for the robot, 1/100 of the limit

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.eyesim_reset()
        observation = self.eyesim_get_observation()
        info = {}
        return observation, info

    def step(self, action):
        angular, linear = action[0], action[1] # linear and angular action

        # Determines if robot is inside left lane or has gotten lost
        result1, result2 = self.eyesim_get_position() 
        
        # Set robot linear and angular speed based on action
        self.eyesim_set_robot_speed(linear, angular) 

        # Read image from camera
        observation = self.eyesim_get_observation()

        # Calculate reward based on position
        drive_reward = self.calculate_drive_reward(result1, result2) # Calculate the drive reward based on the position
        speed_reward = self.calculate_speed_reward(linear) # Calculate the speed reward based on the speed
        if drive_reward != -10.0: # If the robot is inside the next polygon, update the current centroid
            reward = drive_reward + speed_reward # Total reward is the sum of the drive and speed rewards
        else:
            reward = drive_reward

        # Determine if the episode is done
        done = self.is_done(result1,result2)

        # Truncated is not used in this case, but included for compatibility with gym API
        truncated = False

        # Create info dictionary to store additional information
        info = {"Position_reward": drive_reward, "Speed_reward": speed_reward, "Stop_Reached": self.stop_reached, "Current_Centroid": self.current_centroid}

        return observation, reward, done, truncated, info

    def calculate_drive_reward(self, result1, result2):
        # If the robot is inside the next polygon, return a positive reward
        if result2 > 0:
            # update previous polygon to current polygon and repeat 
            self.current_centroid += 1
            self.current_centroid %= len(centroids)
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
        reward = 0.0

        # If the robot has reached a stop, check if it has been stopped for more than 3 seconds before resetting the stop flag
        if self.stop_reached and time.time() - self.stop_time >= 3.0: 
            self.stop_reached = False
            self.stop_time = 0.0

        # Speed limit logic
        if 0 < self.current_centroid < 30:
            self.speed_limit = 0.75
        elif self.current_centroid == stop_centroid:
            if not self.stop_reached:
                self.speed_limit = 0.0
                if speed == 0.0:
                    self.stop_reached = True
                    self.stop_time = time.time()
        else:
            self.speed_limit = 0.5

        # Smooth reward function: reward is higher when speed is close to target
        speed_error = abs(speed - self.speed_limit)
        penalty = 0.0
        if speed_error > 0.05 and speed_error <= 0.25: # If the speed is lower than the speed limit with tolerance
            penalty = -speed_error * 2 # Penalty is proportional to the speed error
        else:
            penalty = -0.5
        reward = 0.5 + penalty

        return reward

    def is_done(self, result1, result2):
        # Determine if the robot left all allowable polygons
        if result1 == -1 and result2 == -1: return True
        else: return False

    # INCLUDED EYESIM HELPER FUNCTIONS --------------------------------------------------------------------------------------------------

    # Function to set the speed of the robot based on the action taken
    def eyesim_set_robot_speed(self, linear, angular): 
        # Set the speed of the robot based on the action taken
        linear_speed = 100
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
            left_lane[self.current_centroid*2],
            left_lane[self.current_centroid*2+1],
            left_lane[(self.current_centroid*2+3)],
            left_lane[(self.current_centroid*2+2)]
        ], np.int32)

        self.next_centroid = (self.current_centroid + 1)% len(centroids)

        self.next_polygon = np.array([
            left_lane[self.next_centroid*2],
            left_lane[self.next_centroid*2+1],
            left_lane[(self.next_centroid*2+3)],
            left_lane[(self.next_centroid*2+2)]
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

        # # Position the robot in the simulation
        x,y,phi = centroids[self.current_centroid]

        SIMSetRobot(1,x,y,10,phi+180) # Add 180 degrees to the angle to flip robot into correct direction

        # Reset object positions in the simulation if they have been moved
        # TODO move to seperate function to check even if the environment is not reset
        if SIMGetObject(2)[0].value != 3700 or SIMGetObject(2)[1].value != 3036 or SIMGetObject(2)[2].value != 315: SIMSetObject(2, 3700, 3036, 10, 315+90) # Set the first object position
        if SIMGetObject(3)[0].value != 3680 or SIMGetObject(3)[1].value != 2019 or SIMGetObject(3)[2].value != 135: SIMSetObject(3, 3680, 2019, 10, 135+90) # Set the second object position        
        if SIMGetObject(4)[0].value != 3664 or SIMGetObject(4)[1].value != 4629 or SIMGetObject(4)[2].value != 0: SIMSetObject(4, 3664, 4629, 10, 0) # Set the third object position
        if SIMGetObject(5)[0].value != 3515 or SIMGetObject(5)[1].value != 1092 or SIMGetObject(5)[2].value != 0: SIMSetObject(5, 3515, 1092, 10, 0) # Set the fourth object position
        if SIMGetObject(6)[0].value != 3515 or SIMGetObject(6)[1].value != 360 or SIMGetObject(6)[2].value != 180: SIMSetObject(6, 3515, 360, 10, 180) # Set the fifth object position
        if SIMGetObject(7)[0].value != 3664 or SIMGetObject(7)[1].value != 3897 or SIMGetObject(7)[2].value != 180: SIMSetObject(7, 3664, 3897, 10, 180) # Set the sixth object position

        # TODO include resetting object positions
        self.update_polygon()

# INITIALIZE ----------------------------------------------------------------------------------------------------------

# Register the environment with gymnasium and create an instance of it
gym.register(
    id="gymnasium_env/EyeSimCaroloEnv",
    entry_point=EyeSimEnv,
)
env = gym.make("gymnasium_env/EyeSimCaroloEnv")

# Training parameters
learning_rate=0.0001
n_steps=2048

# TEST ----------------------------------------------------------------------------------------------------------------

# Function to test the environment and the robot's performance
def test(): 
    
    # Reset the environment and check if it is valid
    env.reset()

    # Initialize the environment and retrieve starting centroid
    action = [0,0]
    _, _, _, _, info= env.step(action)
    current_centroid = info["Current_Centroid"]

    while True:
        LCDMenu("STOP_POS", "RANDOM", "-", "EXIT")
        key = KEYRead()

        # Move robot to the centroid position 53, which is the position before the stop sign
        if key == KEY1:
            while current_centroid != stop_centroid-1:
                current_centroid+=1
                x, y, phi = centroids[current_centroid]
                SIMSetRobot(1,x,y,10,phi+180)
                current_centroid%=(len(centroids)-1)
                action = [0,0]
                _, reward, done, _, info= env.step(action)

        # Test the environment by taking random actions
        if key == KEY2:
            while True:
                action = env.action_space.sample()
                _, reward, done, _, info= env.step(action)
                print(f"Reward: {reward}, Action: {action}, Done: {done}, Info: {info}")
                current_centroid = info["Current_Centroid"]

                # Stop the random actions
                LCDMenu("-", "-", "-", "STOP")
                key = KEYRead()
                if key == KEY4:
                    VWSetSpeed(0,0)
                    break
        elif key == KEY4:
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
    model = PPO("CnnPolicy", env=env, verbose=1, tensorboard_log=logdir, learning_rate=learning_rate,n_steps=n_steps)

    # Train the model for 100,000 steps
    model.learn(total_timesteps=n_steps*50, progress_bar = True, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
    model.save(f"{models_dir}/model_0")

# LOAD ---------------------------------------------------------------------------------------------------------------- 

# Function to load a pre-trained model and test it
def load(): 
    # Find the most recent model
    result = find_latest_model()
    if result is None:
        return
    most_recent_model, _ = result

    print(f"Loading model: {most_recent_model}")

    # Load the pre-trained model
    trained_model = most_recent_model
    model_path = f"{models_dir}/{trained_model}"
    model = PPO.load(model_path,env=env)

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

        # Gather reward and speed information to average out
        if iteration < total_iterations:
            iteration += 1
            # Convert the speed to speed limit
            current_speed += action[1] * 100 
            current_reward += round(float(reward),2)

        else:
            # Clear the LCD area for speed display
            LCDArea(0,DESIRED_CAMHEIGHT, CAMWIDTH, DESIRED_CAMHEIGHT*2, BLACK,1) 
            
            # Check if the robot has reached a stop and if it has been stopped for more than 3 seconds
            if stop_reached and time.time() - stop_time >= 3.0: 
                stop_reached = False
                stop_time = 0.0
            
            # Speed limit logic
            if 0 < current_centroid < 29:
                speed_limit = 0.75
            elif current_centroid == stop_centroid:
                if not stop_reached:
                    speed_limit = 0.0
                    if action[1] == 0.0:
                        stop_reached = True
                        stop_time = time.time()
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
    model = PPO.load(model_path, env = env, learning_rate = learning_rate, n_steps = n_steps, tensorboard_log = logdir)
    
    # Continue training the model
    while True:
        LCDMenu("TRAIN", "-", "-", "STOP")
        key = KEYRead()
        if key == KEY1: # Train the model
            model.learn(total_timesteps = n_steps*25, progress_bar = True, reset_num_timesteps = False, tb_log_name = f"{algorithm}")
            new_model = f"model_{iteration}"
            model.save(f"{models_dir}/{new_model}")
            iteration += 1
        elif key == KEY4:
            break

        
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
        LCDMenu("TRAIN", "TEST", "LOAD", "STOP")

        key = KEYRead()
        if key == KEY1: # Train the model
            train()

        elif key == KEY2: # Load a pre-trained model and test it
            while True:
                LCDMenu("TEST", "OBJECT_POSITIONS", "RESET", "EXIT")
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
                LCDMenu("LOAD_TEST", "LOAD_TRAIN", "-", "EXIT")
                key = KEYRead()
                if key == KEY1: # Load a pre-trained model and test it
                    load()
                elif key == KEY2: # Load a pre-trained model and continue training it
                    load_train()
                elif key == KEY4: # Stop the program
                    break
            
        elif key == KEY4: # Stop the program
            break

main()

