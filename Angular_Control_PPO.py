#!/usr/bin/env python

# TO-DO --------------------------------------------------------------------------------------------------------------

# add potential for carolo_2 map
# fix the path placement to random position along track

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
models_dir = f"models/Carolo/{version}/Angular"
logdir = f"logs/Carolo/{version}/Angular"


# Check if the models directory exists, if not create it
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Check if the logs directory exists, if not create it
if not os.path.exists(logdir):
    os.makedirs(logdir)


# Algorithm used for training
algorithm = "PPO" 
policy_network = "CnnPolicy" # Policy network used for training

# Training parameters
learning_rate=0.0001
n_steps=2048

# Load the lane coordinates from files
track = 2

if track == 1:
    left_lane = load_map_points("Map_points/Track_1/left_lane.txt")
    left_centroids = load_map_points("Map_points/Track_1/left_centroids.txt")
    right_lane = load_map_points("Map_points/Track_1/right_lane.txt")
    right_centroids = load_map_points("Map_points/Track_1/right_centroids.txt")

elif track == 2:
    left_lane = load_map_points("Map_points/Track_2/left_lane.txt")
    left_centroids = load_map_points("Map_points/Track_2/left_centroids.txt")
    right_lane = load_map_points("Map_points/Track_2/right_lane.txt")
    right_centroids = load_map_points("Map_points/Track_2/right_centroids.txt")

# GYMNASIUM ENVIRONMENT --------------------------------------------------------------------------------------------------------

# Custom environment for the robot simulation using OpenAI Gymnasium
class EyeSimEnv(gym.Env):
    
    def __init__(self):
        super(EyeSimEnv, self).__init__()
        # Define the lower and upper bounds
        low = np.array([-1.0], dtype=np.float32)
        high = np.array([1.0], dtype=np.float32)

        # Float action space for robot angular speed, range from -1 to 1
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32) 
        # Image observation space, 3 channels (RGB), 120x160 pixels
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(CAMHEIGHT,CAMWIDTH,3), dtype=np.uint8) 
        
        # Initialize variables
        self.current_polygon = np.array([])
        self.current_centroid = randint(0, len(left_centroids) - 1) # Randomly select a starting centroid
        self.next_polygon = np.array([])
        self.next_centroid = self.current_centroid + 1
        self.current_lane = left_lane
        self.current_centroids = left_centroids

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.eyesim_reset()
        observation = self.eyesim_get_observation()
        info = {}
        return observation, info

    def step(self, action):
        angular = action[0] # linear and angular action

        # Determines if robot is inside left lane or has gotten lost
        result1, result2 = self.eyesim_get_position() 
        
        # Set robot linear and angular speed based on action
        self.eyesim_set_robot_speed(angular) 

        # Read image from camera
        observation = self.eyesim_get_observation()

        # Calculate reward based on position
        reward = self.calculate_drive_reward(result1, result2) # Calculate the drive reward based on the position

        # Determine if the episode is done
        done = self.is_done(result1,result2)

        # Truncated is not used in this case, but included for compatibility with gym API
        truncated = False

        # Create info dictionary to store additional information
        info = {"Reward": reward,
                "Current_Centroid": self.current_centroid,
                "Current_Lane": "left_lane" if self.current_lane == left_lane else "right_lane"}

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
                self.current_centroids = right_centroids
                self.current_centroid = 0
                return True
            else:
                self.current_lane = left_lane
                self.current_centroids = left_centroids
                self.current_centroid = 0
                return True
        return False

    # Function to set the speed of the robot based on the action taken
    def eyesim_set_robot_speed(self, angular): 
        # Set the speed of the robot based on the action taken
        angular_speed = 100
        VWSetSpeed(200,round(angular_speed*angular)) # Set the speed of the robot

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
        # Get the current position of the robot in the simulation
        [x,y,_,_] = SIMGetRobot(1)
        point = (x.value, y.value)

        # Update the current and next polygons based on the current centroid
        self.update_polygon()

        # Reshape the polygon points
        self.current_polygon = self.current_polygon.reshape((-1, 1, 2))
        self.next_polygon = self.next_polygon.reshape((-1, 1, 2))

        # Check if either of the points are inside the polygon
        current_result = cv2.pointPolygonTest(self.current_polygon, point, False)
        next_result = cv2.pointPolygonTest(self.next_polygon, point, False)

        return current_result, next_result

    # Function to reset the robot and can positions in the simulation
    def eyesim_reset(self): 
        # Stop robot movement
        VWSetSpeed(0,0)

        # If robot is not at the first centroid, move it back one centroid
        self.current_centroid = (self.current_centroid - 2) % len(self.current_centroids)
        
        # Position the robot in the correct position based on the current centroid
        x,y,phi = self.current_centroids[self.current_centroid]
        SIMSetRobot(1,x,y,10,phi+180) # Add 180 degrees to the angle to flip robot into correct direction

        # Update the current and next polygons based on the current centroid
        self.update_polygon()

# INITIALIZE ----------------------------------------------------------------------------------------------------------------

# Register the environment with gymnasium and create an instance of it
env_id = "gymnasium_env/AngularEnv"

# Check if the environment is already registered, if not register it
if env_id not in gym.registry:
    gym.register(
        id=env_id,
        entry_point="Angular_Control_PPO:EyeSimEnv",
    )

# Create an instance of the environment
env = gym.make(env_id)

# TEST ----------------------------------------------------------------------------------------------------------------

# Function to test the environment and the robot's performance
def test(): 
    env.reset()
    while True:
        LCDMenu("-", "-", "-", "STOP")
        key = KEYRead()
        
        # Take random actions in the environment
        action = env.action_space.sample()
        _, reward, done, _, info= env.step(action)
        print(f"Reward: {reward}, Action: {action}, Done: {done}, Current_Centroid: {info['Current_Centroid']}, Current_Lane: {info['Current_Lane']}")

        # If the episode is done, reset the environment
        if done: env.reset()

        # Stop the random actions
        if key == KEY4:
            VWSetSpeed(0,0)
            break

# TRAIN ---------------------------------------------------------------------------------------------------------------

# Function to train the robot behaviour using an reinforcement learning algorithm
def train(): 

    # Define the PPO model with the specified parameters
    model = PPO(policy_network, env=env, verbose=1, tensorboard_log=logdir, learning_rate=learning_rate, n_steps=n_steps)

    # Train the model for 100,000 steps
    model.learn(total_timesteps=150*n_steps, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
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
        obs, _, done, _, _= env.step(action)
        
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
                LCDMenu("TEST_ENV", "TEST_RESET", "-", "BACK")
                key = KEYRead()

                if key == KEY1: # Test the environment with random actions
                    test()
                elif key == KEY2: # Reset the environment
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

