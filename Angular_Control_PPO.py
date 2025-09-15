#!/usr/bin/env python

# TO-DO --------------------------------------------------------------------------------------------------------------
"""

"""
# IMPORTS ------------------------------------------------------------------------------------------------------------

import time
from eye import *
import gymnasium as gym
from random import randint
from Helper_Functions import *
from stable_baselines3 import PPO

# GLOBAL VARIABLES ---------------------------------------------------------------------------------------------------

# Constants for camera settings
CAM_SETTING = QVGA
CAMWIDTH = QVGA_X
CAMHEIGHT = QVGA_Y

LCD_Right_Print = 52

# Directory paths for saving models and logs
models_dir = f"models/Angular"
logdir = f"logs/Angular"

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
batch_size=128
ent_coef=0.005
clip_range=0.15
max_grad_norm=0.25
gamma = 0.99

# INITIALISING TRACK ---------------------------------------------------------------------------------------------------

# Load the lane coordinates from files
left_lane = []
left_centroids = []
right_lane = []
right_centroids = []

total_tracks = 2

def set_track(track=1):
    global left_lane
    global left_centroids
    global right_lane
    global right_centroids
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

        # Moving to accomdate track position
        for name in ["left_lane", "right_lane"]:
            globals()[name] = [(x + 4875, y) for (x, y) in globals()[name]]
        for name in ["left_centroids", "right_centroids"]:
            globals()[name] = [(x + 4875, y, phi) for (x, y, phi) in globals()[name]]

set_track()


# GYMNASIUM ENVIRONMENT --------------------------------------------------------------------------------------------------------

# Custom environment for the robot simulation using OpenAI Gymnasium
class EyeSimEnv(gym.Env):
    
    def __init__(self):
        super(EyeSimEnv, self).__init__()
        # Define the lower and upper bounds
        low = np.array([-1.0], dtype=np.float32)
        high = np.array([1.0], dtype=np.float32)

        # Float action space for robot angular speed
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32) 

        # Image observation space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(CAMHEIGHT,CAMWIDTH,3), dtype=np.uint8) 
        
        # Initialize variables
        self.current_centroid = 0
        self.next_centroid = 0
        self.current_polygon = np.array([])
        self.next_polygon = np.array([])
        self.current_lane = left_lane
        self.current_centroids = left_centroids
        self.update_polygon()
        self.finish_centroid = len(self.current_centroids) - 1 # Finish centroid is the one before the current centroid
        self.current_track = 1
        self.lap_count = 0

        self.new_reset_point = (self.current_centroids[self.current_centroid][0],
                            self.current_centroids[self.current_centroid][1],
                            self.current_centroids[self.current_centroid][2],
                            self.current_centroid) 
        self.reset_point = (self.current_centroids[self.current_centroid][0],
                            self.current_centroids[self.current_centroid][1],
                            self.current_centroids[self.current_centroid][2],
                            self.current_centroid) 
        self.timesteps = 0
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.eyesim_reset()
        observation = self.eyesim_get_observation()
        info = {}
        return observation, info

    def step(self, action):
        self.timesteps += 1
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
                "Current_Lane": "left_lane" if self.current_lane == left_lane else "right_lane",
                "reset_point": self.reset_point,
                "new_reset_point": self.new_reset_point}

        if self.timesteps >= n_steps:
            self.timesteps = 0
            VWSetSpeed(0,0)
        
        return observation, reward, done, truncated, info

    def calculate_drive_reward(self, result1, result2):
        # If the robot is inside the next polygon, return a positive reward
        if result2 > 0:
            reward = 5.0
            # Update reset position
            if self.new_reset_point[3] < self.current_centroid: 
                self.reset_point = self.new_reset_point
                self.new_reset_point = (SIMGetRobot(1)[0].value, SIMGetRobot(1)[1].value, abs(360 - SIMGetRobot(1)[3].value)-180, self.current_centroid)

            elif self.new_reset_point[3] == self.current_centroid: 
                self.new_reset_point = (SIMGetRobot(1)[0].value, SIMGetRobot(1)[1].value, abs(360 - SIMGetRobot(1)[3].value)-180, self.current_centroid)

            # update centroid and previous polygon to current polygon
            self.current_centroid = self.current_centroid + 1

            # Additional reward for making it to the end of the track
            if self.current_centroid == len(self.current_centroids) - 1: reward += 15.0
            self.update_polygon()

            return reward
        
        # If the robot is inside neither  polygon, return a big negative reward
        elif result1 < 0 and result2 < 0: 
            return -10.0
        # If the robot is inside the current polygon, return no reward
        else:
            return 0.0

    def is_done(self, result1, result2):
        # Determine if the robot left all allowable polygons
        if (result1 == -1 and result2 == -1) or self.lap_check(): 
            return True
        else: 
            return False

    # INCLUDED EYESIM HELPER FUNCTIONS --------------------------------------------------------------------------------------------------

    def lap_check(self):
        # If the robot has completed a lap, switch sides and reset the current centroid
        if self.current_centroid == self.finish_centroid:
            
            # After two successful laps switch course
            self.lap_count += 1
            if self.lap_count == 2:
                if self.current_track == 1:
                    LCDSetPrintf(2,LCD_Right_Print,"setting track 1")
                    set_track(2)
                    self.current_track = 2
                else:
                    LCDSetPrintf(2,LCD_Right_Print,"setting track 1")
                    set_track(1)
                    self.current_track = 1
                self.lap_count = 0

            return True
        else:
            return False

    # Function to set the speed of the robot based on the action taken
    def eyesim_set_robot_speed(self, angular): 
        # Set the speed of the robot based on the action taken
        angular_speed = 200
        VWSetSpeed(200,round(angular_speed*angular)) # Set the speed of the robot

    # Function to get the image from the camera and process it
    def eyesim_get_observation(self): 
        # Get image from camera
        image = CAMGet() 
        LCDImage(image)

        # Convert the image to a numpy array and reshape to observation space
        processed_image = np.asarray(image, dtype=np.uint8).reshape((CAMHEIGHT, CAMWIDTH, 3))

        return processed_image

    def update_polygon(self):
        # Update the current and next polygon based on the current centroid
        self.current_polygon = np.array([
            self.current_lane[self.current_centroid*2],
            self.current_lane[self.current_centroid*2+1],
            self.current_lane[(self.current_centroid*2+3)],
            self.current_lane[(self.current_centroid*2+2)]
        ], np.int32)

        self.next_centroid = (self.current_centroid + 1) % len(self.current_centroids)

        self.next_polygon = np.array([
            self.current_lane[self.next_centroid*2],
            self.current_lane[self.next_centroid*2+1],
            self.current_lane[(self.next_centroid*2+3)],
            self.current_lane[(self.next_centroid*2+2)]
        ], np.int32)

    # Function to get the current position of the robot in the simulation
    def eyesim_get_position(self): 
        # Get the current position of the robot in the simulation
        [x,y,_,_] = SIMGetRobot(1)
        point = (x.value, y.value)

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
        
        if self.current_centroid == self.finish_centroid: # If robot is at the first centroid, randomly select a new starting centroid
            if self.current_lane == left_lane:
                self.current_lane = right_lane
                self.current_centroids = right_centroids
            else:
                self.current_lane = left_lane
                self.current_centroids = left_centroids

            self.current_centroid = 0 # Randomly select a starting centroid
            self.finish_centroid = len(self.current_centroids) - 1 # Finish centroid is the one before the current centroid

            self.reset_point = (self.current_centroids[self.current_centroid][0],
                                self.current_centroids[self.current_centroid][1],
                                self.current_centroids[self.current_centroid][2],
                                self.current_centroid)
            
            self.new_reset_point = (self.current_centroids[self.current_centroid][0],
                                self.current_centroids[self.current_centroid][1],
                                self.current_centroids[self.current_centroid][2],
                                self.current_centroid)
        
        # Position the robot in the correct position based on the current centroid
        x, y, phi, reset_centroid = self.reset_point
        self.current_centroid = reset_centroid
        SIMSetRobot(1, x, y, 10, phi + 180) # Add 180 degrees to the angle to flip robot into correct direction

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
        LCDMenu("-", "-", "-", "Stop")
        key = KEYRead()
        
        # Take random actions in the environment
        action = env.action_space.sample()
        _, reward, done, _, info= env.step(action)
        print(f"Reward: {reward}, Action: {action}, Done: {done}, Current_Centroid: {info['Current_Centroid']}, Current_Lane: {info['Current_Lane']}, Reset_point: {info['reset_point']}, New_Reset_Point: {info['new_reset_point']}")

        # If the episode is done, reset the environment
        if done: env.reset()

        # Stop the random actions
        if key == KEY4:
            VWSetSpeed(0,0)
            LCDClear()
            break

# TRAIN ---------------------------------------------------------------------------------------------------------------

# Function to train the robot behaviour using an reinforcement learning algorithm
def train(): 

    # Define the PPO model with the specified parameters
    model = PPO(policy_network, env=env, verbose=1, tensorboard_log=logdir, n_steps=n_steps,
                learning_rate=learning_rate, batch_size=batch_size, ent_coef=ent_coef, 
                clip_range=clip_range, max_grad_norm=max_grad_norm, gamma=gamma)

    # Train the model
    for i in range(4): # Train the model
        model.learn(total_timesteps=100*n_steps, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
        model.save(f"{models_dir}/angular_model_{i}")

# LOAD ---------------------------------------------------------------------------------------------------------------- 

# Function to load a pre-trained model and test it
def load_test(model): 

    LCDSetPrintf(3,LCD_Right_Print,f"Loading model:     ")
    LCDSetPrintf(4,LCD_Right_Print,f"{model[:-4]}     ")

    # Load the pre-trained model
    model_path = f"{models_dir}/{model}"
    loaded_model = PPO.load(model_path)

    LCDSetPrintf(3,LCD_Right_Print,f"Loaded model:      ")
    LCDSetPrintf(4,LCD_Right_Print,f"{model[:-4]}     ")

    # Continue testing the loaded model until the user decides to stop
    while True:
        LCDMenu("-", "-", "-", "Stop")
        key = KEYRead()

        # Get image from camera and display it on LCD
        image = CAMGet() 
        LCDImage(image)
    
        # Convert the image to a numpy array
        processed_image = np.asarray(image, dtype=np.uint8).reshape((CAMHEIGHT, CAMWIDTH, 3))

        # Predict the action using the loaded model and given observation
        action, _ = loaded_model.predict(processed_image, deterministic=False)
        print(action)
        LCDSetPrintf(5,LCD_Right_Print,f"Prediction: {round(float(action),2)}       ")
        angular_speed = 200
        VWSetSpeed(200,round(angular_speed*float(action))) # Set the speed of the robot
        
        # End testing if user presses the stop key
        if key == KEY4: 
            LCDClear()
            VWSetSpeed(0,0)
            break
    
# LOAD AND TRAIN --------------------------------------------------------------------------------------------------------
    
# Function to load a pre-trained model and continue training it
def load_train(model, iteration): 

    print(f"Loading model: {model}")

    # Load the pre-trained model
    model_path = f"{models_dir}/{model}"
    model = PPO.load(model_path, env=env, tensorboard_log=logdir)
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
                new_model = f"angular_model_{new_iteration}"
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
    # Initialize the camera
    CAMInit(CAM_SETTING) 
    LCDImageStart(0,0,CAMWIDTH,CAMHEIGHT)
    LCDSetPrintf(0,LCD_Right_Print,"Angular Control")

    while True:
        LCDMenu("Train", "Test", "Load", "Quit")
        key = KEYRead()

        # Train the model
        if key == KEY1: 
            train()

        # Testing Menu
        elif key == KEY2: 
            while True:
                LCDMenu("Env", "Reset", "Track", "Back")
                key = KEYRead()

                if key == KEY1: # Test the environment with random actions
                    test()
                elif key == KEY2: # Reset the environment
                    env.reset()
                
                elif key == KEY3:
                    track = 1
                    while(True):
                        LCDMenu("Up", "Down", "Test", "Back")
                        key = KEYRead()
                        if key == KEY1:
                            if track < total_tracks: track +=1
                            LCDSetPrintf(2,LCD_Right_Print,f"Selecting Track {track}")
                        elif key == KEY2:
                            if track > 1: track -=1
                            LCDSetPrintf(2,LCD_Right_Print,f"Selecting Track {track}")
                        elif key == KEY3:
                            set_track(track)
                            for (x,y,phi) in left_centroids:
                                SIMSetRobot(1,x,y,10,phi+180)
                                time.sleep(0.1)
                        elif key == KEY4:
                            LCDClear()
                            break

                elif key == KEY4: # Back to the main menu
                    break 
        
        # Load Menu
        elif key == KEY3: 
            model = "None"
            model_number = 0
            while True:
                LCDMenu("Test", "Train", "Select", "Back")
                key = KEYRead()
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
                            LCDSetPrintf(20,LCD_Right_Print,f"Selected Model = {model_number}")
                            if key == KEY1:
                                if model_number < len(model_list): 
                                    model_number +=1
                                    model = f"angular_model_{model_number}.zip"
                                LCDSetPrintf(20,LCD_Right_Print,f"Selected Model = {model_number}")
                            elif key == KEY2:
                                if model_number > 1: 
                                    model_number -= 1
                                    model = f"angular_model_{model_number}.zip"
                                LCDSetPrintf(20,LCD_Right_Print,f"Selected Model = {model_number}")
                            elif key == KEY3:
                                model_number = most_recent_model
                                model = f"angular_model_{model_number}.zip"
                                LCDSetPrintf(20,LCD_Right_Print,f"Selected Model = {model_number}")
                            elif key == KEY4:
                                break
                    else:
                        print("No models available, please train one to select.")

                elif key == KEY4: # Back to the main menu
                    VWSetSpeed(0,0)
                    LCDClear()
                    break
        
        # Stop the program
        elif key == KEY4:
            break
            

main()

