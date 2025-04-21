#!/usr/bin/env python

import random
from random import randint
from eye import *
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

CAMWIDTH = 160
CAMHEIGHT = 120

# Environment -------------------------------------------------------------------------------------------------------

class EyeSimEnv(gym.Env):
    
    def __init__(self):
        super(EyeSimEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,)) # turn left or right
        self.observation_space = spaces.Box(low=0, high=80, shape=(1,)) # change to cam image

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # eyesim_reset()
        observation = eyesim_get_observation()
        info = {}
        return observation, info

    def step(self, action):
        
        eyesim_set_robot_speed(action[0]) # action[0] is angular motor speeds.

        observation = eyesim_get_observation() # position of red peak from 0 top 160

        position = eyesim_get_position() # distance to red peak

        # Calculate reward based on position or sensor readings
        reward = self._calculate_reward(position)
        # Determine if the episode is done
        done = self._is_done(position)
        info = {}
        return observation, reward, done, False, info

    def _calculate_reward(self, position):
        return 80 - position

    def _is_done(self, position):
        if position == 0:
            return True
        else:
            return False

def eyesim_set_robot_speed(speed):
    VWSetSpeed(0,10*speed) # 10 or -10 angular velocity
    OSWait(5) # 5 ms
    VWSetSpeed(0,0) # stop from moving

def eyesim_get_observation():
    # returns camera image
    # reduced camera size
    return 

def eyesim_get_position():
    return find_center() - 80

def eyesim_reset():
    # stop robot movement
    # set robot and can location randomly and make sure they arent in same spot
    VWSetSpeed(0,0)
    S4_pos_x = randint(200, 1800)
    S4_pos_y = randint(200, 1800)
    S4_phi = randint(0,360)

    CAN_pos_x = randint(200, 1800)
    CAN_pos_y = randint(200, 1800)

    while (CAN_pos_x == S4_pos_x):
        CAN_pos_x = randint(200, 1800)
    while (CAN_pos_y == S4_pos_y):
        CAN_pos_y = randint(200, 1800)

    SIMSetRobot(0,S4_pos_x,S4_pos_y,0,S4_phi)
    SIMSetObject(0,CAN_pos_x,CAN_pos_y,0,0)
    return

gym.register(
    id="gymnasium_env/EyeSimEnv-v0",
    entry_point=EyeSimEnv,
)

# TRAINING --------------------------------------------------------------------------------------------------

# Create the Lunar Lander environment
env = gym.make("gymnasium_env/EyeSimEnv-v0", render_mode="human")

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
gamma = 0.99  # discount factor
epsilon = 1.0  # exploration rate
epsilon_min = 0.01  # minimum exploration rate
epsilon_decay = 0.995  # decay rate of exploration probability
learning_rate = 0.001
batch_size  =  64
max_steps   = 500
memory_size = 1000000
target_update = 10  # how frequently to update the target network
num_episodes = 1000  # number of training episodes

# Neural Network for the DQN agent
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(memory_size)
        self.epsilon = epsilon

        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Initialize target model with the same weights as the original model
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))  # Explore
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.cpu().data.numpy())  # Exploit

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def train(self, batch_size):
        if self.memory.size() < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Q-values for current states
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Q-values for next states using the target network
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (gamma * next_q_values * (1 - dones))

        # Loss: Mean Squared Error between target and current Q-values
        loss = nn.MSELoss()(q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Train the DQN agent
def train_dqn_agent():
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    scores = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        score = 0
        step  = 0
        done = False

        while (not done) and (step<max_steps):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            score += reward

            agent.train(batch_size)

            step = step+1
            # stop at some stage
            if step >= max_steps : 
            	score = -50
            	done = True

        scores.append(score)
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

        if episode % target_update == 0:
            agent.update_target_model()

        print(f"Episode {episode + 1}/{num_episodes}, Score: {score:.2f}, Epsilon: {agent.epsilon:.2f}, Steps: {step}")

    return scores


# COLOUR DETECTION -------------------------------------------------------------------------------------------------------

def colour_search(h, s, i, CAMWIDTH, CAMHEIGHT):
    histogram = [0] * CAMWIDTH  # Initialize a histogram array for each column (0 to 159)

    # Loop over each column of the image
    index, max = 0, 0
    for x in range(CAMWIDTH):
        count = 0  # Reset the count of red pixels for each column

        # Loop over each row of the column
        for y in range(CAMHEIGHT):
            pos = y * CAMWIDTH + x  # Calculate the position in the 1D array
            
            # Check if the pixel matches the criteria to be considered red
            if (0 <= h[pos] < 45 or 359 > h[pos] > 345) and (i[pos] > 60 or i[pos] < -100)  and (s[pos] > 50 or s[pos] < -100):
                count += 1
                
        # Store the count of red pixels in the histogram array for this column
        histogram[x] = count
        
        if count > max:
            max = count
            index = x

        # Draw a bar or pixel to visualize the count for each column
        # Count represented as a vertical line on the LCD, with the height proportional to the count
        # LCDLine(x, 2*CAMHEIGHT, x, 2*CAMHEIGHT - count, RED)
    return index


# Main program loop
def find_center():
    index = 0
    CAMInit(QQVGA)  # Initialize camera
    LCDImageStart(0, 0, 160, 120)  # Set image start position for LCD display
    
    while True:
        img = CAMGet()  # Get image data from the camera
        LCDImage(img)  # Display the image on the LCD
        # draw centered line
        LCDLine(int(0.5*CAMWIDTH), 0, int(0.5*CAMWIDTH), CAMHEIGHT-1, BLUE)

        [h, s, i] = IPCol2HSI(img)  # Convert the image to HSI format

        index = colour_search(h, s, i, CAMWIDTH, CAMHEIGHT)

        # draw line where red is maximum
        LCDLine(index, 0, index, CAMHEIGHT-1, GREEN)

        # LCDClear()
        
# MAIN -------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    LCDMenu("START", "-", "-", "STOP")

    while (1):
        key = KEYRead()
        if (key == KEY1):
            pass

        if (key == KEY2):
            pass

        if (key == KEY3):
            pass

        if (key == KEY4):
            break