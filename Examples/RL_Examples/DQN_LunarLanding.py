import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Create the Lunar Lander environment
env = gym.make("LunarLander-v3", render_mode="human")

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

if __name__ == "__main__":
    train_dqn_agent()