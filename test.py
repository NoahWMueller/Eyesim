import gymnasium as gym
import numpy as np

# Define the lower and upper bounds
low = np.array([-1.0, 0.0], dtype=np.float32)
high = np.array([1.0, 1.0], dtype=np.float32)

# Create the action space
action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

# Example: sample a random action
action = action_space.sample()
print(action)