# Vision‑Based Mobile Robot Control with PPO

**EyeSim Project** is an advanced simulation framework that combines the semi-realistic EyeBot simulator with modern reinforcement learning tools. It enables training of the Eyebot—using camera-only inputs.

---

## Background: EyeSim Simulator

EyeSim is a mobile robot simulator developed to mirror the behavior of EyeBot hardware, including differential steering, onboard synthetic vision, odometry, and more. It allows programs written for the EyeBot to be run in simulation, supporting rapid development and testing.

## Background: Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) is a state-of-the-art reinforcement learning algorithm widely used for training agents in complex environments. PPO belongs to the family of policy gradient methods, which directly optimize the agent's behavior policy.

Key features of PPO:
- **Stability and Efficiency**: PPO improves training stability by limiting how much the policy can change at each update step, preventing destructive updates.
- **Clipped Objective**: The main innovation in PPO is the use of a clipped surrogate objective, which ensures that policy updates do not deviate too much from the previous policy.
- **Continuous and Discrete Actions**: PPO can handle both continuous and discrete action spaces, making it versatile for various tasks.
- **Simplicity**: Despite its strong performance, PPO is relatively simple to implement and tune compared to other advanced algorithms.

In the context of EyeSim, PPO is used to train the Eyebot to navigate using vision alone, learning to interpret raw camera inputs and produce effective driving commands. Its robustness and generalization capabilities make it a preferred choice for vision-based robot control tasks.

---

## Core Objectives

- **Visual Navigation**: Train the Eyebot to drive along road-like track layouts and adhere to simulated traffic elements such as lane markings and signs, using only its camera feed.
- **Object Tracking**: Implement a training regime where the Eyebot learns to rotate toward a red-colored object placed in the environment—again, relying solely on vision.

---

## Technical Stack

- **Simulation Environment**: Gymnasium-based wrapper around the EyeSim engine, providing a standardized RL environment interface.
- **RL Algorithm**: Proximal Policy Optimization (PPO) implemented via **Stable‑Baselines3**.
- **Observation Space**: Raw camera images from the Eyebot—no additional sensor inputs.
- **Action Space**: Continuous driving commands (linear and angular velocity).

---

## Repository Structure

├── Map_points/                # text files with EyeSim coordinates for current tracks  
├── Red_Object_Tracking/       # PPO training scripts for red object tracking  
├── sims/                      # EyeSim world files and simulator setups
├── images-videos/             # Output media and visualisations  
├── logs/                      # TensorBoard logs and metrics  
├── models/                    # Saved trained models  
├── Angular_Control_PPO.py     # Angular controller script  
├── helper_functions.py        # Shared utility functions  
├── Image_Point_Processing.py  # Processes given points into EyeSim coordinates   
├── Linear_Control_PPO.py      # linear controller script  
├── eye.py                     # EyeSim commands script  
└── README.md                  # Project documentation  

---

## Installation

### Prerequisites

- Python 3.10+
- [EyeSim](https://roblab.org/eyesim/) 
- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- PyTorch, OpenCV, NumPy, Matplotlib

### Install Dependencies

```bash
git clone https://github.com/NoahWMueller/Eyesim.git
cd Eyesim
pip install gymnasium stable-baselines3 opencv-python numpy
```

---

## Training and Loading the models

### Main Programs

To train the robot for any of the 3 independent tasks you must launch EyeSim, load the correct *.sim file found in the sims folder and run respective python file.

```bash
python3 Angular_Control_PPO.py
python3 Linear_Control_PPO.py
python3 Red_Object_Tracking/red_object_finder_PPO.py
```

After loading the dependencies it will produce a popup display with 4 buttons located at the bottom, these provide various different commands and controls for training, loading, and testing the model and its behaviour.

### Additional Programs

There are 2 additional programs, **Image_Point_Processing.py** which is used to take image points and convert them to correctly scaled and oriented EyeSim world coordinates, **Helper_Functions.py** includes seperate functions that have been moved to help reduce file sizes and improve overall organisation.

```bash
python3 Image_Point_Processing.py
python3 Helper_Functions.py
```
