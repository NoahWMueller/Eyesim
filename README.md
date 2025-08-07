# EyeSim: Vision‑Based Mobile Robot Control with PPO

**EyeSim** is an advanced simulation framework that combines the semi-realistic EyeBot simulator with modern reinforcement learning tools. It enables training of the Eyebot—using camera-only inputs.

---

## Background: EyeSim Simulator

EyeSim is a mobile robot simulator developed to mirror the behavior of EyeBot hardware, including differential steering, onboard synthetic vision, odometry, and more. It allows programs written for the EyeBot to be run in simulation, supporting rapid development and testing.

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
