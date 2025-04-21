#!/usr/bin/env python

import tensorflow as tf
import random

tf.disable_v2_behavior()

# Environment definition (CartPole-like, simplified)
class CartPoleEnv:
    def __init__(self):
        self.state = [0.0, 0.0, 0.0, 0.0]  # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        self.action_space = [0, 1]  # 0: push left, 1: push right
        self.state_size = 4
        self.action_size = 2
        self.max_steps = 200
        self.steps = 0

    def reset(self):
        self.state = [random.uniform(-0.05, 0.05) for _ in range(4)]
        self.steps = 0
        return self.state

    def step(self, action):
        cart_pos, cart_vel, pole_angle, pole_vel = self.state

        force = -10.0 if action == 0 else 10.0
        cart_acc = force  # Simplified physics
        pole_acc = 0.1 * cart_pos - 0.5 * pole_angle # simplified physics

        cart_vel += cart_acc * 0.02
        cart_pos += cart_vel * 0.02
        pole_vel += pole_acc * 0.02
        pole_angle += pole_vel * 0.02

        self.state = [cart_pos, cart_vel, pole_angle, pole_vel]
        self.steps += 1

        reward = 1.0
        done = (abs(pole_angle) > 0.2 or abs(cart_pos) > 2.4 or self.steps >= self.max_steps)

        return self.state, reward, done

# Hyperparameters
learning_rate = 0.001
gamma = 0.99

# Q-network
inputs = tf.placeholder(tf.float32, [None, 4])
W1 = tf.get_variable("W1", shape=[4, 16], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", shape=[16], initializer=tf.zeros_initializer())
hidden = tf.nn.relu(tf.matmul(inputs, W1) + b1)

W2 = tf.get_variable("W2", shape=[16, 2], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", shape=[2], initializer=tf.zeros_initializer())
Q_values = tf.matmul(hidden, W2) + b2

# Target Q-values and loss
target_Q = tf.placeholder(tf.float32, [None, 2])
loss = tf.reduce_mean(tf.square(target_Q - Q_values))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Training
episodes = 1000
env = CartPoleEnv()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(env.max_steps):
            if random.random() > 0.1:
                action = sess.run(tf.argmax(Q_values, 1), feed_dict={inputs: [state]})[0]
            else:
                action = random.choice(env.action_space)

            next_state, reward, done = env.step(action)
            total_reward += reward

            target_Q_val = sess.run(Q_values, feed_dict={inputs: [state]})
            if done:
                target_Q_val[0, action] = reward
            else:
                target_Q_val[0, action] = reward + gamma * sess.run(tf.reduce_max(Q_values, 1), feed_dict={inputs: [next_state]})

            sess.run(optimizer, feed_dict={inputs: [state], target_Q: target_Q_val})

            state = next_state

            if done:
                break

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

print("Training finished.")