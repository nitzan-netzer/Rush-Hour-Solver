# Rotem's algorithm for POC

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import os
import rush_hour_env


# Define the neural network
def build_model(input_dim, output_dim):
    model = keras.Sequential([
        layers.Dense(64, input_shape=(input_dim,), activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


# DQN Agent
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = build_model(self.input_dim, self.output_dim)
        self.target_model = build_model(self.input_dim, self.output_dim)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)

        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        updated_q_values = current_q_values.copy()

        for i in range(self.batch_size):
            if dones[i]:
                updated_q_values[i, actions[i]] = rewards[i]
            else:
                updated_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        self.model.fit(states, updated_q_values, epochs=1, verbose=0, batch_size=self.batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn():
    env = gym.make("RushHourEnv-v0")
    agent = DQNAgent(env)
    episodes = 500
    target_update_freq = 10

    for episode in range(episodes):
        state, _ = env.reset()  # Unpacking for Gym v26+
        state = np.array(state, dtype=np.float32)
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)  # Gym v26+ returns (s, r, d, t, i)
            next_state = np.array(next_state, dtype=np.float32)
            agent.remember(state, action, reward, next_state, done or truncated)
            agent.replay()
            state = next_state
            total_reward += reward

        if episode % target_update_freq == 0:
            agent.update_target_model()

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

    env.close()


if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train_dqn()
