import imageio
import os
import random
import numpy as np
import configparser
import pygame

# Importing GoalVsHoleEnv from the env/goal_vs_hole_env file
from env.goal_vs_hole_env import GoalVsHoleEnv

class QLearningAgent:
    def __init__(self, config, screen = None):
        self.q_table = np.zeros((config['grid_size'] * config['grid_size'], 4))
        self.alpha = config['alpha'] 
        self.gamma = config['gamma'] 
        self.epsilon = config['epsilon'] 
        self.epsilon_decay = config['epsilon_decay']
        self.screen = screen
        self.maxgifs = config['max_gifs']

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])  # Random action
        else:
            return np.argmax(self.q_table[state])  # Best action based on Q-table

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])  # Choose the best action for next state
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]  # Q-value update
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action]) 

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay  # Reduce exploration rate

    def print_q_table(self):
        print("Q-Table:")
        print(self.q_table)

def train_agent():
    config = configparser.ConfigParser()
    config.read("configs/env.ini")

    env_config = {
        "grid_size": int(config['ENV']['grid_size']),
        "cell_size": int(config['ENV']['cell_size']),
        "holes": list(map(int, config['ENV']['holes'].split(','))),
        "goals": list(map(int, config['ENV']['goals'].split(','))),
        "alpha": float(config['AGENT']['alpha']),
        "gamma": float(config['AGENT']['gamma']),
        "epsilon": float(config['AGENT']['epsilon']),
        "epsilon_decay": float(config['AGENT']['epsilon_decay']),
        "total_episode_count": int(config['AGENT']['total_episode_count']),
        "max_wins": int(config['AGENT']['max_wins']),
        "max_gifs": int(config['AGENT']['max_gifs'])
    }

    env = GoalVsHoleEnv(config_file="configs/env.ini")
    agent = QLearningAgent(env_config)

    GIF_FOLDER = "gifs"
    if not os.path.exists(GIF_FOLDER):
        os.makedirs(GIF_FOLDER)
    
    for file in os.listdir(GIF_FOLDER):
        os.remove(os.path.join(GIF_FOLDER, file))

    saved_gifs = []
    total_wins = 0

    for episode in range(env_config['total_episode_count']):
        state = env.reset()
        total_reward = 0
        done = False
        frames = []

        frame = env.render(state, episode + 1, total_reward, total_wins)  # Render the environment and capture the frame
        frames.append(frame)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return  # Exit if the window is closed

            action = agent.choose_action(state) 
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            agent.update(state, action, reward, next_state)

            frame = env.render(state, episode + 1, total_reward, total_wins)  # Render the environment and capture the frame
            frames.append(frame)
            
            state = next_state

        if done:
            if state in env.holes:
                reward = env.config['rewards']['hole']
            elif state in env.goals:
                reward = env.config['rewards']['goal']
                total_wins += 1  # Increment wins if the agent reaches a goal

            if episode >= env_config['total_episode_count'] - env_config['max_gifs']:
                gif_path = os.path.join(GIF_FOLDER, f"episode_{episode + 1}.gif")
                saved_gifs.append(gif_path)
                imageio.mimsave(gif_path, frames, fps=2)  # Saving the episode as a gif

        agent.decay_epsilon()  # Reducing exploration over time

        if total_wins >= env_config['max_wins']:
            print(f"Maximum wins reached: {total_wins}. Stopping training.")
            break

    agent.print_q_table()
    env.close()

if __name__ == "__main__":
    train_agent()
