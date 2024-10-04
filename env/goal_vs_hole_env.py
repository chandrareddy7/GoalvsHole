import gymnasium as gym
import numpy as np
import configparser
import pygame
from gymnasium import spaces
from gymnasium.envs.registration import register

# Register the custom environment
register(
    id='GoalVsHole-v0',
    entry_point='env.goal_vs_hole_env:GoalVsHoleEnv',
    max_episode_steps=100,
)

class GoalVsHoleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config_file="configs/env.ini"):
        self.config = self.load_config(config_file)
        self.grid_size = self.config['grid_size']
        self.cell_size = self.config['cell_size']
        
        self.start = 0
        self.holes = self.config['holes']
        self.goals = self.config['goals']
        self.state = self.start

        # action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.grid_size ** 2)

        # Render-related attributes
        self.screen = None
        self.clock = None
        self.render_mode = self.config['render_mode']
        
        self.reset()

    def load_config(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        return {
            "holes": list(map(int, config['ENV']['holes'].split(','))),
            "goals": list(map(int, config['ENV']['goals'].split(','))),
            "grid_size": int(config['ENV']['grid_size']),
            "cell_size": int(config['ENV']['cell_size']),
            "rewards": {
                "hole": int(config['ENV']['reward_hole']),
                "goal": int(config['ENV']['reward_goal']),
                "step": int(config['ENV']['reward_non_terminal']),
            },
            "render_mode": config['ENV']['render_mode']
        }

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        grid_size = self.config['grid_size']
        if action == 0:  # Up
            new_state = self.state - grid_size if self.state >= grid_size else self.state
        elif action == 1:  # Down
            new_state = self.state + grid_size if self.state < grid_size * (grid_size - 1) else self.state
        elif action == 2:  # Left
            new_state = self.state - 1 if self.state % grid_size > 0 else self.state
        elif action == 3:  # Right
            new_state = self.state + 1 if self.state % grid_size < grid_size - 1 else self.state

        reward = self.config['rewards']['step']
        done = False

        if new_state in self.holes:
            reward = self.config['rewards']['hole']
            done = True
        elif new_state in self.goals:
            reward = self.config['rewards']['goal']
            done = True

        self.state = new_state
        return self.state, reward, done, {}

    # Render method to display the environment
    def render(self, state, episode, total_reward, wins):
        if self.render_mode is None:
            return
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            )

        self.screen.fill((255, 255, 255))  # Clearing screen
        CELL_SIZE = self.cell_size
        GRID_SIZE = self.grid_size

        # Drawing grid lines
        for x in range(0, GRID_SIZE * CELL_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, GRID_SIZE * CELL_SIZE))
        for y in range(0, GRID_SIZE * CELL_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, (0, 0, 0), (0, y), (GRID_SIZE * CELL_SIZE, y))

        font = pygame.font.Font(None, 36)

        # Drawing the agent
        agent_row, agent_col = divmod(self.state, GRID_SIZE)
        agent_text = font.render('A', True, (0, 255, 0))
        self.screen.blit(agent_text, (agent_col * CELL_SIZE + (CELL_SIZE // 4), agent_row * CELL_SIZE))

        # Drawing the holes
        for hole in self.holes:
            hole_row, hole_col = divmod(hole, GRID_SIZE)
            pygame.draw.rect(self.screen, (255, 0, 0), (hole_col * CELL_SIZE, hole_row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            hole_text = font.render('H', True, (255, 255, 255))
            hole_text_rect = hole_text.get_rect(center=(hole_col * CELL_SIZE + CELL_SIZE // 2, hole_row * CELL_SIZE + CELL_SIZE // 2))
            self.screen.blit(hole_text, hole_text_rect)

        # Drawing the goals
        for goal in self.goals:
            goal_row, goal_col = divmod(goal, GRID_SIZE)
            pygame.draw.rect(self.screen, (0, 255, 0), (goal_col * CELL_SIZE, goal_row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            goal_text = font.render('G', True, (255, 255, 255)) 
            goal_text_rect = goal_text.get_rect(center=(goal_col * CELL_SIZE + CELL_SIZE // 2, goal_row * CELL_SIZE + CELL_SIZE // 2))
            self.screen.blit(goal_text, goal_text_rect)

        agent_row, agent_col = divmod(self.state, GRID_SIZE)
        pygame.draw.rect(self.screen, (0, 0, 255), (agent_col * CELL_SIZE, agent_row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        agent_text = font.render('A', True, (255, 255, 255))
        agent_text_rect = agent_text.get_rect(center=(agent_col * CELL_SIZE + CELL_SIZE // 2, agent_row * CELL_SIZE + CELL_SIZE // 2))
        self.screen.blit(agent_text, agent_text_rect)
        
        small_font = pygame.font.Font(None, 24)

        info_text = f'Episode: {episode} | Total Reward: {total_reward} | Current Wins: {wins}'
        info_surface = small_font.render(info_text, True, (0, 0, 0))
        self.screen.blit(info_surface, (10, 10))  

        pygame.display.flip()
        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))

    # Close method for cleanup
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
