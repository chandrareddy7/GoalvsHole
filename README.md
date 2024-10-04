# Goal vs Hole: Conversion of Non-Gym Game to Gymnasium Environment

## Assignment 1
**Course:** AI w/ Reinforcement Learning  
**Instructor:** Ashis Kumer Biswas, PhD  
**Fall 2024**  
**Title:** Conversion of the Non-Gym “Goal-vs-Hole-v1” Game to Gymnasium

### Overview
This project involves converting the 2D non-gym game "Goal vs Hole" into a Gymnasium environment. In this game, an agent starts from the top-left cell of a 4x4 grid and can choose one of four possible actions: moving **up**, **down**, **left**, or **right**. The objective is to find the optimal path to maximize rewards.


#### Specifications:
- The environment is a **4x4 grid**.
- It has **terminal states**: Goal and Holes; moving into any terminal state ends the game.
- There are **non-terminal states**.
- The agent starts at the **"Start"** state located at the top-left grid cell.
- Rewards:
  - **0** for non-terminal states.
  - **-100** for Hole states.
  - **+100** for the Goal state.

### Tasks Completed
1. Created a custom Gymnasium environment for the "Goal vs Hole" game.
2. Implemented the agent using Q-learning with a Q-table to optimize the Bellman’s equation.
3. Adjusted exploration-exploitation rates with a discount factor of **γ=0.9**.
4. Rendered multiple episodes of the intelligent agent as GIF animations.
5. Configured the environment settings via a configuration file (`env.ini`).

### Installation

1. Unzip the project file
2. Install all the required libraries using `pip install -r requirements.txt`
3. Run `python train_agent.py`. This will train the agent and save the GIFs in gifs folder
4. Ensure that Python and PIP are installed on your system.
5. Modify the configuration file (env.ini) to adjust parameters as needed.