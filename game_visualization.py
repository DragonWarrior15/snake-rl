# script to visualize how agent plays a game
# useful to study different iterations

import numpy as np
from agent import QLearningAgent
from game_environment import Snake
from tqdm import tqdm
from utils import visualize_game
import keras.backend as K

# some global variables
board_size = 11
frames = 4

# setup the environment
env = Snake(board_size=board_size, frames=frames)
s = env.reset()

# setup the agent
K.clear_session()
agent = QLearningAgent(board_size=board_size, frames=frames, buffer_size=20000)

for iteration in [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]:
    agent.load_model('models_v2', iteration=iteration)
    visualize_game(env, agent, path='images/game_visual_v3_{:d}.png'.format(iteration), debug=True)
