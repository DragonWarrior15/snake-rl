# script to visualize how agent plays a game
# useful to study different iterations

import numpy as np
from agent import QLearningAgent
from game_environment import Snake
from tqdm import tqdm
from utils import visualize_game
import keras.backend as K

# some global variables
board_size = 5
frames = 2

# setup the environment
env = Snake(board_size=board_size, frames=frames)
s = env.reset()
# set the max time limit manually
env._max_time_limit = 48

# setup the agent
K.clear_session()
agent = QLearningAgent(board_size=board_size, frames=frames, buffer_size=100000)

# for iteration in [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]:
for iteration in [3000, 5000, 7000, 9000, 10000]:
    agent.load_model('models/v04', iteration=iteration)
    visualize_game(env, agent, path='images/game_visual_v4_{:d}.png'.format(iteration), debug=True)
