# script to visualize how agent plays a game
# useful to study different iterations

import numpy as np
from agent import DeepQLearningAgent, PolicyGradientAgent, AdvantageActorCriticAgent
from game_environment import Snake
from utils import visualize_game
import keras.backend as K

# some global variables
board_size = 10
frames = 2
version = 'v13'
iteration_list = [9200]
max_time_limit = 98

# setup the environment
env = Snake(board_size=board_size, frames=frames, max_time_limit=max_time_limit)
s = env.reset()

# setup the agent
K.clear_session()
# agent = DeepQLearningAgent(board_size=board_size, frames=frames, buffer_size=10)
# agent = PolicyGradientAgent(board_size=board_size, frames=frames, buffer_size=10)
agent = AdvantageActorCriticAgent(board_size=board_size, frames=frames, buffer_size=10)

# for iteration in [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]:
for iteration in iteration_list:
    agent.load_model('models/{:s}'.format(version), iteration=iteration)
    for i in range(5):
        visualize_game(env, agent,
            path='images/game_visual_{:s}_{:d}_{:d}.mp4'.format(version, iteration, i),
            debug=False, animate=True)
