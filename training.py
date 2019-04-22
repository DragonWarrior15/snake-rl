'''
script for training the agent for snake using q learning
'''
# run on cpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras.backend as K
import numpy as np
from agent import QLearningAgent
from game_environment import Snake
from tqdm import tqdm
from collections import deque
import pandas as pd
from utils import play_game
import time

# some global variables
board_size = 10
frames = 2
version = 'v09'


# setup the environment
env = Snake(board_size=board_size, frames=frames)
s = env.reset()
n_actions = env.get_num_actions()

# setup the agent
K.clear_session()
agent = QLearningAgent(board_size=board_size, frames=frames, buffer_size=50000)
# agent.print_models()

# setup the epsilon range and decay rate for epsilon
epsilon = 1
epsilon_end = 0.01
episodes = 10**5
# decay = np.exp(np.log((epsilon_end/epsilon))/episodes)
decay = 0.99

# play some games initially and train the model
_ = play_game(env, agent, n_actions, n_games=5000, record=True, epsilon=epsilon, verbose=True, reset_seed=False)
# _ = agent.train_agent(batch_size=5000)

# training loop
model_logs = {'iteration':[], 'reward_mean':[], 'reward_dev':[], 'loss':[]}
for index in tqdm(range(episodes)):
    # make small changes to the buffer and slowly train
    _ = play_game(env, agent, n_actions, epsilon=epsilon, n_games=5, record=True)
    loss = agent.train_agent(batch_size=64)
    # check performance every once in a while
    if((index+1)%500 == 0):
        model_logs['loss'].append(loss)
        # keep track of agent rewards_history
        current_rewards = play_game(env, agent, n_actions, n_games=10, epsilon=-1,
                                    record=False)
        model_logs['iteration'].append(index+1)
        model_logs['reward_mean'].append(np.mean(current_rewards))
        model_logs['reward_dev'].append(np.std(current_rewards))
        pd.DataFrame(model_logs)[['iteration', 'reward_mean', 'reward_dev', 'loss']].to_csv('model_logs/{:s}.csv'.format(version), index=False)

    # copy weights to target network and save models
    if((index+1)%500 == 0):
        agent.update_target_net()
        agent.save_model(file_path='models/{:s}'.format(version), iteration=(index+1))
        # keep some epsilon alive for training
        epsilon = max(epsilon * decay, epsilon_end)
