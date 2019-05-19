'''
script for training the agent for snake using q learning
'''
# run on cpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras.backend as K
import numpy as np
from agent import DeepQLearningAgent, PolicyGradientAgent
from game_environment import Snake
from tqdm import tqdm
from collections import deque
import pandas as pd
from utils import play_game
import time

# some global variables
board_size = 10
frames = 2
version = 'v11'


# setup the environment
env = Snake(board_size=board_size, frames=frames)
s = env.reset()
n_actions = env.get_num_actions()

# setup the agent
agent_type = 'PolicyGradientAgent'
K.clear_session()
# agent = DeepQLearningAgent(board_size=board_size, frames=frames, buffer_size=60000)
agent = PolicyGradientAgent(board_size=board_size, frames=frames, buffer_size=60000)
# agent.print_models()

# setup the epsilon range and decay rate for epsilon
if(agent_type == 'DeepQLearningAgent'):
    epsilon, epsilon_end = 1, 0.01
else:
    epsilon, epsilon_end = -1, -1
episodes = 10000 # 2 * (10**5)
decay = 0.99
log_frequency = 200
# decay = np.exp(np.log((epsilon_end/epsilon))/episodes)

# use only for DeepQLearningAgent
if(agent_type == 'DeepQLearningAgent'):
    # play some games initially and train the model
    _ = play_game(env, agent, n_actions, n_games=6000, record=True, epsilon=epsilon, verbose=True, reset_seed=False)

# training loop
model_logs = {'iteration':[], 'reward_mean':[], 'reward_dev':[], 'loss':[]}
for index in tqdm(range(episodes)):
    # make small changes to the buffer and slowly train
    _ = play_game(env, agent, n_actions, epsilon=epsilon, n_games=1, record=True)
    loss = agent.train_agent(batch_size=64)
    if(agent_type in ['PolicyGradientAgent']):
        agent.reset_buffer()
    # check performance every once in a while
    if((index+1)%log_frequency == 0):
        model_logs['loss'].append(loss)
        # keep track of agent rewards_history
        current_rewards = play_game(env, agent, n_actions, n_games=10, epsilon=-1,
                                    record=False)
        model_logs['iteration'].append(index+1)
        model_logs['reward_mean'].append(np.mean(current_rewards))
        model_logs['reward_dev'].append(np.std(current_rewards))
        pd.DataFrame(model_logs)[['iteration', 'reward_mean', 'reward_dev', 'loss']].to_csv('model_logs/{:s}.csv'.format(version), index=False)

    # copy weights to target network and save models
    if((index+1)%log_frequency == 0):
        agent.update_target_net()
        agent.save_model(file_path='models/{:s}'.format(version), iteration=(index+1))
        # keep some epsilon alive for training
        epsilon = max(epsilon * decay, epsilon_end)
