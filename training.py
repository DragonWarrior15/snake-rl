'''
script for training the agent for snake using various methods
'''
# run on cpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from tqdm import tqdm
from collections import deque
import pandas as pd
import time
from utils import play_game
from game_environment import Snake
import tensorflow as tf
from agent import DeepQLearningAgent, PolicyGradientAgent, AdvantageActorCriticAgent

# some global variables
tf.random.set_seed(42)
board_size = 10
frames = 2 # keep frames >= 2
version = 'v15'
max_time_limit = 998 # 998


# setup the environment
env = Snake(board_size=board_size, frames=frames, max_time_limit=max_time_limit)
s = env.reset()
n_actions = env.get_num_actions()

# setup the agent
agent = DeepQLearningAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=40000)
# agent = PolicyGradientAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=2000)
# agent = AdvantageActorCriticAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=2000)
# agent.print_models()

# check in the same order as class hierarchy
if(isinstance(agent, DeepQLearningAgent)):
    agent_type = 'DeepQLearningAgent'
if(isinstance(agent, PolicyGradientAgent)):
    agent_type = 'PolicyGradientAgent'
if(isinstance(agent, AdvantageActorCriticAgent)):
    agent_type = 'AdvantageActorCriticAgent'
print('Agent is {:s}'.format(agent_type))

# setup the epsilon range and decay rate for epsilon
# define rewrad type and update frequency, see utils for more details
if(agent_type in ['DeepQLearningAgent']):
    epsilon, epsilon_end = 1, 0.01
    reward_type = 'current'
    sample_actions = False
    n_games_training = 5
    decay = 0.99
    if(False):
        # load the existing model from a supervised method
        # or some other pretrained model
        agent.load_model(file_path='models/{:s}'.format(version))
if(agent_type in ['PolicyGradientAgent']):
    epsilon, epsilon_end = -1, -1
    reward_type = 'discounted_future'
    sample_actions = True
    exploration_threshold = 0.1
    n_games_training = 16
    decay = 1
if(agent_type in ['AdvantageActorCriticAgent']):
    epsilon, epsilon_end = -1, -1
    reward_type = 'current'
    sample_actions = True
    exploration_threshold = 0.1
    n_games_training = 10
    decay = 1

# define no of episodes, logging frequency
episodes = 2 * (10**4)
log_frequency = 500
# decay = np.exp(np.log((epsilon_end/epsilon))/episodes)

# use only for DeepQLearningAgent
if(agent_type in ['DeepQLearningAgent']):
    # play some games initially to fill the buffer
    _ = play_game(env, agent, n_actions, n_games=6000, record=True,
                    epsilon=epsilon, verbose=True, reset_seed=False)

# training loop
model_logs = {'iteration':[], 'reward_mean':[], 'reward_dev':[], 'loss':[]}
for index in tqdm(range(episodes)):
    # make small changes to the buffer and slowly train
    _ = play_game(env, agent, n_actions, epsilon=epsilon,
                        n_games=n_games_training, record=True,
                    sample_actions=sample_actions, reward_type=reward_type)
    loss = agent.train_agent(batch_size=64, num_games=n_games_training, reward_clip=True)

    if(agent_type in ['PolicyGradientAgent', 'AdvantageActorCriticAgent']):
        # for policy gradient algorithm, we only take current episodes for training
        agent.reset_buffer()

    # check performance every once in a while
    if((index+1)%log_frequency == 0):
        model_logs['loss'].append(loss)
        # keep track of agent rewards_history
        current_rewards = play_game(env, agent, n_actions, n_games=10, epsilon=-1,
                                    record=False, sample_actions=False)
        model_logs['iteration'].append(index+1)
        model_logs['reward_mean'].append(round(np.mean(current_rewards), 2))
        model_logs['reward_dev'].append(round(np.std(current_rewards), 2))
        pd.DataFrame(model_logs)[['iteration', 'reward_mean', 'reward_dev', 'loss']].to_csv('model_logs/{:s}.csv'.format(version), index=False)

    # copy weights to target network and save models
    if((index+1)%log_frequency == 0):
        agent.update_target_net()
        agent.save_model(file_path='models/{:s}'.format(version), iteration=(index+1))
        # keep some epsilon alive for training
        epsilon = max(epsilon * decay, epsilon_end)
