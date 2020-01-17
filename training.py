'''
script for training the agent for snake using various methods
'''
# run on cpu
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from tqdm import tqdm
from collections import deque
import pandas as pd
import time
from utils import play_game, play_game2
from game_environment import Snake, SnakeNumpy
import tensorflow as tf
from agent import DeepQLearningAgent, PolicyGradientAgent,\
                AdvantageActorCriticAgent, mean_huber_loss
import json

# some global variables
tf.random.set_seed(42)
version = 'v17.1'

# get training configurations
with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames'] # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])
    buffer_size = m['buffer_size']

# define no of episodes, logging frequency
episodes = 2 * (10**5)
log_frequency = 500
games_eval = 8

# setup the agent
agent = DeepQLearningAgent(board_size=board_size, frames=frames, n_actions=n_actions, 
                           buffer_size=buffer_size, version=version)
# agent = PolicyGradientAgent(board_size=board_size, frames=frames, n_actions=n_actions, 
        # buffer_size=2000, version=version)
# agent = AdvantageActorCriticAgent(board_size=board_size, frames=frames, n_actions=n_actions, 
                                  # buffer_size=10000, version=version)
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
    n_games_training = 8*16
    decay = 0.97
    if(supervised):
        # lower the epsilon since some starting policy has already been trained
        epsilon = 0.01
        # load the existing model from a supervised method
        # or some other pretrained model
        agent.load_model(file_path='models/{:s}'.format(version))
        # agent.set_weights_trainable()
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
    n_games_training = 32
    decay = 1

# decay = np.exp(np.log((epsilon_end/epsilon))/episodes)

# use only for DeepQLearningAgent
if(agent_type in ['DeepQLearningAgent']):
    # play some games initially to fill the buffer
    # or load from an existing buffer (supervised)
    if(supervised):
        try:
            agent.load_buffer(file_path='models/{:s}'.format(version), iteration=1)
        except FileNotFoundError:
            pass
    else:
        # setup the environment
        games = 512
        env = SnakeNumpy(board_size=board_size, frames=frames, 
                    max_time_limit=max_time_limit, games=games,
                    frame_mode=True, obstacles=obstacles, version=version)
        ct = time.time()
        _ = play_game2(env, agent, n_actions, n_games=games, record=True,
                       epsilon=epsilon, verbose=True, reset_seed=False,
                       frame_mode=True, total_frames=games*64)
        print('Playing {:d} frames took {:.2f}s'.format(games*64, time.time()-ct))

env = SnakeNumpy(board_size=board_size, frames=frames, 
            max_time_limit=max_time_limit, games=n_games_training,
            frame_mode=True, obstacles=obstacles, version=version)
env2 = SnakeNumpy(board_size=board_size, frames=frames, 
            max_time_limit=max_time_limit, games=games_eval,
            frame_mode=True, obstacles=obstacles, version=version)

# training loop
model_logs = {'iteration':[], 'reward_mean':[],
              'length_mean':[], 'games':[], 'loss':[]}
for index in tqdm(range(episodes)):
    if(agent_type in ['DeepQLearningAgent']):
        # make small changes to the buffer and slowly train
        _, _, _ = play_game2(env, agent, n_actions, epsilon=epsilon,
                       n_games=n_games_training, record=True,
                       sample_actions=sample_actions, reward_type=reward_type,
                       frame_mode=True, total_frames=n_games_training, 
                       stateful=True)
        loss = agent.train_agent(batch_size=64,
                                 num_games=n_games_training, reward_clip=True)

    if(agent_type in ['AdvantageActorCriticAgent']):
        # play a couple of games and train on all
        _, _, total_games = play_game2(env, agent, n_actions, epsilon=epsilon,
                       n_games=n_games_training, record=True,
                       sample_actions=sample_actions, reward_type=reward_type,
                       frame_mode=True, total_games=n_games_training*2)
        loss = agent.train_agent(batch_size=agent.get_buffer_size(), 
                                 num_games=total_games, reward_clip=True)

    if(agent_type in ['PolicyGradientAgent', 'AdvantageActorCriticAgent']):
        # for policy gradient algorithm, we only take current episodes for training
        agent.reset_buffer()

    # check performance every once in a while
    if((index+1)%log_frequency == 0):
        # keep track of agent rewards_history
        current_rewards, current_lengths, current_games = \
                    play_game2(env2, agent, n_actions, n_games=games_eval, epsilon=-1,
                               record=False, sample_actions=False, frame_mode=True, 
                               total_frames=-1, total_games=games_eval)
        
        model_logs['iteration'].append(index+1)
        model_logs['reward_mean'].append(round(int(current_rewards)/current_games, 2))
        # model_logs['reward_dev'].append(round(np.std(current_rewards), 2))
        model_logs['length_mean'].append(round(int(current_lengths)/current_games, 2))
        model_logs['games'].append(current_games)
        model_logs['loss'].append(loss)
        pd.DataFrame(model_logs)[['iteration', 'reward_mean', 'length_mean', 'games', 'loss']]\
          .to_csv('model_logs/{:s}.csv'.format(version), index=False)

    # copy weights to target network and save models
    if((index+1)%log_frequency == 0):
        agent.update_target_net()
        agent.save_model(file_path='models/{:s}'.format(version), iteration=(index+1))
        # keep some epsilon alive for training
        epsilon = max(epsilon * decay, epsilon_end)
