'''
script for training the agent for snake using various methods
'''
# run on cpu
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import numpy as np
from tqdm import tqdm
from collections import deque
import pandas as pd
import time
from utils import play_game2
from game_environment import SnakeNumpy
from agent import BreadthFirstSearchAgent, SupervisedLearningAgent
import json

# some global variables
version = 'v15.5'

# get training configurations
with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames'] # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])

max_time_limit = 28 # 998
generate_training_data = False
do_training = True
n_games_training = 100

# setup the environment
env = SnakeNumpy(board_size=board_size, frames=frames, games=n_games_training,
                 max_time_limit=max_time_limit, obstacles=obstacles, version=version)
n_actions = env.get_num_actions()

if(generate_training_data):
    '''
    first generate the training data using a perfect agent
    only play with this agent for upto 3 or 4 points so that
    the reinforcement learning agent can get a head start
    but still have a good amount of playing to be done
    while its training
    '''
    # generate training data
    agent = BreadthFirstSearchAgent(board_size=board_size, frames=frames, 
                                    n_actions=n_actions, buffer_size=60000,
                                    version=version)
    for index in tqdm(range(1)):
        # make small changes to the buffer and slowly train
        curr_time = time.time()
        _, _, _ = play_game2(env, agent, n_actions, epsilon=-1,
                       n_games=n_games_training, record=True, 
                       reward_type='current', frame_mode=True, 
                       total_frames=60000, stateful=True)

        print('Buffer size {:d} filled in {:.2f}s'.format(agent.get_buffer_size(), 
                                                          time.time()-curr_time))
        file_path = 'models/{:s}'.format(version)
        if(not os.path.exists(file_path)):
            os.mkdir(file_path)
        agent.save_buffer(file_path=file_path, iteration=(index+1))
        agent.reset_buffer()

if(do_training):
    # setup the agent
    agent = SupervisedLearningAgent(board_size=board_size, frames=frames, 
                                    n_actions=n_actions, buffer_size=1,
                                    version=version)
    # agent.print_models()
    total_files = 1
    for index in tqdm(range(1 * total_files)):
        # read the saved training data
        file_path = 'models/{:s}'.format(version)
        agent.load_buffer(file_path=file_path, iteration=((index%total_files)+1))
        print(agent.get_buffer_size())
        # make small changes to the buffer and slowly train
        loss = agent.train_agent(epochs=20)
        print('Loss at buffer {:d} is : {:.5f}'.format((index%total_files)+1, loss))
    '''
    before saving the model, normalize the output layer weights
    to prevent explosion in outputs, keep track of max of output
    during training, inspired from arXiv:1709.04083v2
    '''
    agent.normalize_layers(agent.get_max_output())
    agent.update_target_net()
    # save the trained model
    agent.save_model(file_path='models/{:s}'.format(version))

