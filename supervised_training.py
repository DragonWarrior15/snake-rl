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
from utils import play_game
from game_environment import Snake
from agent import HamiltonianCycleAgent, BreadthFirstSearchAgent, SupervisedLearningAgent

# some global variables
board_size = 10
frames = 2
version = 'v15.1'
max_time_limit = 18 # 998
generate_training_data = False
do_training = True


# setup the environment
env = Snake(board_size=board_size, frames=frames, max_time_limit=max_time_limit)
s = env.reset()
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
    # agent = HamiltonianCycleAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10000)
    agent = BreadthFirstSearchAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=40000)
    for index in tqdm(range(1)):
        # make small changes to the buffer and slowly train
        current_rewards = play_game(env, agent, n_actions, epsilon=-1,
                            n_games=2000, record=True, sample_actions=False,
                            reward_type='discounted_future')

        file_path = 'models/{:s}'.format(version)
        if(not os.path.exists(file_path)):
            os.mkdir(file_path)
        agent.save_buffer(file_path=file_path, iteration=(index+1))
        agent.reset_buffer()

if(do_training):
    # setup the agent
    agent = SupervisedLearningAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=1)
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
        agent.update_target_net()
    '''
    before saving the model, normalize the output layer weights
    to prevent explosion in outputs, keep track of max of output
    during training, inspired from arXiv:1709.04083v2
    '''
    agent.normalize_layers(agent.get_max_output())
    agent.update_target_net()
    # save the trained model
    agent.save_model(file_path='models/{:s}'.format(version))

