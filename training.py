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

# some global variables
board_size = 11
frames = 4

def play_game(env, agent, n_games=100, record=True,
              verbose=False):
    '''
    function to play some games and return the rewards list
    '''
    rewards = []
    iterator = range(n_games)
    for _ in (tqdm(iterator) if verbose else iterator):
        rewards.append(0)
        s = env.reset()
        done = 0
        while(not done):
            action = agent.move(s)
            next_s, reward, done, info = env.step(action)
            if(record):
                agent.add_to_buffer(s, action, reward, next_s, done)
            rewards[-1] += reward
            s = next_s.copy()
    return rewards

# setup the environment
env = Snake(board_size=board_size, frames=frames)
s = env.reset()

# setup the agent
K.clear_session()
agent = QLearningAgent(board_size=board_size, frames=frames, buffer_size=20000)
# agent.print_models()

# setup the epsilon range and decay rate for epsilon
epsilon = 0.9
epsilon_end = 0.01
episodes = 10**5
# decay = np.exp(np.log((epsilon_end/epsilon))/episodes)
decay = 0.98

# play some games initially and train the model
agent.set_epsilon(epsilon)
for i in range(3):
    _ = play_game(env, agent, n_games=1000//3, record=True, verbose=True)
# _ = agent.train_agent(sample_size=5000, epochs=20)

rewards_history = []
loss_history = []
# training loop
for index in tqdm(range(episodes)):
    # make small changes to the buffer and slowly train
    _ = play_game(env, agent, n_games=5, record=True)
    loss_history.append(agent.train_agent(sample_size=128, epochs=1))
    # check performance every once in a while
    if((index+1)%100 == 0):
        # keep track of agent rewards_history
        agent.set_epsilon(0)
        current_rewards = play_game(env, agent, n_games=20, record=False)
        rewards_history.append(np.mean(current_rewards))
        # if episodes is large, print only once in a while
        print('Current Reward mean, dev : {:.2f}, {:.3f}'.\
                        format(np.mean(current_rewards), np.std(current_rewards)))
        print('Current MA Reward mean : {:.2f}'.format(np.mean(rewards_history[-20:])))
        print('Current MA Loss mean : {:.3f}'.format(np.mean(loss_history[-20:])))
        # ewma(rewards_history, span=20, min_periods=20)
    # copy weights to target network and save models
    if((index+1)%500 == 0):
        agent.update_target_net()
        agent.save_model(file_path='models/', iteration=(index+1))
        # keep some epsilon alive for training
        epsilon = max(epsilon * decay, epsilon_end)
    agent.set_epsilon(epsilon)
