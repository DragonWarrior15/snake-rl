'''
script for training the agent for snake using q learning
'''
import keras.backend as K
import numpy as np
from agent import QLearningAgent
from game_environment import Snake
from tqdm import tqdm
from collections import deque
from pandas import ewma

# some global variables
board_size = 10

def play_game(env, agent, n_games=100, record=True,
              verbose=False):
    '''
    function to play some games and return the rewards list
    '''
    rewards = []
    buffer_add = {'s':deque(), 'next_s':deque(), 'r':deque(),
                  'a':deque(), 'done':deque()}
    iterator = range(n_games)
    for _ in (tqdm(iterator) if verbose else iterator):
        rewards.append(0)
        s = env.reset()
        done = 0
        while(not done):
            action = agent.move(s)
            next_s, reward, done, info = env.step(action)
            if(record):
                # agent.add_to_buffer(s, next_s, reward, action, done)
                buffer_add['s'].append(s)
                buffer_add['next_s'].append(next_s)
                buffer_add['r'].append(reward)
                buffer_add['a'].append(action)
                buffer_add['done'].append(done)
            rewards[-1] += reward
            s = next_s.copy()
    # batch add to the buffer
    if(record):
        agent.add_to_buffer_batch(buffer_add['s'], buffer_add['next_s'],
                    buffer_add['r'], buffer_add['a'], buffer_add['done'])
    return rewards

# setup the environment
env = Snake(board_size=board_size)
s = env.reset()

# setup the agent
K.clear_session()
agent = QLearningAgent(board_size=board_size, frames=4)
# agent.print_models()

# setup the epsilon range and decay rate for epsilon
epsilon = 0.9
epsilon_end = 0.01
episodes = 500
decay = np.exp(np.log((epsilon_end/epsilon))/episodes)

# play some games initially and train the model
agent.set_epsilon(epsilon)
_ = play_game(env, agent, n_games=1000, record=True, verbose=True)
_ = agent.train_agent()

rewards_history = []
# training loop
for index in tqdm(range(episodes)):
    _ = play_game(env, agent, n_games=500, record=True)
    _ = agent.train_agent()

    # keep track of agent rewards_history
    agent.set_epsilon(0)
    rewards_history.append(np.mean(play_game(env, agent,
                                        n_games=100, record=False)))
    print('Current MA Reward : {:.2f}'.format(rewards_history[-1]))
    # ewma(np.array(rewards_history), span=20, min_periods=20)
    agent.set_epsilon(epsilon)

    # copy weights to target network
    # save models
    if((index+1)%50 == 0):
        agent.update_target_net()
        agent.save_model(file_path='models/')

    epsilon *= decay
