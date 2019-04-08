'''
script for training the agent for snake using q learning
'''
import keras.backend as K
import numpy as np
from agent import QLearningAgent
from game_environment import Snake

# some global variables
board_size = 10

def play_game(env, agent, n_games=100, record=True):
    '''
    function to play some games and return the rewards list
    '''
    rewards = []
    for _ in range(100):
        rewards.append(0)
        s = env.reset()
        done = 0
        while(not done):
            action = agent.move(s)
            next_s, reward, done, info = env.step(action)
            if(record):
                agent.add_to_buffer(s, next_s, reward, action, done)
            reward[-1] += reward
    return rewards

# setup the environment
env = Snake(board_size=board_size)
s = env.reset()

# setup the agent
agent = QLearningAgent(board_size=board_size, frames=4)
K.clear_session()

# play some games initially and train the model
_ = play_game(env, agent, n_games=1000, record=True)
_ = agent.train_agent()

# training loop
for index in range(10):
    _ = play_game(env, agent, n_games=1000, record=True)
    print(agent.train_agent())

    # copy weights to target network
    if(index%50 == 0):
        agent.update_target_net()
