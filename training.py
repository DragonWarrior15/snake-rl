'''
script for training the agent for snake using q learning
'''
import keras.backend as K
import numpy as np
from agent import QLearningAgent
from game_environment import Snake
from tqdm import tqdm

# some global variables
board_size = 10

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
                agent.add_to_buffer(s, next_s, reward, action, done)
            rewards[-1] += reward
    return rewards

# setup the environment
env = Snake(board_size=board_size)
s = env.reset()

# setup the agent
K.clear_session()
agent = QLearningAgent(board_size=board_size, frames=4)
# agent.print_models()

# play some games initially and train the model
_ = play_game(env, agent, n_games=1000, record=True, verbose=True)
_ = agent.train_agent()

# training loop
for index in tqdm(range(500)):
    _ = play_game(env, agent, n_games=1000, record=True)
    _ = agent.train_agent()

    # keep track of agent rewards
    print('Current Reward : {:.2f}'.format(np.mean(play_game(env, agent, n_games=100, record=False))))

    # copy weights to target network
    # save models
    if((index+1)%50 == 0):
        agent.update_target_net()
        agent.save_model(file_path='models/')
