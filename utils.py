# some utility functions for the project
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import time
import pandas as pd

def play_game(env, agent, n_actions, n_games=100, epsilon=0.01, record=True,
              verbose=False, reset_seed=False):
    '''
    function to play some games and return the rewards list
    has reset seed option to keep the board exactly same every time
    if epsilon is being used, it should be between 0 to 1
    '''
    # epsilon = min(max(0, epsilon), 1)
    rewards = []
    iterator = tqdm(range(n_games)) if verbose else range(n_games)
    for _ in iterator:
        if(reset_seed):
            np.random.seed(429834)
        rewards.append(0)
        s = env.reset()
        done = 0
        while(not done):
            # use epsilon greedy policy to get next action
            if(np.random.random() <= epsilon):
                action = np.random.choice(list(range(n_actions)))
            else:
                action = agent.move(s)
            next_s, reward, done, info = env.step(action)
            if(record):
                agent.add_to_buffer(s, action, reward, next_s, done)
            rewards[-1] += reward
            s = next_s.copy()
    return rewards

def visualize_game(env, agent, path='images/game_visual.png', debug=False):
    game_images = []
    time = []
    qvalues = []
    board_size = env.get_board_size()
    half_width = 1.0/(2*board_size)
    delta = 0.025*2*half_width
    half_width-=delta
    color_map = {0: 'lightgray', 1: 'g', 2: 'b', 3: 'r', 4: 'darkgray'}
    s = env.reset()
    game_images.append(s[:,:,0].copy())
    done = 0
    while(not done):
        a = agent.move(s)
        next_s, r, done, info = env.step(a)
        qvalues.append(agent._get_qvalues(s)[0])
        time.append(info['time'])
        game_images.append(next_s[:,:,0].copy())
        s = next_s.copy()
        if(debug):
            print(time[-1], qvalues[-1], a, r, done)
    qvalues.append([0,0,0])
    time.append(time[-1]+1)
    # plot the game
    ncols = 5
    nrows = len(game_images)//ncols + (1 if len(game_images)%ncols > 0 else 0)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), squeeze=False)
    for i in range(nrows):
        for j in range(ncols):
            index = i*ncols+j
            if(index < len(game_images)):
                # plot the individual small squares in the frame
                for i2 in range(board_size):
                    for j2 in range(board_size):
                        rect = Rectangle(((half_width+delta)*(2*j2)+delta, (half_width+delta)*(2*(board_size-1-i2))+delta),
                                        width=2*half_width, height=2*half_width,
                                        color=color_map[game_images[index][i2, j2]])
                        axs[i][j].add_patch(rect)
                # axs[i][j].imshow(game_images[index], cmap='gray')
                title = 'time:{:2d}\nright:{:.2f}, nothing:{:.2f}, left:{:.2f}'.format(time[index], *qvalues[index])
                axs[i][j].set_title(title)
    fig.savefig(path, bbox_inches='tight')

def plot_from_logs(data):
    ''' utility function to plot the learning curves '''
    if(isinstance(data, str)):
        # read from file and plot
        data = pd.read_csv(data)
    elif(isinstance(data, dict)):
        # use the lists in dict to plot
        pass
    else:
        print('Provide a dictionary or file path for the data')
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].plot(data['iteration'], data['reward_mean'])
    axs[0].set_ylabel('Mean Reward')
    axs[1].plot(data['iteration'], data['loss'])
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Iteration')
    plt.show()
