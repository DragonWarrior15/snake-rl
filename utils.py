# some utility functions for the project
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import numpy as np
import time
import pandas as pd
import sys

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
            if(record and (info['termination_reason'] != 'time_up')):
                agent.add_to_buffer(s, action, reward, next_s, done)
            rewards[-1] += reward
            s = next_s.copy()
    return rewards

def visualize_game(env, agent, path='images/game_visual.png', debug=False,
                    animate=False):
    print('Starting Visualization')
    game_images = []
    qvalues = []
    food_count = []
    color_map = {0: 'lightgray', 1: 'g', 2: 'lightgreen', 3: 'r', 4: 'darkgray'}
    s = env.reset()
    board_size = env.get_board_size()
    game_images.append([s[:,:,0], 0])
    done = 0
    while(not done):
        a = agent.move(s)
        next_s, r, done, info = env.step(a)
        qvalues.append(agent._get_qvalues(s)[0])
        food_count.append(info['food'])
        game_images.append([next_s[:,:,0], info['time']])
        s = next_s.copy()
        if(debug):
            print(info['time'], qvalues[-1], a, r, info['food'], done)
    qvalues.append([0,0,0])
    food_count.append(food_count[-1])
    print('Game ran for {:d} frames'.format(len(game_images)))
    # plot the game
    if(animation):
        fig, axs = plt.subplots(1, 1,
                        figsize=(board_size//2 + 1,board_size//2 + 1))
        anim = animation.FuncAnimation(fig, anim_frames_func,
                              frames=game_images,
                              blit=False, interval=10,
                              repeat=True, init_func=None,
                              fargs=(axs, color_map, food_count, qvalues))
        # anim.save(path, writer='imagemagick', fps=5) # too much memory intensive
        anim.save(path, writer=animation.writers['ffmpeg'](fps=15, metadata=dict(artist='Me'), bitrate=1800))
    else:
        ncols = 5
        nrows = len(game_images)//ncols + (1 if len(game_images)%ncols > 0 else 0)
        fig, axs = plt.subplots(nrows, ncols,
                    figsize=(board_size*ncols, board_size*nrows), squeeze=False)
        for i in range(nrows):
            for j in range(ncols):
                idx = i*ncols+j
                if(idx < len(game_images)):
                    # plot the individual small squares in the frame
                    axs[i, j] = anim_frames_func(game_images[idx], axs[i, j],
                                    color_map, food_count, qvalues)
                else:
                    fig.delaxes(axs[i, j])
        fig.savefig(path, bbox_inches='tight')

# some functions for matplotlib animation
def anim_init_func(axs):
    ''' initialize a plain grid to plot the board '''
    axs.clear
    return axs

def anim_frames_func(board_time, axs, color_map, food_count, qvalues):
    ''' use the board to prepare the actual game grid '''
    axs.clear()
    board, time = board_time
    board_size = board.shape[0]
    half_width = 1.0/(2*board_size)
    delta = 0.025*2*half_width
    half_width-=delta
    for i in range(board_size):
        for j in range(board_size):
            rect = Rectangle(((half_width+delta)*(2*j)+delta, (half_width+delta)*(2*(board_size-1-i))+delta),
                            width=2*half_width, height=2*half_width,
                            color=color_map[board[i, j]])
            axs.add_patch(rect)
    # axs[i][j].imshow(game_images[index], cmap='gray')
    title = 'time:{:d}, score:{:d}\nright:{:.2f}, nothing:{:.2f}, left:{:.2f}'.\
                format(time, food_count[time], *qvalues[time])
    axs.set_title(title)
    plt.tight_layout()
    return axs

def plot_from_logs(data, title="Rewards and Loss Curve for Agent"):
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
    axs[0].set_title(title)
    axs[1].plot(data['iteration'], data['loss'])
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Iteration')
    plt.tight_layout()
    plt.show()
