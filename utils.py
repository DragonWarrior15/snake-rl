# some utility functions for the project
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import numpy as np
import time
import pandas as pd
import sys

def calculate_discounted_rewards(rewards, discount_factor):
    discounted_rewards = []
    for i in range(len(rewards)-1, -1, -1):
        if(i == len(rewards)-1):
            discounted_rewards.append(rewards[i])
        else:
            discounted_rewards.append(discounted_rewards[-1] * discount_factor + rewards[i])
    discounted_rewards = discounted_rewards[::-1]
    return discounted_rewards.copy()

def play_game(env, agent, n_actions, n_games=100, epsilon=0.01, record=True,
              verbose=False, reset_seed=False, sample_actions=False,
              reward_type='current'):
    '''
    function to play some games and return the rewards list
    has reset seed option to keep the board exactly same every time
    if epsilon is being used, it should be between 0 to 1
    use negative epsilon in case using on policy algorithms
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
        s_list, action_list, reward_list, next_s_list, done_list = [], [], [], [], []
        while(not done):
            # use epsilon greedy policy to get next action
            if(np.random.random() <= epsilon):
                action = np.random.choice(list(range(n_actions)))
            else:
                if(sample_actions):
                    probs = agent.get_action_proba(s)
                    action = np.random.choice(list(range(n_actions)), p=probs)
                else:
                    action = agent.move(s)
            next_s, reward, done, info = env.step(action)
            if(record and (info['termination_reason'] != 'time_up')):
                if(reward_type == 'current'):
                    agent.add_to_buffer(s, action, reward, next_s, done)
                elif(reward_type == 'discounted_future'):
                    s_list.append(s)
                    action_list.append(action)
                    reward_list.append(reward)
                    next_s_list.append(next_s)
                    done_list.append(done)
                else:
                    assert reward_type in ['current', 'discounted_future'], \
                            'reward type not understood !'
            rewards[-1] += reward
            s = next_s.copy()
        # if using future discounted rewards, then add everything to buffer here
        if(record and reward_type == 'discounted_future'):
            reward_list = calculate_discounted_rewards(reward_list, agent.get_gamma())
            for i in range(len(reward_list)):
                agent.add_to_buffer(s_list[i], action_list[i], reward_list[i],\
                                    next_s_list[i], done_list[i])
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
        qvalues.append(agent._get_model_outputs(s)[0])
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
        anim.save(path, writer=animation.writers['ffmpeg'](fps=10, metadata=dict(artist='Me'), bitrate=1800))
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

def plot_from_logs(data, title="Rewards and Loss Curve for Agent",
                    loss_titles=['Loss']):
    '''
    utility function to plot the learning curves
    loss_index is only applicable if the object is a
    example usage:
    plot_from_logs('model_logs/v12.csv', loss_titles=['Total Loss', 'Policy Gradient Loss', 'Entropy'])
    plot_from_logs('model_logs/v11.csv')
    '''
    loss_count = 1
    if(isinstance(data, str)):
        # read from file and plot
        data = pd.read_csv(data)
        if(data['loss'].dtype == 'O'):
            # get no of values in loss
            loss_count = len(data.iloc[0, data.columns.tolist().index('loss')].replace('[', '').replace(']', '').split(','))
            for i in range(loss_count):
                data['loss_{:d}'.format(i)] = data['loss'].apply(lambda x: float(x.replace('[', '').replace(']', '').split(',')[i]))
            if(len(loss_titles) != loss_count):
                loss_titles = loss_titles[0] * loss_count
    elif(isinstance(data, dict)):
        # use the lists in dict to plot
        pass
    else:
        print('Provide a dictionary or file path for the data')
    fig, axs = plt.subplots(1 + loss_count, 1, figsize=(8, 8))
    axs[0].plot(data['iteration'], data['reward_mean'])
    axs[0].set_ylabel('Mean Reward')
    axs[0].set_title(title)
    for i in range(loss_count):
        axs[i+1].plot(data['iteration'], data['loss_{:d}'.format(i) if loss_count > 1 else 'loss'])
        axs[i+1].set_ylabel(loss_titles[i])
        axs[i+1].set_xlabel('Iteration')
    plt.tight_layout()
    plt.show()
