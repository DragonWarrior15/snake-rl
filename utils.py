# some utility functions for the project
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import numpy as np
import time
import pandas as pd
import sys

def calculate_discounted_rewards(rewards, discount_factor=0.99):
    """Utility to calculate rewards discounted for
    future values, useful in policy gradient, A2C
    
    Parameters
    ----------
    rewards : list
        list containing individual step rewards for a single game
    discount_factor : float, optional
        the discount factor to apply for accounting future rewards
        advisable to keep below one to get convergence
    
    Returns
    ------
    discounted rewards : list
        same size as rewads, but accounting for future discounts
    """
    discounted_rewards = np.zeros(rewards.shape, dtype=np.int16)
    discounted_rewards[rewards.shape[0]-1] = rewards[rewards.shape[0]-1]
    i = rewards.shape[0] - 2
    while(i > -1):
        discounted_rewards[i] = rewards[i] + discount_factor * discounted_rewards[i+1]
        i -= 1
    return discounted_rewards.copy()

def play_game(env, agent, n_actions, n_games=100, epsilon=0.01, record=True,
              verbose=False, reset_seed=False, sample_actions=False,
              reward_type='current'):
    '''
    function to play some games and return the rewards list
    has reset seed option to keep the board exactly same every time
    if epsilon is being used, it should be between 0 to 1
    use negative epsilon in case using on policy algorithms
    this function runs env sequentially
    play_game2 is for the parallel env implementation

    Parameters
    ----------
    env : Environment Object
    agent : Agent object
        interacts with env
    n_actions : int
        count of actions
    n_games : int, optional
        Total games to play
    epsilon : float, optional
        for epsilon greedy policy, value < 0 means
        no epsilon, and > 0 means always random policy
    record : bool, optional
        whether to add frames to agent buffer
    verbose : bool, optional
        whether to show progress (deprecated)
    reset_seed : bool, optional
        whether to reset numpy seed every time
    sample_actions : bool, optional
        whether to sample actions from agent probability dist
    reward_type : str, optional
        'current' or 'discounted future', latter used in policy gradient

    Returns
    -------
    rewards : list
        contains total reward values across n_games 
    '''
    # epsilon = min(max(0, epsilon), 1)
    rewards = []
    iterator = tqdm(range(n_games)) if verbose else range(n_games)
    for _ in iterator:
        if(reset_seed):
            np.random.seed(429834)
        rewards.append(0)
        s = env.reset()
        # done set only for first run of while loop
        done = 0
        # the following is useful for discounted rewards as not known in advance
        s_list, action_list, reward_list, next_s_list, done_list = [], [], [], [], []
        while(not done):
            if(np.random.random() <= epsilon):
                # use epsilon greedy policy to get next action
                action = np.random.choice(list(range(n_actions)))
            else:
                if(sample_actions):
                    # sample from prob dist
                    probs = agent.get_action_proba(s)
                    action = np.random.choice(n_actions, p=probs)
                else:
                    # get action with best q value
                    action = agent.move(s, env.get_values())
            next_s, reward, done, info = env.step(action)

            if(record and (info['termination_reason'] != 'time_up')):
                if(reward_type == 'current'):
                    agent.add_to_buffer(s, action, reward, next_s, done)
                elif(reward_type == 'discounted_future'):
                    # add everything later to the buffer
                    s_list.append(s.copy())
                    action_list.append(action)
                    reward_list.append(reward)
                    next_s_list.append(next_s.copy())
                    done_list.append(done)
                else:
                    assert reward_type in ['current', 'discounted_future'], \
                            'reward type not understood !'
            s = next_s.copy()
            # update current game total rewards
            rewards[-1] += reward
        # if using future discounted rewards, then add everything to buffer here
        if(record and reward_type == 'discounted_future'):
            reward_list = calculate_discounted_rewards(reward_list, agent.get_gamma())
            for i in range(len(reward_list)):
                agent.add_to_buffer(s_list[i], action_list[i], reward_list[i],\
                                    next_s_list[i], done_list[i])
    return rewards

def play_game2(env, agent, n_actions, n_games=100, epsilon=0.01, record=True,
              verbose=False, reset_seed=False, sample_actions=False,
              reward_type='current', frame_mode=False, total_frames=10,
              total_games=None, stateful=False, debug=False):
    '''
    function to play some games and return the rewards list
    has reset seed option to keep the board exactly same every time
    if epsilon is being used, it should be between 0 to 1
    use negative epsilon in case using on policy algorithms
    this function utilises the parallel numpy env
    play_game is for the basic env implementation

    Parameters
    ----------
    env : Environment Object
    agent : Agent object
        interacts with env
    n_actions : int
        count of actions
    n_games : int, optional
        Games to run in parallel
    epsilon : float, optional
        for epsilon greedy policy, value < 0 means
        no epsilon, and > 0 means always random policy
    record : bool, optional
        whether to add frames to agent buffer
    verbose : bool, optional
        whether to show progress (deprecated)
    reset_seed : bool, optional
        whether to reset numpy seed every time
    sample_actions : bool, optional
        whether to sample actions from agent probability dist
    reward_type : str, optional
        'current' or 'discounted future', latter used in policy gradient
    frame_mode : bool, optional
        whether to run parallel env based on count of frames or count of games
    total_frames : int, optional
        if running frame mode, total frames to process before stopping
    total_games : int, optional
        if running frame mode and this is not None, use the value to determine
        when to end, depending on total games played till the point
    stateful : bool, optional
        whether to preserve env state and not reset to beginning

    Returns
    -------
    rewards : np array
        contains total reward values across n_games, only useful
        when not using the frame_mode, otherwise values across multiple
        games, and may be incomplete
    lengths : int
        total sum of snake lengths across games that have terminated, not
        useful in stateful as env could be running from some intermediate state
    games : int
        total no of games that have terminated, not useful in stateful as
        env could be running from some intermediate state  
    '''
    rewards = 0 # to keep track of total reward across games
    lengths = 0 # to keep track of total length across all games
    if(reset_seed):
        np.random.seed(42)
    s = env.reset(stateful)
    # this done is just for first run of the while loop
    done = np.zeros((1,), dtype=np.uint8)
    # the following is useful for discounted rewards as not known in advance
    s_list, action_list, reward_list, next_s_list, done_list, legal_moves_list \
                = [], [], [], [], [], []
    frames, games = 0, 0

    '''3 conditions to check, 
    1) if not using frame mode then all games should not have ended
    2) if using frame mode and total games is not provided, then fames
       playes should be less than total frames asked for
    3) if using frame mode and total games is provded, then total games
       playes < total games asked for
    ''' 
    while(not frame_mode and not done.all()) or \
         (frame_mode and total_games is None and frames < total_frames) or\
         (frame_mode and total_games is not None and games < total_games):
        legal_moves = env.get_legal_moves()
        if(np.random.random() <= epsilon):
            # use epsilon greedy policy to get next action
            # action = np.random.choice(n_actions, n_games)
            action = np.argmax(np.where(legal_moves>0, 
                        np.random.random((n_games, n_actions)),-1), axis=1)
        else:
            # else select action using agent outputs
            if(sample_actions):
                # sample from prob dist
                probs = agent.get_action_proba(s)
                # direct np.random.choice cannot be used on matrix
                # so we get cumsum and the generate random nos to select an "interval"
                # through which we can pick the action to be selected
                action = ((probs/probs.sum(axis=1).reshape(-1,1)).cumsum(axis=1)\
                          <np.random.random((probs.shape[0],1))).sum(axis=1)
                action[action==4] = 3
            else:
                # get action with best q value
                action = agent.move(s, legal_moves, env.get_values())
        # take 1 step in env across all games 
        next_s, reward, done, info, next_legal_moves = env.step(action)

        if(record):
            # handle (info['termination_reason'] != 'time_up') later
            if(reward_type == 'current'):
                agent.add_to_buffer(s, action, reward, next_s, done, 
                                    next_legal_moves)
            elif(reward_type == 'discounted_future'):
                # add everything later to the buffer
                s_list.append(s.copy())
                action_list.append(action)
                reward_list.append(reward)
                next_s_list.append(next_s.copy())
                done_list.append(done)
                legal_moves_list.append(next_legal_moves)
            else:
                assert reward_type in ['current', 'discounted_future'], \
                        'reward type not understood !'
        s = next_s.copy()
        rewards += np.dot(done, info['cumul_rewards'])
        frames += n_games
        games += done.sum()
        # get only lengths where game ended
        lengths += np.dot(done, info['length'])

    # if using future discounted rewards, then add everything to buffer here
    if(record and reward_type == 'discounted_future'):
        reward_list = calculate_discounted_rewards(reward_list, agent.get_gamma())
        for i in range(len(reward_list)):
            agent.add_to_buffer(s_list[i], action_list[i], reward_list[i],\
                                next_s_list[i], done_list[i], legal_moves_list[i])
    
    # since not frame mode, calculate lenghts at the end to avoid
    # double counting 
    if(not frame_mode):
        lengths = np.dot(done, info['length'])
        rewards = np.dot(done, info['cumul_rewards'])
    
    return rewards, lengths, games

def visualize_game(env, agent, path='images/game_visual.png', debug=False,
                    animate=False, fps=10):
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
        # print('frame no ', len(game_images))
        legal_moves = env.get_legal_moves()
        a = agent.move(s, legal_moves, env.get_values())
        next_s, r, done, info, _ = env.step(a)
        qvalues.append(agent._get_model_outputs(s)[0])
        food_count.append(info['food'])
        game_images.append([next_s[:,:,0], info['time']])
        s = next_s.copy()
        if(debug):
            print(info['time'], qvalues[-1], a, r, info['food'], done, legal_moves)
    qvalues.append([0] * env.get_num_actions())
    food_count.append(food_count[-1])
    print('Game ran for {:d} frames'.format(len(game_images)))
    # append a few static frames in the end for pause effect
    for _ in range(5):
        qvalues.append(qvalues[-1])
        food_count.append(food_count[-1])
        game_images.append(game_images[-1])
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
        anim.save(path, writer=animation.writers['ffmpeg'](fps=fps, metadata=dict(artist='Me'), bitrate=1800))
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
    title = 'time:{:d}, score:{:d}\n{:.2f} {:.2f} {:.2f} {:.2f}'.\
                format(time, food_count[time], *qvalues[time])
    axs.set_title(title)
    plt.tight_layout()
    return axs

def plot_logs(data, title="Rewards and Loss Curve for Agent",
                    loss_titles=['Loss']):
    '''
    utility function to plot the learning curves
    loss_index is only applicable if the object is a
    example usage:
    python -c "from utils import plot_logs; plot_logs('model_logs/v15.2.csv')"
    python -c "from utils import plot_logs; plot_logs('model_logs/v15.3.csv', loss_titles=['Total Loss', 'Actor Loss', 'Critic Loss'])"
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
    fig, axs = plt.subplots(1 + loss_count + 1 if 'length_mean' in data.columns else 0, 1, figsize=(8, 8))
    axs[0].set_title(title)
    index = 0
    if('length_mean' in data.columns):
        axs[0].plot(data['iteration'], data['length_mean'])
        axs[0].set_ylabel('Mean Length')
        index = 1        
    
    axs[index].plot(data['iteration'], data['reward_mean'])
    axs[index].set_ylabel('Mean Reward')
    index += 1

    for i in range(index, index+loss_count):
        axs[i].plot(data['iteration'], data['loss_{:d}'.format(i-index) if loss_count > 1 else 'loss'])
        axs[i].set_ylabel(loss_titles[i-index])
        axs[i].set_xlabel('Iteration')
    plt.tight_layout()
    plt.show()
