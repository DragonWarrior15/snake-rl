# some utility functions for the project
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def visualize_game(env, agent, path='images/game_visual.png'):
    game_images = []
    time = []
    qvalues = []
    s = env.reset()
    game_images.append(s[:,:,0].copy())
    done = 0
    while(not done):
        a = agent.move(s)
        next_s, r, done, info = env.step(a)
        game_images.append(next_s[:,:,0].copy())
        qvalues.append(agent._get_qvalues(s)[0])
        time.append(info['time'])
        s = next_s.copy()
    # plot the game
    ncols = 5
    nrows = len(game_images)//ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    for i in range(nrows):
        for j in range(ncols):
            index = i*ncols+j
            if(index < len(game_images)):
                axs[i][j].imshow(game_images[index], cmap='gray')
                title = 'time:{:2d}\nqvalues->right:{:.2f}, nothing:{:.2f}, left:{:.2f}'.format(time[index], *qvalues[index])
                axs[i][j].set_title(title)
    fig.savefig(path, bbox_inches='tight')
