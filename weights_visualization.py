import numpy as np
from agent import DeepQLearningAgent, BreadthFirstSearchAgent
from game_environment import Snake
import matplotlib.pyplot as plt
from tensorflow.keras import Model
import json
import os
import sys

# some global variables
version = 'v15.1'
iteration = 188000

with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames'] # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])
max_time_limit = -1

# setup the environment
env = Snake(board_size=board_size, frames=frames, max_time_limit=max_time_limit)
s = env.reset()
# setup the agent
agent = DeepQLearningAgent(board_size=board_size, frames=frames, n_actions=n_actions,
                           buffer_size=1, version=version)

# load weights into the agent
agent.load_model(file_path='models/{:s}/'.format(version), iteration=iteration)

'''
# make some moves
for i in range(3):
    env.print_game()
    action = agent.move(s)
    next_s, _, _, _, _ = env.step(action)
    s = next_s.copy()
env.print_game()
'''

# define temporary model to get intermediate outputs
layer_num = 1
model_temp = Model(inputs=agent._model.input, outputs=agent._model.layers[layer_num].output)
output_temp = model_temp.predict(s.reshape(1, board_size, board_size, frames))[0,:,:,:]
print('selected layer shape : ', output_temp.shape)

# save layer weights
plt.clf()
w = agent._model.layers[layer_num].weights[0].numpy()
nrows, ncols = (w.shape[2]*w.shape[3])//8, 8
fig, axs = plt.subplots(nrows, ncols, figsize=(17, 17))
for i in range(nrows):
    for j in range(ncols):
        axs[i][j].imshow(w[:, :, j%2, i*(ncols//2)+(j//2)], cmap='gray')
fig.savefig('images/weight_visual_{:s}_{:04d}_conv{:d}.png'\
            .format(version, iteration, layer_num), 
            dpi=72, bbox_inches='tight')
# sys.exit()

done = 0
t = 0
fig = plt.figure(figsize=(17,17))
while(not done):
    # print(s[:,:,0])
    # print('##########')
    output_temp = model_temp.predict(s.reshape(1, board_size, board_size, frames))[0,:,:,:]
    # play game
    action = agent.move(s, env.get_legal_moves(), env.get_values())
    next_s, _, done, _, _ = env.step(action)
    # visualize weights, we will add the game state as well
    plt.clf()
    """
    fig, axs = plt.subplots(output_temp.shape[2]//4, 4, figsize=(17, 17))
    for i in range(output_temp.shape[2]//4):
        for j in range(4):
            axs[i][j].imshow(output_temp[:,:,i*4+j], cmap='gray')
    """
    nrows, ncols = output_temp.shape[2]//4, 4
    # add the game image
    ax = plt.subplot(nrows, ncols+2, (1, ncols+2+2))
    ax.imshow(s[:,:,0], cmap='gray')
    ax.set_title('Frame : {:d}\nCurrent board'.format(t))
    ax = plt.subplot(nrows, ncols+2, (2*(ncols+2)+1, 3*(ncols+2)+2))
    ax.imshow(s[:,:,1], cmap='gray')
    ax.set_title('Frame : {:d}\nPrevious board'.format(t))
    # add the convolutional layers
    for i in range(nrows):
        for j in range(ncols):
            ax = plt.subplot(nrows, ncols+2, i*(ncols+2) + (j+2) + 1)
            ax.imshow(output_temp[:,:,i*4+j], cmap='gray')
    fig.savefig('images/weight_visual_{:s}_{:02d}.png'.format(version, t), 
                dpi=72, bbox_inches='tight')
    # plt.show()
    # update current state
    s = next_s.copy()
    t += 1

os.system('ffmpeg -y -framerate 1 -pattern_type sequence -i "images/weight_visual_{:s}_%02d.png" \
          -c:v libx264 -pix_fmt gray images/weight_visual_{:s}_{:04d}_conv{:d}.mp4'\
          .format(version, version, iteration, layer_num))

for i in range(t):
    os.remove('images/weight_visual_{:s}_{:02d}.png'.format(version, i))

""" -t 40 specifies pick 40s of the video, fps=1 is 1 frame per second, -loop 0 is
loop till infinity
ffmpeg -t 40 -i images/activation_visual_v15.1_188000_conv1.mp4 -vf "fps=1,scale=1200:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 images\activation_visual_v15.1_188000_conv1.gif -y
"""
