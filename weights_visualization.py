import keras.backend as K
from keras.models import Model
import numpy as np
from agent import QLearningAgent
from game_environment import Snake
from tqdm import tqdm
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt

# some global variables
board_size = 11
frames = 4

# setup the environment
env = Snake(board_size=board_size, frames=frames)
s = env.reset()

# setup the agent
K.clear_session()
agent = QLearningAgent(board_size=board_size, frames=frames, buffer_size=20000)

# load weights into the agent
agent.load_model(file_path='models/', iteration=46000)

agent.set_epsilon(0)
# make some moves
for i in range(3):
    env.print_game()
    action = agent.move(s)
    next_s, _, _, _ = env.step(action)
    s = next_s.copy()
env.print_game()

# define temporary model to get intermediate outputs
model_temp = Model(inputs=agent._model_pred.input, outputs=agent._model_pred.layers[1].output)
output_temp = model_temp.predict(s.reshape(1, board_size, board_size, frames))[0,:,:,:]

# visualize weights
n_output_figs = int(output_temp.shape[2] ** 0.5)
fig, axs = plt.subplots(n_output_figs, n_output_figs, figsize=(17, 17))
for i in range(n_output_figs):
    for j in range(n_output_figs):
        axs[i][j].imshow(output_temp[:,:,i*n_output_figs+j], cmap='gray')
plt.show()
