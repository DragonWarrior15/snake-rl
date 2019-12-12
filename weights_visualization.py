import numpy as np
from agent import DeepQLearningAgent
from game_environment import Snake
import matplotlib.pyplot as plt
from tensorflow.keras import Model

# some global variables
board_size = 10
frames = 2

# setup the environment
env = Snake(board_size=board_size, frames=frames)
s = env.reset()
print(s[:,:,0])
# setup the agent
agent = DeepQLearningAgent(board_size=board_size, frames=frames, buffer_size=1)

# load weights into the agent
agent.load_model(file_path='models/v15/', iteration=0)

'''
# make some moves
for i in range(3):
    env.print_game()
    action = agent.move(s)
    next_s, _, _, _ = env.step(action)
    s = next_s.copy()
env.print_game()
'''

# define temporary model to get intermediate outputs
model_temp = Model(inputs=agent._model.input, outputs=agent._model.layers[2].output)
output_temp = model_temp.predict(s.reshape(1, board_size, board_size, frames))[0,:,:,:]
print('selected layer shape : ', output_temp.shape)
# visualize weights
# n_output_figs = int(output_temp.shape[2] ** 0.5)
fig, axs = plt.subplots(8, 4, figsize=(17, 17))
for i in range(8):
    for j in range(4):
        axs[i][j].imshow(output_temp[:,:,i*4+j], cmap='gray')
plt.show()
