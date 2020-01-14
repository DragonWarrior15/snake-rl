import numpy as np
from agent import DeepQLearningAgent
from game_environment import Snake
import matplotlib.pyplot as plt
from tensorflow.keras import Model

# some global variables
version = 'v15.1'

with open('model_config/{:s}.json', 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames'] # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = m['supervised']
    n_actions = m['n_actions']
    obstacles = m['obstacles']

# setup the environment
env = Snake(board_size=board_size, frames=frames)
s = env.reset()
print(s[:,:,0])
# setup the agent
agent = DeepQLearningAgent(board_size=board_size, frames=frames, n_actions=actions,
                           buffer_size=1, version=version)

# load weights into the agent
agent.load_model(file_path='models/{:s}/'.format(version), iteration=188000)

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
model_temp = Model(inputs=agent._model.input, outputs=agent._model.layers[1].output)
output_temp = model_temp.predict(s.reshape(1, board_size, board_size, frames))[0,:,:,:]
print('selected layer shape : ', output_temp.shape)
# visualize weights
# n_output_figs = int(output_temp.shape[2] ** 0.5)
fig, axs = plt.subplots(output_temp.shape[2]//4, 4, figsize=(17, 17))
for i in range(output_temp.shape[2]//4):
    for j in range(4):
        axs[i][j].imshow(output_temp[:,:,i*4+j], cmap='gray')
plt.show()
