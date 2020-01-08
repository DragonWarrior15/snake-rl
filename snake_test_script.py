from game_environment_parallel import Snake
import numpy as np

env = Snake(board_size=10, frames=2, n_games=3)
s = env.reset()
env.print_game()
'''
done = 0
while(not done):
    # action = np.random.choice([-1, 0, 1], 1)[0]
    # instead of random action, take input from user
    action = int(input('Enter action [-1, 0, 1] : '))
    # print(action)
    s, r, done, info = env.step(action)
    # print(env._snake_direction)
    # for i, x in enumerate(env._snake):
        # print(i, x.row, x.col)
    env.print_game()
'''