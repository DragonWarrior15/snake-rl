from game_environment import Snake
import numpy as np

env = Snake()
s = env.reset()
env.print_game()

for i in range(10):
    action = np.random.choice([-1, 0, 1], 1)[0]
    print(action)
    s = env.step(action)
    print(env._snake_direction)
    for i, x in enumerate(env._snake):
        print(i, x.row, x.col)
    env.print_game()
