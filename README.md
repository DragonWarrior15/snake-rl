# Snake Reinforcement Learning

Code for training a Deep Reinforcement Learning (DQN) agent to play the game of Snake.
The agent takes 2 frames of the game as input (image) and predicts the action values for
the next action to take.
***
<img align="left" width="350" height="350" src="https://github.com/DragonWarrior15/snake-rl/blob/master/images/game_visual_v10_174500_4.gif" alt="model v10 agent">
<img align="right" width="350" height="350" src="https://github.com/DragonWarrior15/snake-rl/blob/master/images/game_visual_v10_174500_0_.gif" alt="model v10 agent">
***

## Code Structure
[game_environment.py](../game_environment.py) contains the necessary code to create and interact with the snake environment (class Snake). The interface is similar to openai gym interface.

[agent.py](../agent.py) contains the agent for playing the game (class QLearningAgent). It implements and trains a convolutional neural network for the action values.

[training.py](../training.py) contains the complete code to train an agent.

[game_visualization.py](../game_visualization.py) contains the code to convert the game to mp4 format.

```python
from game_environment import Snake
from agent import QLearningAgent
import numpy as np
env = Snake(board_size=10, frames=2)
state = env.reset() # first manual reset required to initialize few variables
agent = QLearningAgent(board_size=10, frames=2, buffer_size=10000)
done = 0
total_reward = 0
epsilon = 0.1
while(not done):
    if(np.random.random() <= epsilon):
        action = np.random.randint(0, 3)
    else:
        action = agent.move(s)
    next_state, reward, done, info = env.step(action)
    # info contains time, food (food count), termination_reason (if ends)
    agent.add_to_buffer([state, action, reward, next_state, done])
    total_reward += reward
    state = next_state.copy()
agent.train_agent(batch_size=32) # perform one step of gradient descent
agent.update_target_net() # update the target network
```

## Experiments
Configuration for different experiments can be found in [model_versions.json][../model_versions.json] file.

Adam optimizer gives a very noisy curve with very slow increase in rewards. Loss is also not stable.
![alt text](https://github.com/DragonWarrior15/snake-rl/blob/master/images/model_logs_v04.png "model version v04")

Switching to RMSprop optimizer steers things in the right direction and the agent start to learn.
Key point to note is that the increasing rewards with length of snake send a strong signal for propagation across the different states. Although, the loss here is noisy. Current learning rate is 0.0005, and reducing it to 0.0001 makes the model relatively unstable.
![alt text](https://github.com/DragonWarrior15/snake-rl/blob/master/images/model_logs_v07.png "model version v07")

RMSprop with both positive and negative rewards increasing with snake length gives a more stable loss curve, and the rewards also steadily increase to quite high values.
![alt text](https://github.com/DragonWarrior15/snake-rl/blob/master/images/model_logs_v10.png "model version v10")
