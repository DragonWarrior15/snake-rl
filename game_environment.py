'''
This script stores the game environment. Note that the snake is a part of the
environment itself in this implementation.
The environment state is a set of frames, we want the agent to be able to discern
the movement of the snake as well, for which multiple frames are needed.
We will keep track of a history of 2 frames.
Important to manually reset the environment by user after initialization.
The board borders are different from board color
The player cannot play infinitely, and hence max time limit is imposed
'''

import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class Position:
    '''
    Class for defining any position on a 2D grid
    Attributes:
        row (int) : contains the row for a 2D grid
        col (int) : contains the column for a 2D grid
    '''
    def __init__(self, row = 0, col = 0):
        self.row = row
        self.col = col

    def set_position(self, row = None, col = None):
        ''' modify the existing position coordinate '''
        if(row is not None):
            self.row = row
        if(col is not None):
            self.col = col

class Snake:
    '''
    Class for the snake game.
    Attributes:
        size (int) : size of the game board, assumed to be square
        board : numpy array containing information about various objects in the
                board, including snake, food and obstacles
        snake_length (int) : current length of the snake
        snake_head (int, int) : Position containing the row and column no of board
                                for snake head
        food (int, int) : Position containing the coordinates for the food
        snake_direction (int) : direction (left:2, right:0, up:1, down:3) where
                                   snake is moving
        snake (queue) : a queue to store the positions of the snake body
    '''
    def __init__(self, board_size=11, frames=4, start_length=5, seed=42,
                 max_time_limit=998):
        '''
        Initialization function for the environment.
        '''
        # self._value = {'snake':255, 'board':0, 'food':128, 'head':180, 'border':80}
        self._value = {'snake':1, 'board':0, 'food':3, 'head':2, 'border':4}
        self._actions = [-1, 0, 1] # -1 left, 0 nothing, 1 right
        self._n_actions = 3
        self._board_size = board_size
        self._n_frames = frames
        self._rewards = {'out':-10, 'food':10, 'time':0, 'no_food':0}
        # start length is constrained to be less than half of board size
        self._start_length = min(start_length, (board_size-2)//2)
        # set numpy seed for reproducible results
        # np.random.seed(seed)
        # time limit to contain length of game
        self._max_time_limit = max_time_limit
        # other variables that can be quickly reused across multiple games
        self._static_board_template = self._value['board'] * np.ones((self._board_size, self._board_size))
        # make board borders
        self._static_board_template[:, 0] = self._value['border']
        self._static_board_template[:, self._board_size-1] = self._value['border']
        self._static_board_template[0, :] = self._value['border']
        self._static_board_template[self._board_size-1, :] = self._value['border']
        # variable to hold all positions where food can be put

    def _queue_to_board(self):
        '''
        Convert the current queue of frames to a 3D matrix
        Returns:
            board : np array of 3 dimensions
        '''
        board = np.dstack([x for x in self._board])
        return board.copy()

    def print_game(self):
        ''' prints the current state (board) '''
        fig, axs = plt.subplots(1, self._n_frames)
        for i in range(self._n_frames):
            axs[i].imshow(self._board[i], cmap = 'gray')
        plt.show()

    def get_board_size(self):
        ''' get board_size '''
        return self._board_size

    def get_n_frames(self):
        ''' get frame count '''
        return self._n_frames

    def reset(self):
        '''
        reset the environment
        Returns:
            board : the current board state
        '''
        board = self._static_board_template.copy()
        # initialize snake
        self._snake = deque()
        self._snake_length = self._start_length
        self._count_food = 0
        # modify the board values for the snake, assumed to be lying horizontally initially
        for i in range(1, self._snake_length+1):
            board[self._board_size//2, i] = self._value['snake']
            self._snake.append(Position(self._board_size//2, i))
        # modify the snake head position
        self._snake_head = Position(self._board_size//2, i)
        board[self._snake_head.row, self._snake_head.col] = self._value['head']
        # queue, left most entry is the latest frame
        self._board = deque(maxlen = self._n_frames)
        for i in range(self._n_frames):
            self._board.append(board.copy())
            # self._board.append((self._value['board'] * np.ones_like(board)).copy())

        # modify the food position on the board, after board queue initialized
        self._get_food()
        self._snake_direction = 0
        # set time elapsed to 0
        self._time = 0
        return self._queue_to_board()

    def get_num_actions(self):
        ''' get total count of actions '''
        return self._n_actions

    def _action_map(self, action):
        ''' converts integer to internatl action mapping '''
        return self._actions[action]

    def _get_snake_tail(self):
        '''
        get the head of the snake, right most element in the queue
        Returns:
        head : Position of the head
        '''
        return self._snake[0]

    def _get_food(self):
        '''
        find the coordinates of the point to put the food at
        first randomly locate a row to put the food in, then remove all
        the cells with snake and choose amongst the remaining
        since board has borders, food cannot be at them
        '''
        while(1):
            food_x = list(range(1,self._board_size-1))
            food_x = np.random.choice(food_x)
            food_y = [i for i in range(1, self._board_size-1) \
                        if self._board[0][food_x, i] == self._value['board']]
            if(len(food_y) == 0):
                continue
            else:
                food_y = np.random.choice(food_y)
                break
        self._food = Position(food_x, food_y)
        self._put_food()

    def _put_food(self):
        ''' put the food in the required spot '''
        self._board[0][self._food.row, self._food.col] = self._value['food']

    def _get_new_direction(self, action, current_direction):
        '''
        get the new direction after taking the specified action
        Returns:
            direction (int) : the new direction of motion
        '''
        direction = (current_direction + self._action_map(action))%4
        return direction

    def _get_new_head(self, action, current_direction):
        '''
        get the position for the new head through the action
        Returns:
            new_head (Position) : position class for the new head
        '''
        new_dir  = self._get_new_direction(action, current_direction)
        del_x, del_y = (new_dir%2)*(new_dir-2), (1-(new_dir%2))*(1-new_dir)
        new_head = Position(self._snake_head.row + del_x,
                            self._snake_head.col + del_y)
        return new_head

    def step(self, action):
        '''
        takes an action and performs one time step of the game, returns updated
        board
        Arguments:
            action (int) : should be among the possible actions
        Returns:
            board : updated after taking the step
            reward : agent's reward for performing the current action
            done : whether the game is over or not (1 or 0)
            info : any auxillary game information
        '''
        assert action in list(range(self._n_actions)), "Action must be in " + list(range(self._n_actions))
        reward, done = 0, 0

        # check if the current action is feasible
        reward, done, can_eat_food, termination_reason = self._check_if_done(action)
        if(done == 0):
            # if not done, move the snake
            self._move_snake(action, can_eat_food)
            # update the direction of motion
            self._snake_direction = self._get_new_direction(action, self._snake_direction)
            # get the next food location
            if(can_eat_food):
                self._get_food()

        # update time
        self._time += 1
        # info contains time elapsed etc
        info = {'time':self._time, 'food':self._count_food,
                'termination_reason':termination_reason}


        return self._queue_to_board(), reward, done, info

    def _get_food_reward(self):
        ''' try different rewards schemes for when food is eaten '''
        return((self._snake_length - self._start_length + 1) * self._rewards['food'])
        # return self._rewards['food']

    def _get_death_reward(self):
        ''' try different rewards schemes for death '''
        return((self._snake_length - self._start_length + 1) * self._rewards['out'])
        # return self._rewards['out']

    def _check_if_done(self, action):
        '''
        checks if the game has ended or if food has been taken
        Returns:
            reward : reward for the current action
            done : 1 if ended else 0
            can_eat_food : whether the current action leads to eating food
        '''
        reward, done, can_eat_food, termination_reason = \
                            self._rewards['time'], 0, 0, ''
        # check if the current action forces snake out of board
        new_head = self._get_new_head(action, self._snake_direction)
        while(1):
            # snake is colliding with border/obstacles
            if(self._board[0][new_head.row, new_head.col] == self._value['border']):
                done = 1
                reward = self._get_death_reward()
                termination_reason = 'collision_wall'
                break
            # collision with self, collision with tail is allowed
            if(self._board[0][new_head.row, new_head.col] == self._value['snake']):
                snake_tail = self._get_snake_tail()
                if(not(new_head.row == snake_tail.row and new_head.col == snake_tail.col)):
                    done = 1
                    reward = self._get_death_reward()
                    termination_reason = 'collision_self'
                    break
            # check if food
            if(self._board[0][new_head.row, new_head.col] == self._value['food']):
                done = 0
                reward += self._get_food_reward()
                self._count_food += 1
                can_eat_food = 1
            # check if time up
            if(self._time >= self._max_time_limit):
                done = 1
                # check if no food eaten
                if(self._snake_length == self._start_length and self._rewards['no_food'] != 0):
                    termination_reason = 'time_up_no_food'
                    reward += self._rewards['no_food']
                else:
                    termination_reason = 'time_up'
                break
            # if normal movement, no other updates needed
            break
        return reward, done, can_eat_food, termination_reason

    def _move_snake(self, action, can_eat_food):
        '''
        moves the snake using the given action
        and updates the board accordingly
        '''
        # get the coordinates for the new head
        new_head = self._get_new_head(action, self._snake_direction)
        # prepare new board as the last frame
        new_board = self._board[0].copy()
        # modify the next block of the snake body to be same color as snake
        new_board[self._snake_head.row, self._snake_head.col] = self._value['snake']
        # insert the new head into the snake queue
        # different treatment for addition of food
        # update the new board view as well
        # if new head overlaps with the tail, special handling is needed
        self._snake.append(new_head)
        self._snake_head = new_head

        if(can_eat_food):
            self._snake_length += 1
        else:
            delete_pos = self._snake.popleft()
            new_board[delete_pos.row, delete_pos.col] = self._value['board']
        # update head position in last so that if head is same as tail, updation
        # is still correct
        new_board[new_head.row, new_head.col] = self._value['head']
        self._board.appendleft(new_board.copy())
