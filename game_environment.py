'''
This script stores the game environment. Note that the snake is a part of the
environment itself in this implementation.
The environment state is a set of frames, we want the agent to be able to discern
the movement of the snake as well for which multiple frames are needed.
We will keep track of a history of 4 frames.
Important to manually reset the environment by user after initialization.
'''

import numpy as np
from collections import deque

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
    '''
    def __init__(self, board_size = 10, start_length = 5, seed = 42):
        '''
        Initialization function for the environment.
        '''
        self._value = {'snake':10, 'board':0, 'food':5}
        self._actions = {0:'none', 1:'left', 2:'right'}
        self._size = board_size
        self._n_frames = 4
        # set numpy seed for reproducible results
        np.random.seed(seed)
        # initialize other parameters
        self._snake_length = start_length

    def _queue_to_board(self):
        '''
        Convert the current queue of frames to a 3D matrix
        Returns:
            board : np array of 3 dimensions
        '''
        board = np.dstack([x for x in self._board])
        return board.copy()

    def _print_game():
        ''' prints the current state (board) '''
        pass

    def reset(self):
        '''
        reset the environment
        Returns:
            board : the current board state
        '''
        board = np.zeros(self._size ** 2)
        self._snake_head = Position(self._size//2, 0 + self._snake_length)
        # modify the board values for the snake, assumed to be lying horizontally initially
        for i in range(self._snake_length):
            board[5, i] = self._value['snake']
        # modify the food position on the board
        self._get_food()
        # queue, left most entry is the latest frame
        self._board = deque(self._n_frames)
        for i in range(self._n_frames):
            if(i == 0):
                self._board.appendleft(board.copy())
            else:
                self._board.appendleft(np.zeros_like(board))

        return self._queue_to_board()

    def _get_food(self):
        '''
        find the coordinates of the point to put the food at
        first randomly locate a row to put the food in, then remove all
        the cells with snake and choose amongst the remaining
        '''

        while(1):
            food_x, food_y = list(range(self._size)), list(range(self._size))
            food_x = np.random.choice(food_x, 1)[0]
            for i in range(self._size):
                if(self._board[0][food_x, i] == self._value['snake']):
                    food_y.remove(i)
            if(len(food_y) == 0):
                continue
            else:
                food_y = np.random.choice(food_y, 1)[0]
                break
        self._food = Position(food_x, food_y)
        self._put_food()

    def _put_food(self):
        ''' put the food in the required spot '''
        self._board[0][self._food.row, self._food.col] = self._value['food']

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
        assert action in list(self._actions.keys()), "Action must be in {}".format(self._actions)
        reward, done = 0, 0

        # check if the current action is feasible

        # for now, assume info is none
        info = None

        return self._board.copy(), reward, done, info

    def _check_if_done(self, action):
        '''
        checks if the game has ended or not
        Returns:
            done : 1 if ended else 0
        '''
        done = 0

        return done
