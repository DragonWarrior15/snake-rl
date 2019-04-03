'''
This script stores the game environment. Note that the snake is a part of the
environment itself in this implementation.
'''

import numpy as np

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
        snake_tail (int, int) : Position containing the row and column no of board
                                for snake tail
        food (int, int) : Position containing the coordinates for the food
    '''
    def __init__(self, board_size = 10, start_length = 5, seed = 42):
        '''
        Initialization function for the environment.
        '''
        self._value = {'snake':10, 'board':0, 'food':5}
        self._actions = {0:'none', 1:'left', 2:'right'}
        self._size = board_size
        # set numpy seed for reproducible results
        np.random.seed(seed)
        # initialize other parameters
        self._snake_length = start_length
        self._snake_tail = Position(self._size//2, 0)
        self._snake_head = Position(self._size//2, 0 + self._snake_length)

        # randomly initialize the location of food in the empty cells
        food_x, food_y = list(range(self._size)), list(range(self._size))
        food_x.remove(self._snake_head.row)
        food_x = np.random.choice(food_x, 1)[0]
        food_y = np.random.choice(food_y, 1)[0]
        self._food = Position(food_x, food_y)

        self._board = self.reset()


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

        # modify the board values for the snake, assumed to be lying horizontally initially
        for i in range(self._snake_length):
            board[5, i] = self._value['snake']
        # modify the food position on the board
        board[self._food.row, self._food.col] = self._value['food']

        return board.copy()

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
