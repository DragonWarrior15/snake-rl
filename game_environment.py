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
    def __init__(self, board_size = 10, start_length = 5):
        '''
        Initialization function for the environment.
        '''
        self._value = {'snake':10, 'board':0, 'food':5}
        self._size = board_size
        self._board = np.zeros(self._size ** 2)
        self._snake_length = start_length
        self._snake_tail = Position(self._size//2, 0)
        self._snake_head = Position(self._size//2, 0 + self._snake_length)

        # randomly initialize the location of food in the empty cells
        food_x = np.random.choice([o] + [], 1)[0]
        self._food = Position(x, y)

        # modify the board values for the snake, assumed to be lying horizontally initially
        for i in range(self._snake_length):
            self._board[5, i] = self._value['snake']
        # modify the food position on the board
        self._board[self._food.row, self._food.col] = self._value['food']
