'''
This script stores the game environment. Note that the snake is a part of the
environment itself in this implementation.
The environment state is a set of frames, we want the agent to be able to discern
the movement of the snake as well for which multiple frames are needed.
We will keep track of a history of 4 frames.
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
    def __init__(self, board_size=11, frames=4, start_length=4, seed=42):
        '''
        Initialization function for the environment.
        '''
        self._value = {'snake':255, 'board':0, 'food':128, 'head':180, 'border':80}
        self._actions = {0:'none', 1:'left', -1:'right'}
        self._board_size = board_size
        self._n_frames = frames
        self._reward = {'out':-10, 'food':10, 'time':1}
        # start length is constrained to be less than half of board size
        self._start_length = min(start_length, (board_size-2)//2)
        # set numpy seed for reproducible results
        np.random.seed(seed)
        # time limit to contain length of game
        self._max_time_limit = 50

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

    def reset(self):
        '''
        reset the environment
        Returns:
            board : the current board state
        '''
        board = self._value['board'] * np.ones((self._board_size, self._board_size))
        # make board borders
        board[:, 0] = self._value['border']
        board[:, self._board_size-1] = self._value['border']
        board[0, :] = self._value['border']
        board[self._board_size-1, :] = self._value['border']
        # initialize snake
        self._snake = deque()
        self._snake_length = self._start_length
        # modify the board values for the snake, assumed to be lying horizontally initially
        for i in range(1, self._snake_length+1):
            board[5, i] = self._value['snake']
            self._snake.append(Position(5, i))
        # modify the snakek head position
        head = self._get_snake_head()
        board[head.row, head.col] = self._value['head']
        # queue, left most entry is the latest frame
        self._board = deque(maxlen = self._n_frames)
        for i in range(self._n_frames):
            if(i == 0):
                self._board.append(board.copy())
            else:
                self._board.append((self._value['board'] * np.ones_like(board)).copy())

        # modify the food position on the board, after board queue initialized
        self._get_food()
        self._snake_direction = 0
        # set time elapsed to 0
        self._time = 0
        return self._queue_to_board()

    def _get_snake_head(self):
        '''
        get the head of the snake, right most element in the queue
        Returns:
            head : Position of the head
        '''
        return self._snake[-1]

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
            food_x, food_y = list(range(1,self._board_size-1)), list(range(1,self._board_size-1))
            food_x = np.random.choice(food_x, 1)[0]
            for i in range(1,self._board_size-1):
                if(self._board[0][food_x, i] != self._value['board']):
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

    def _get_new_direction(self, action, current_direction):
        '''
        get the new direction after taking the specified action
        Returns:
            direction (int) : the new direction of motion
        '''
        direction = (current_direction + action)%4
        return direction

    def _get_new_head(self, action, current_direction):
        '''
        get the position for the new head through the action
        Returns:
            new_head (Position) : position class for the new head
        '''
        new_dir  = self._get_new_direction(action, current_direction)
        del_x, del_y = (new_dir%2)*(new_dir-2), (1-(new_dir%2))*(1-new_dir)
        snake_head = self._get_snake_head()
        new_head = Position(snake_head.row + del_x,
                            snake_head.col + del_y)
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
        assert action in list(self._actions.keys()), "Action must be in {}".format(self._actions)
        reward, done = 0, 0

        # check if the current action is feasible
        reward, done, can_eat_food = self._check_if_done(action)
        if(done == 0):
            # if not done, move the snake
            self._move_snake(action, can_eat_food)
            # update the direction of motion
            self._snake_direction = self._get_new_direction(action, self._snake_direction)
            # get the next food location
            if(can_eat_food):
                self._get_food()

        # info contains time elapsed
        info = {'time':self._time}

        # update time
        self._time += 1

        return self._queue_to_board(), reward, done, info

    def _check_if_done(self, action):
        '''
        checks if the game has ended or if food has been taken
        Returns:
            reward : reward for the current action
            done : 1 if ended else 0
            can_eat_food : whether the current action leads to eating food
        '''
        reward, done, can_eat_food = self._reward['time'], 0, 0
        # check if the current action forces snake out of board
        new_head = self._get_new_head(action, self._snake_direction)
        while(1):
            # snake is colliding with border/obstacles
            try:
                if(self._board[0][new_head.row, new_head.col] == self._value['border']):
                    done = 1
                    reward = self._reward['out']
                    break
            except IndexError:
                # for debug
                print(self._board[0][:,:])
                print(self._board[1][:,:])
                print(new_head.row, new_head.col)
            # collision with self, collision with tail is allowed
            snake_tail = self._get_snake_tail()
            if(self._board[0][new_head.row, new_head.col] == self._value['snake']
               and not(new_head.row == snake_tail.row and new_head.col == snake_tail.col)):
                done = 1
                reward = self._reward['out']
                break
            # check if food
            if(self._board[0][new_head.row, new_head.col] == self._value['food']):
                done = 0
                reward += self._reward['food']
                can_eat_food = 1
                break
            # check if time is up
            if(self._time >= self._max_time_limit):
                done = 1
                break
            # if normal movement, no other updates needed
            break
        return reward, done, can_eat_food

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
        temp_neck = self._snake.pop()
        new_board[temp_neck.row, temp_neck.col] = self._value['snake']
        self._snake.append(temp_neck)
        # insert the new head into the snake queue
        # different treatment for addition of food
        # update the new board view as well
        # if new head overlaps with the tail, special handling is needed
        self._snake.append(new_head)

        if(can_eat_food):
            self._snake_length += 1
        else:
            delete_pos = self._snake.popleft()
            new_board[delete_pos.row, delete_pos.col] = self._value['board']
        # update head position in last so that if head is same as tail, updation
        # is still correct
        new_board[new_head.row, new_head.col] = self._value['head']
        self._board.appendleft(new_board.copy())
