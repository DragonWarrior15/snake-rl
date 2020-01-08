"""
This module stores the game environment. Note that the snake is a part of the
environment itself in this implementation.
The environment state is a set of frames, we want the agent to be able to discern
the movement of the snake as well, for which multiple frames are needed.
We will keep track of a history of 2 frames.
Important to manually reset the environment by user after initialization.
The board borders are different from board color
"""

import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class Position:
    """Class for defining any position on a 2D grid
    
    Attributes
    ----------
        row : int
            The row number for a 2D grid
        col : int
            The column for a 2D grid
    """
    def __init__(self, row=0, col=0):
        """Initalizer for the Position class, sets default values as zero

        Parameters
        ----------
        row : int, optional
            Row value to set (default 0)
        col : int, optional
            Column value to set (default 0)
        """
        self.row = row
        self.col = col

    def set_position(self, row=None, col=None):
        """Modify the existing position coordinate with given values
        update ignored if None

        Parameters
        ----------
        row : int
            Row value to set
        col : int
            Column value to set
        """
        if(row is not None):
            self.row = row
        if(col is not None):
            self.col = col

class Snake:
    """Class for the snake game. Call the reset function to get the first
    state of the environment after initialization.

    Note : the game is playable even if it has terminated. Although,
    any action provided will have no effect whatsoever on the env. Separate
    logic must be returned to run the env only till variable done stays 0.
    done is returned by the step function.
    
    Code flow
    init with parameters
            v
    reset (same env can be reset multiple times)
            v
    action input by agent/user to the step function
            v
    step function checks if the game is terminated due to action
    or can continue as is or if food has been eaten, and accordingly
    sets the values of done and reward variables
            v
    step function calls the move snake function which does the actual
    movement and updated both the snake and board queues
            v
    step function calls the get food function in case current food
    has been eaten by the snake
            v
    next state, reward, done and info variables are returned 

    Attributes
    ----------
        _value : dict
            Dictionary containing color values for different parts of board
        _actions : dict
            Dictionary containing mapping from user action to board action
        _n_actions : int
            Count of actions available in the environment, should be same
            for both the environment and the agent that plays
        _board_size : int
            Length of one side of the square board
        _n_frames : int
            Number of frames kept in any state, helps with movement information
        _rewards : dict
            Values for rewards for different events
        _start_length : int
            The length of the snake when the game starts
        _max_time_limit : int
            The maximum time to run the game for, -1 indicates forever
        _static_board_template : Numpy array
            Contains all zeros except borders, set according to _value['border']
        _snake : Deque
            Deque containing the individual positions of the snake body.
            For movement, last value is simply popped and appended to the left
            of the queue. When increasing length, new position is appended 
            to the left of the queue
        _snake_length : int
            Keeps track of the length of the snake, updated when food is eaten
        _snake_head : Position
            Keeps track of the head of the snake (row and col)
        _board : Deque
            Keeps track of individual frames in a game state. During update,
            new frame is appended to the left. Queue length is always fixed
        _snake_direction : int
            Keeps track of in which direction is the snake moving. This is
            necessary to correctly update the snake position given an action
        _time : int
            Keeps track of time elapsed (in steps) since game started

        board : numpy array containing information about various objects in the
                board, including snake, food and obstacles
        
    """
    def __init__(self, board_size=10, frames=2, start_length=5, seed=42,
                 max_time_limit=298):
        """Initializer for the snake class. Some of the attributes are
        initialized here while the remaining are done in the reset function
        depending on which need to be refreshed every time game restarts

        Parameters
        ----------
        board_size : int, optional
            The board size of the environment (env is square)
        frames : int, optional
            Total historic "images" kept in the state
        start_length : int, optional
            The starting length of the snake (constrained below to be less
            than half the size of board)
        seed : int, optional
            Seed value to set (Not used here for randomness)
        max_time_limit : int, optional
            Maximum steps for the env to run (-1 indicates no bound)
        """
        
        # self._value = {'snake':255, 'board':0, 'food':128, 'head':180, 'border':80}
        self._value = {'snake':1, 'board':0, 'food':3, 'head':2, 'border':4}
        # self._actions = [-1, 0, 1] # -1 left, 0 nothing, 1 right
        self._actions = {-1:-1, 0:0, 1:1, 2:2, 3:3, 4:-1}
        self._n_actions = 4
        self._board_size = board_size
        self._n_frames = frames
        self._rewards = {'out':-1, 'food':1, 'time':0, 'no_food':0}
        # start length is constrained to be less than half of board size
        # self._start_length = min(start_length, (board_size-2)//2)
        self._start_length = 2
        # set numpy seed for reproducible results
        # np.random.seed(seed)
        # time limit to contain length of game, -1 means run till end
        self._max_time_limit = max_time_limit
        # other variables that can be quickly reused across multiple games
        self._get_static_board_template()

    def _get_static_board_template(self):
        """Creates the static board template. By default a single border
        board is created, otherwise obstacles are also present
        """
        # make board borders
        self._static_board_template = self._value['board'] * np.ones((self._board_size, self._board_size))
        self._static_board_template[:, 0] = self._value['border']
        self._static_board_template[:, self._board_size-1] = self._value['border']
        self._static_board_template[0, :] = self._value['border']
        self._static_board_template[self._board_size-1, :] = self._value['border']

        if(0):
            # add obstacles to the board where no border
            for i in range(1):
                pos_board = np.random.random(self._static_board_template.shape) * \
                        (self._static_board_template == self._value['board'])
                self._static_board_template[pos_board == pos_board.max()] = \
                                        self._value['border']

    def reset(self):
        """Resets the environment to the starting state. Snake is kept same
        but food is randomly initialzed. Board and snake queues are initialized
        here.
        
        Returns
        -------
        board : Numpy array
            Starting game state
        """
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

    def _queue_to_board(self):
        """Convert the current queue of frames to a 3D matrix
        of shape board size * board size * frame count

        Returns
        -------
        board : Numpy array
            Current environment state
        """        
        board = np.dstack([x for x in self._board])
        return board.copy()

    def _get_food(self):
        """Find the coordinates of the point to put the food at
        first randomly locate a row to put the food in, then remove all
        the cells with snake, head and borders to choose among the 
        remaining
        """
        # create a random ordering for row
        ord_x = list(range(1,self._board_size-1))
        np.random.shuffle(ord_x)
        found = False
        '''
        iterate over rows in the shuffled order
        and search for available y positions
        if no y position is available, move to the next row
        if no x is valid, food position is not set and game must
        have terminated
        '''
        for x in ord_x:
            food_y = [i for i in range(1, self._board_size-1) \
                        if self._board[0][x, i] == self._value['board']]
            if(len(food_y) == 0):
                continue
            else:
                food_y = np.random.choice(food_y)
                self._food = Position(x, food_y)
                self._put_food()
                found = True
                break

    def print_game(self):
        """Prints the current state (board) as a plot"""
        fig, axs = plt.subplots(1, self._n_frames)
        if(self._n_frames == 1):
            axs.imshow(self._board[0], cmap = 'gray')
        else:
            for i in range(self._n_frames):
                axs[i].imshow(self._board[i], cmap = 'gray')
        plt.show()

    def get_board_size(self):
        """Gets board size"""
        return self._board_size

    def get_n_frames(self):
        """Gets frame count"""
        return self._n_frames

    def get_values(self):
        """Gets dictionary containing values for different
        board parts (snake, head, food, border)
        """
        return self._value

    def get_num_actions(self):
        """Gets total count of actions in environment"""
        return self._n_actions

    def _action_map(self, action):
        """Converts action integer to internal action value using 
        action mapping

        Returns
        -------
        action : int
            Action converted to environment action representation
        """
        return self._actions[action]

    def _get_snake_tail(self):
        """Gets the head of the snake which is the left most element 
        in the queue
        
        Returns
        -------
        head : Position
            (row, col) value for the snake head
        """
        return self._snake[0]

    def _put_food(self):
        """Put food in the required spot in the left most (latest) frame"""
        self._board[0][self._food.row, self._food.col] = self._value['food']

    def _get_new_direction(self, action, current_direction):
        '''
        get the new direction after taking the specified action
        Returns:
            direction (int) : the new direction of motion
        '''
        # direction = (current_direction + self._action_map(action))%4
        if(self._action_map(action) == -1):
            return current_direction
        elif(abs(self._action_map(action) - current_direction) == 2):
            return current_direction
        else:
            return self._action_map(action)

    def _get_new_head(self, action, current_direction):
        '''
        get the position for the new head through the action
        Returns:
            new_head (Position) : position class for the new head
        '''
        new_dir  = self._get_new_direction(action, current_direction)
        # del_x, del_y = (new_dir%2)*(new_dir-2), (1-(new_dir%2))*(1-new_dir)
        if(new_dir == 0):
            del_x, del_y = 1, 0
        elif(new_dir == 1):
            del_x, del_y = 0, 1
        elif(new_dir == 2):
            del_x, del_y = -1, 0
        else:
            del_x, del_y = 0, -1
        new_head = Position(self._snake_head.row - del_y,
                            self._snake_head.col + del_x)
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
        # assert action in list(range(self._n_actions)), "Action must be in " + list(range(self._n_actions))
        # assert action in self._actions, "Action must be in " + [k for k in self._actions]
        reward, done = 0, 0

        if isinstance(action, np.ndarray):
            action = int(action[0])

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

    def get_legal_moves(self):
        """Get legal moves for the current board state using
        the current snake direction (all moves except moving in the opposite
        direction are valid)

        Returns
        -------
        valid_moves : Numpy array
            valid moves mask for all games
        """    
        a = np.ones((1, self._n_actions), dtype=np.uint8)
        a[0, (self._snake_direction-2)%4] = 0
        return a.copy()

    def _get_food_reward(self):
        ''' try different rewards schemes for when food is eaten '''
        # return((self._snake_length - self._start_length + 1) * self._rewards['food'])
        return self._rewards['food']

    def _get_death_reward(self):
        ''' try different rewards schemes for death '''
        # return((self._snake_length - self._start_length + 1) * self._rewards['out'])
        return self._rewards['out']

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
            # check if no position available for food
            if((self._board[0] == self._value['board']).sum() == 0 and \
               (self._board[0] == self._value['food']).sum() == 0):
                done = 1
                reward += self._get_food_reward()
                termination_reason = 'game_end'
                break
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
            if(self._time >= self._max_time_limit and self._max_time_limit != -1):
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

class SnakeNumpy:
    '''
    Class for the snake game.
    
    Attributes
    ----------
        _total_rewards : Numpy array
            Keeps track of total rewards accumulated till current timestamp
            resets whenever a new game is started
    '''
    def __init__(self, board_size=10, frames=2, games=10, start_length=2, seed=42,
                 max_time_limit=298, frame_mode=False):
        '''
        Initialization function for the environment.
        '''
        # self._value = {'snake':255, 'board':0, 'food':128, 'head':180, 'border':80}
        self._value = {'snake':1, 'board':0, 'food':3, 'head':2, 'border':4}
        # self._actions = [-1, 0, 1] # -1 left, 0 nothing, 1 right
        self._actions = {-1:-1, 0:0, 1:1, 2:2, 3:3, 4:-1}
        self._n_actions = 4 # 0, 1, 2, 3
        self._board_size = board_size
        self._n_frames = frames
        self._n_games = games
        self._rewards = {'out':-1, 'food':1, 'time':0, 'no_food':0}
        # start length is constrained to be less than half of board size
        # self._start_length = min(start_length, (board_size-2)//2)
        self._start_length = 2 # fix for random positioning
        # set numpy seed for reproducible results
        # np.random.seed(seed)
        # time limit to contain length of game, -1 means run till end
        self._max_time_limit = max_time_limit
        # queue for board
        self._board = deque(maxlen = self._n_frames)
        # define the convolutions for movement operations
        self._action_conv = np.zeros((3,3,self._n_actions), dtype=np.uint8)
        self._action_conv[1,0,0] = 1
        self._action_conv[2,1,1] = 1
        self._action_conv[1,2,2] = 1
        self._action_conv[0,1,3] = 1
        # terminaiton reason dict
        self._termination_reason_dict = {
            'game_end'        : 1,
            'collision_wall'  : 2,
            'collision_self'  : 3,
            'time_up'         : 4,
            'time_up_no_food' : 5
        }
        # whether frame mode or game mode, in former, game
        # does a soft reset every time any board ends
        self._frame_mode = frame_mode

    def _queue_to_board(self):
        '''
        Convert the current queue of frames to a tensor
        Returns:
            board : np array of 4 dimensions
        '''
        board = np.stack([x for x in self._board], axis=3)
        return board.copy()

    def _random_seq(self):
        ''' shuffle the positions for food '''
        seq = np.arange(1,1+self._board_size**2, dtype=np.uint16)
        self._seq = np.zeros((self._n_games,self._board_size,self._board_size))
        for i in range(self._n_games):
            np.random.shuffle(seq)
            self._seq[i] = seq.copy().reshape((1,self._board_size,self._board_size))

    def _random_snake(self):
        ''' templates for snake spawn '''
        strides = self._board_size - 2 - self._start_length + 1
        total_boards = strides * (self._board_size-2) * 4
        self._body_random = np.zeros((total_boards,
                                      self._board_size, self._board_size), 
                                      dtype=np.uint16)
        self._head_random = self._body_random.copy()
        self._direction_random = np.zeros((total_boards,), dtype=np.uint8)
        # snake pointing towards right)
        for i in range(strides):
            idx1 = np.arange(0+i*(self._board_size-2),0+(i+1)*(self._board_size-2), dtype=np.uint8)
            idx2 = np.arange(1,self._board_size-1, dtype=np.uint8)
            self._body_random[idx1,idx2,i+1:i+1+self._start_length-1] = (np.arange(self._start_length-1, dtype=np.uint16)+1)
            self._head_random[idx1,idx2,i+1+self._start_length-1] = 1

        # mirror image (snake pointing towards left)
        idx1 = np.arange(total_boards//4, (total_boards//4)*2)
        idx2 = np.arange(total_boards//4)
        self._body_random[idx1,:,::-1] = self._body_random[idx2,:,:].copy()
        self._head_random[idx1,:,::-1] = self._head_random[idx2,:,:].copy()
        self._direction_random[idx1] = 2
        
        # snake pointing down
        idx1 = np.arange(total_boards//4, (total_boards//4)*2)
        for i in idx1:
            self._body_random[i+(total_boards//4),:,:] = self._body_random[i,::-1,:].copy().T
            self._head_random[i+(total_boards//4),:,:] = self._head_random[i,::-1,:].copy().T
        self._direction_random[idx1 + (total_boards//4)] = 3

        # snake pointing up (mirror image of above)
        idx1 = np.arange((total_boards//4)*3, (total_boards//4)*4)
        idx2 = np.arange((total_boards//4)*2, (total_boards//4)*3)
        self._body_random[idx1,::-1,:] = self._body_random[idx2,:,:].copy()
        self._head_random[idx1,::-1,:] = self._head_random[idx2,:,:].copy()
        self._direction_random[idx1] = 1

    def _static_board(self):
        """Generates the static borders"""
        self._border = self._value['board'] * np.ones((self._board_size-2,self._board_size-2), 
                                                      dtype=np.uint8)
        # make board borders
        self._border = np.pad(self._border, 1, mode='constant',
                              constant_values=self._value['border'])\
                          .reshape(1,self._board_size,self._board_size)
        self._border = np.zeros((self._n_games, self._board_size, self._board_size)) \
                        + self._border

    def _calculate_board(self):
        ''' combine all elements together to get the board '''
        board = self._border + (self._body > 0)*self._value['snake'] + \
                self._head*self._value['head'] + self._food*self._value['food']
        return board.copy()

    def _calculate_board_wo_food(self):
        ''' combine all elements together to get the board without food '''
        board = self._border + (self._body > 0)*self._value['snake'] + \
                self._head*self._value['head']
        return board.copy()

    def _weighted_sum(self, w, x1, x2):
        w = w.reshape(-1,1,1)
        return (w*x1 + (1-w)*x2).copy()

    def _set_first_frame(self):
        board = self._calculate_board()
        self._board[0] = self._weighted_sum((1-self._done), board, self._board[0])

    def _reset_frames(self, f):
        """ reset old frames only for games where f (done) is applicable """
        board = self._calculate_board_wo_food()
        for i in range(1, len(self._board)):
            self._board[i][f] = board[f]

    def print_game(self):
        ''' prints the current state (board) '''
        board = self._queue_to_board()
        fig, axs = plt.subplots(self._n_games, self._n_frames)
        if(self._n_games == 1 and self._n_frames == 1):
            axs.imshow(board[0], cmap='gray')
        elif(self._n_games == 1):
            for i in range(self._n_frames):
                axs[i].imshow(board[0,:,:,i], cmap='gray')
        elif(self._n_frames == 1):
            for i in range(self._n_games):
                axs[i].imshow(board[i,:,:,0], cmap='gray')
        else:
            for i in range(self._n_games):
                for j in range(self._n_frames):
                    axs[i][j].imshow(board[i,:,:,j], cmap = 'gray')
        plt.show()

    def get_board_size(self):
        ''' get board_size '''
        return self._board_size

    def get_n_frames(self):
        ''' get frame count '''
        return self._n_frames

    def get_head_value(self):
        ''' get color of head '''
        return self._value['head']

    def get_values(self):
        return self._value

    def get_legal_moves(self):
        """Get legal moves for the current board state using
        the current snake direction (all moves except moving in the opposite
        direction are valid)

        Returns
        -------
        valid_moves : Numpy array
            valid moves mask for all games
        """    
        a = np.ones((self._n_games, self._n_actions), dtype=np.uint8)
        a[np.arange(self._n_games), (self._snake_direction-2)%4] = 0
        return a.copy()

    def reset(self, stateful=False):
        '''
        reset the environment
        Returns:
            board : the current board state
        '''
        if(stateful and len(self._board)>0):
            return self._queue_to_board()
        # random generations
        # random number seq for food
        self._random_seq()
        # random boards for snake position (all horizontal)
        self._random_snake()

        # set the random boards (with/without obstacles)
        self._static_board()
        
        # initialize snake, head takes the value 1 always
        self._food = np.zeros((self._n_games, self._board_size, self._board_size), dtype=np.uint8)
        random_indices = np.random.choice(np.arange(self._body_random.shape[0]), self._n_games)
        # random_indices = np.ones((self._n_games), dtype=np.uint8) * ((self._board_size-2)//2)
        self._body, self._head, self._snake_direction = \
                                self._body_random[random_indices].copy(),\
                                self._head_random[random_indices].copy(),\
                                self._direction_random[random_indices].copy()

        self._snake_length = self._start_length * np.ones((self._n_games), dtype=np.uint16)
        self._count_food = np.zeros((self._n_games), dtype=np.uint16)
        # first view of the board
        board = self._calculate_board()
        # initialize the queue
        for _ in range(self._n_frames):
            self._board.append(board.copy())
        # modify the food position on the board, after board queue initialized
        self._get_food()
        # set time elapsed to 0
        self._time = np.zeros((self._n_games), dtype=np.uint16)
        self._done = np.zeros((self._n_games,), dtype=np.uint8)
        self._cumul_rewards = np.zeros((self._n_games,), dtype=np.int16)
        # set first frame
        self._set_first_frame()
        return self._queue_to_board()

    def _soft_reset(self):
        '''
        function to do soft reset of the game, works when playing
        in frame mode, will reset all the boards where the game
        has ended and then initialize them
        '''
        f = (self._done == 1)
        fsum = self._done.sum()
        self._food[f] = np.zeros((fsum, self._board_size,self._board_size),
                                 dtype=np.uint8)
        random_indices = np.random.choice(np.arange(self._body_random.shape[0]), fsum)
        # random_indices = np.ones((fsum), dtype=np.uint8) * ((self._board_size-2)//2)
        self._body[f], self._head[f], self._snake_direction[f] = \
                        self._body_random[random_indices].copy(),\
                        self._head_random[random_indices].copy(),\
                        self._direction_random[random_indices].copy()

        # assign the body
        self._snake_length[f] = self._start_length
        self._time[f] = 0
        self._done[f] = 0
        self._cumul_rewards[f] = 0
        self._get_food()
        self._set_first_frame()
        self._reset_frames(f)

        if(np.random.random() < 0.01):
            self._random_seq()

    def get_num_actions(self):
        ''' get total count of actions '''
        return self._n_actions

    def _action_map(self, action):
        ''' converts integer to internal action mapping '''
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
        places which are occupied by the board
        '''
        # food_pos = (self._board[0] == self._value['board']) * self._seq
        food_pos = ((self._border + self._body + self._head) == self._value['board']) * self._seq
        # get the position where to place food, which is max of random nos from seq
        m = food_pos.max((1,2)).reshape(self._n_games,1,1)
        # the second condition is used in place no position is available to place food
        # since the max will be 0 above, which is the value everywhere
        food_pos = ((food_pos == m) & (food_pos > self._value['board']))
        # if _food is already populated, do not populate again
        self._food = self._weighted_sum(1-self._food.max((1,2)), food_pos, self._food).astype(np.uint8)

    def _get_new_direction(self, action, current_direction):
        '''
        get the new direction after taking the specified action
        Returns:
            direction (int) : the new direction of motion
        '''
        new_dir = current_direction.copy()
        f = (np.abs(action - current_direction) != 2) & (action != -1)
        new_dir[f] = action[f]
        return new_dir.copy()

    def _get_new_head(self, action, current_direction):
        '''
        get the position for the new head through the action
        first do convolution operations for all actions, then use
        one hot encoded actions for each game to get the final position of the new head
        Returns:
            new_head (Position) : position class for the new head
        '''
        action = self._get_new_direction(action, current_direction)
        one_hot_action = np.zeros((self._n_games,1,1,self._n_actions), dtype=np.uint8)
        one_hot_action[np.arange(self._n_games), :, :, action] = 1
        hstr = self._head.strides
        new_head = np.lib.stride_tricks.as_strided(self._head, 
                       shape=(self._n_games,self._board_size-3+1,self._board_size-3+1,3,3),
                       strides=(hstr[0],hstr[1],hstr[2],hstr[1],hstr[2]),
                       writeable=False)
                       # strides determine how much steps are needed to reach the next element
                       # in that direction, to decide strides for the function, visualize
                       # with the expected output
        # where conv is (3,3,4) and sum along last axis
        new_head = (np.tensordot(new_head,self._action_conv) * one_hot_action).sum(3)
        new_head1 = np.zeros((self._n_games,self._board_size,self._board_size), dtype=np.uint8)
        new_head1[:,1:self._board_size-1,1:self._board_size-1] = new_head
        return new_head1.copy()

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
        # assert action in list(range(self._n_actions)), "Action must be in " + list(range(self._n_actions))
        # assert action in self._actions, "Action must be in " + [k for k in self._actions]
        # check if the current action is feasible
        reward, can_eat_food, termination_reason, new_head \
                    = self._check_if_done(action)
        # if not done, move the snake
        self._move_snake(action, can_eat_food, new_head)
        # update the direction of motion
        self._snake_direction = self._get_new_direction(action, self._snake_direction)
        # update time
        self._time += (1-self._done)
        # update cumulative rewards, no need to check for done as that is
        # accounted for already in _check_if_done function
        self._cumul_rewards += reward
        # info contains time elapsed etc
        info = {'time':self._time.copy(), 'food':self._count_food.copy(),
                'termination_reason':termination_reason.copy(),
                'length':self._snake_length.copy(),
                'cumul_rewards':self._cumul_rewards.copy()}
        done_copy = self._done.copy()
        if(self._frame_mode):
            self._soft_reset()
        next_legal_moves = self.get_legal_moves()
        return self._queue_to_board(), reward.copy(), done_copy.copy(),\
                info, next_legal_moves.copy()

    def _get_food_reward(self, f):
        ''' try different rewards schemes for when food is eaten '''
        return((self._snake_length[f] - self._start_length + 1) * self._rewards['food'])
        # return self._rewards['food']

    def _get_death_reward(self, f):
        ''' try different rewards schemes for death '''
        return (self._snake_length[f] - self._start_length+1)*self._rewards['out']
        # return self._rewards['out']

    def _check_if_done(self, action):
        '''
        checks if the game has ended or if food has been taken
        Returns:
            reward : reward for the current action
            done : 1 if ended else 0
            can_eat_food : whether the current action leads to eating food
        '''
        reward, can_eat_food, termination_reason = \
                            self._rewards['time'] * np.ones((self._n_games,), dtype=np.int16),\
                            np.zeros((self._n_games,), dtype=np.uint8),\
                            np.zeros((self._n_games), dtype=np.uint8)
        done_copy = self._done.copy()
        # get the new head
        #####################################
        new_head = self._get_new_head(action, self._snake_direction)
        # check if no position available for food
        f1 = (self._snake_length == (self._board_size-2)**2)
        self._done[f1] = 1
        reward[f1] += self._get_food_reward(f1)
        termination_reason[f1] = 1
        #####################################
        # snake is colliding with border/obstacles, conv returns board-2 size matrix, hence
        # following logic is not valid since head will be all zeros if running into wall
        # f2 = ((new_head + self._border) == (self._value['border'] + 1)).sum((1,2)).astype(np.bool)
        f2 = (new_head.sum((1,2)) == 0)
        f = f2 & ~f1
        self._done[f] = 1
        reward[f] = self._get_death_reward(f)
        termination_reason[f] = 2
        #####################################
        # collision with self, collision with tail is allowed
        # the tail is defined to be equal to 1 in reset function
        f3 = ((self._body * new_head).sum((1,2)) > 0) & \
            ~((new_head * self._body).sum((1,2)) == 1)
        f = f3 & ~f2 & ~f1
        self._done[f] = 1
        reward[f] = self._get_death_reward(f)
        termination_reason[f] = 3
        #####################################
        # check if food
        f4 = ((self._food * new_head).sum((1,2)) == 1)
        f = f4 & ~f3 & ~f2 & ~f1
        reward[f] += self._get_food_reward(f)
        # self._count_food += 1
        can_eat_food[f] = 1
        #####################################
        # check if time up
        if(self._max_time_limit != -1):
            f5 = (self._time >= self._max_time_limit)
            f = f5 & ~f4 & ~f3 & ~f2 & ~f1
            self._done[f] = 1
            termination_reason[f] = 4
            # check if no food eaten
            if(self._rewards['no_food'] != 0):
                f6 = (self._snake_length == self._start_length)
                f = f6 & ~f5 & ~f4 & ~f3 & ~f2 & ~f1
                termination_reason[f] = 5
                reward[f] += self._rewards['no_food']
        #####################################
        # if normal movement, no other updates needed
        # if game already ended in prev frame, set all rewards to zero
        reward[done_copy == 1] = 0

        return reward.copy(), can_eat_food.copy(), termination_reason.copy(), new_head.copy()

    def _move_snake(self, action, can_eat_food, new_head):
        '''
        moves the snake using the given action
        and updates the board accordingly
        '''
        # update snake
        new_body = self._body.copy()
        body_max = self._body.max((1,2))
        new_body[self._body>0] -= 1
        self._body = (self._done).reshape(-1,1,1)*self._body + \
                     ((1-self._done)*can_eat_food).reshape(-1,1,1)*(self._body+(body_max+1).reshape(-1,1,1)*self._head) +\
                     ((1-self._done)*(1-can_eat_food)).reshape(-1,1,1)*(new_body+body_max.reshape(-1,1,1)*self._head)
        # update head
        self._head = self._weighted_sum(self._done, self._head, new_head)
        # get the next food location
        if(can_eat_food.sum()>0):
            # update parameters
            self._snake_length[can_eat_food == 1] += 1
            self._count_food[can_eat_food == 1] += 1
            # adjust food position
            self._food = self._weighted_sum((1-can_eat_food), self._food, 0)
            self._get_food()
        # calculate new board and append
        self._board.appendleft(self._board[0].copy())
        self._set_first_frame()
