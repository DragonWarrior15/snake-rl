"""This module stores the game environment. Note that the snake is a part of
the environment itself in this implementation.
The environment state is a set of frames, we want the agent to be able to discern
the movement of the snake as well, for which multiple frames are needed.
We will keep track of a history of 2 frames.
Important to manually reset the environment by user after initialization.
The board borders are different from board color
"""

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle

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

    Usage:
    env = SnakeNumpy(board_size=10, frames=2, games=10, start_length=2, 
                    seed=42, max_time_limit=-1)
    s = env.reset()
    done = 0
    while(not done):
        legal_moves = env.get_legal_moves()
        next_s, r, done, info, next_legal_moves = \
            agent.move(s, legal_moves, env.get_values())
        s = next_s.copy()

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
    _obstacles : bool
        Whether to use obstacles based board
    _version : str
        String representing the model version to pick obstacles files

    board : numpy array containing information about various objects in the
            board, including snake, food and obstacles
        
    """
    def __init__(self, board_size=10, frames=2, start_length=5, seed=42,
                 max_time_limit=298, obstacles=False, version=''):
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
        obstacles : bool, optional
            Whether to use obstacles boards
        version : str, optional
            Model version from which to pickup obstacle boards, should be given
            if obstacles is set to True
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
        # obstacles
        self._obstacles = obstacles
        self._version = version
        # other variables that can be quickly reused across multiple games
        # self._get_static_board_template()

    def _get_static_board_template(self):
        """Creates the static board template. By default a single border
        board is created, otherwise obstacles are also present
        """
        if(not self._obstacles):
            # make board borders
            self._static_board_template = self._value['board'] * np.ones((self._board_size, self._board_size))
            self._static_board_template[:, 0] = self._value['border']
            self._static_board_template[:, self._board_size-1] = self._value['border']
            self._static_board_template[0, :] = self._value['border']
            self._static_board_template[self._board_size-1, :] = self._value['border']
        else:
            # read obstacles boards from file and randomly select one
            with open('models/{:s}/obstacles_board'.format(self._version), 'rb') as f:
                self._static_board_template = pickle.load(f)

            self._static_board_template = self._static_board_template[\
                        np.random.choice(self._static_board_template.shape[0], 1), :, :]
            self._static_board_template = self._static_board_template.reshape((self._board_size, -1))
            self._static_board_template *= self._value['border']

    def reset(self):
        """Resets the environment to the starting state. Snake is kept same
        but food is randomly initialzed. Board and snake queues are initialized
        here.
        
        Returns
        -------
        board : Numpy array
            Starting game state
        """
        self._get_static_board_template()
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
        """Get the new direction after taking the specified action
        In case the action is directly opposite to the current direction,
        the same direction is maintained

        Returns
        -------
        direction : int
            the new direction of motion
        """
        # direction = (current_direction + self._action_map(action))%4
        if(self._action_map(action) == -1):
            return current_direction
        elif(abs(self._action_map(action) - current_direction) == 2):
            return current_direction
        else:
            return self._action_map(action)

    def _get_new_head(self, action, current_direction):
        """Get the position for the new head through the action
        Calculation is done using relative movement along the rows
        and columns of the board

        Returns
        -------
        new_head : Position
            position class for the new head
        """
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
        """Takes an action and performs one time step of the game, returns updated
        board along with reward, whether game is terminated, and additional info
        in the form of a dictionary (like time, snake length)

        Arguments
        ---------
        action : int
            should be among the possible actions
        Returns
        -------
        board : Numpy Array
            updated after taking the step
        reward : Integer (can be float if negative reward for time)
            agent's reward for performing the current action
        done : int
            takes the value 1 if game is terminated, otherwise 0)
        info : dict
            any auxillary game information (time, food, termination reason)
        """
        # assert action in list(range(self._n_actions)), "Action must be in " + list(range(self._n_actions))
        # assert action in self._actions, "Action must be in " + [k for k in self._actions]
        reward, done = 0, 0

        # for compatibility with all the agents return types
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

        next_legal_moves = self.get_legal_moves().copy()

        return self._queue_to_board(), reward, done, info, next_legal_moves

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
        """Calculate and return the reward for eating food
        try different rewards schemes for when food is eaten
        
        Returns
        -------
        reward : int (float if allowed in init function)
            the reward value, can be static or dependent on snake length
        """
        # return((self._snake_length - self._start_length + 1) * self._rewards['food'])
        return self._rewards['food']

    def _get_death_reward(self):
        """Calculate and return the reward for termination
        try different rewards schemes for termination
        
        Returns
        -------
        reward : int (float if allowed in init function)
            the reward value, can be static or dependent on snake length
        """
        # return((self._snake_length - self._start_length + 1) * self._rewards['out'])
        return self._rewards['out']

    def _check_if_done(self, action):
        """Checks if the game has ended or if food has been taken
        the checks are done by first calculating the new position of the head
        and then sequentially performing the following checks
        1) if no position is available for food 
            (game has ended without negative reward)
        2) collision check with borders
            board[new head] must equal a border value
        3) collision with self (except tail)
            board[new head] equals snake value, but not
            the tail (since tail will be displaced by every motion of snake)
        4) check for food
            board[new head] equals food value, then fodd can be eaten
        5) check for time up
            if time > max time limit (except when max time limit is -1)
            separate reward is also specified if game terminates here
            without the snake eating anything (probably going around in loops)
        no updates are made in any other case (which is normal snake movement)
        
        Returns
        -------
        reward : int (float if defined in init dict)
            reward for the current action
        done : int
            1 if game terminates else 0
        can_eat_food : int
            1 if the current action leads to eating food else 0
        termination_reason : string
            the reason for termination of game, empty string otherwise
        """
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
        """Moves the snake using the given action and updates the board 
        in accordance with whether food can be eater or not
        Motion is simply performed by moving the tail of the snake (popping
        from deque) and putting it on the other end (append)
        In case food is eaten, a new element is added to the from of the snake
        (thereby increasing the snake length) and new food position is populated
        """
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
    """ Class for the snake game using Numpy arrays
    A single array representing the game state contains multiple games
    at different stages, which can be evaluated by the agent in a single pass
    through the network to predict the next moves. This coupled with numpy 
    optimized matrix calculations leads to faster training and allows more
    randomization of the environment.

    Note : the game is playable even if it has terminated. Although,
    any action provided will have no effect whatsoever on the env. Separate
    logic must be run to determine till when must the env run..
    done variable returned by the step function can be utilized for this.

    In frame mode, the environment does soft resets of those games
    which have been terminated, thus allowing continuous running of all the
    games. Otherwise, the games that are terminated are not modified and the
    environment only modifies those games that are not terminated yet

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
    move snake function calls the get food function in case food
    has been eaten by the snake
            v
    step function calls the soft reset function in case the environment
    is being run in the frame mode
            v
    next state, reward, done, info next legal moves variables are returned

    Usage:
    env = SnakeNumpy(board_size=10, frames=2, games=10, start_length=2, 
                    seed=42, max_time_limit=-1, frame_mode=True)
    s = env.reset()
    while(some condition):
        legal_moves = env.get_legal_moves()
        next_s, r, done, info, next_legal_moves = \
            agent.move(s, legal_moves, env.get_values())
        s = next_s.copy()
    
    Attributes
    ----------
    _value : dict
        contains numerical mapping for different parts/cells of a snake board
    _actions : dict
        contains mapping for different actions, actions like -1 etc defined
    _n_actions : int
        total count of distinct actions available in the environment
    _board_size : int
        size of one edge of the game board
    _n_frames : int
        total no of frames kept in the history (for board state)
    _n_games : int
        total no of games represented in the board state
    _rewards : Numpy Array
        array of size _n_games containing reward values for each of the game
        reward is just the current reward obtained after takin action
    _start_length : int
        the starting length of the snake, fixed at 2 for now
    _max_time_limit : int
        the maximum time for which any game is allowed to run
        -1 will indicate that the game can be run forever until termination
    _board : deque
        a deque containing the individual frames of the game that form the 
        history, this deque is directly concatenated and returned as a game state
    _action_conv : Numpy array
        contains the different convolutions that can be used for snake movement
    _termination_reason_dict : dict
        numerical mapping from integer to the termination reason code
    _frame_mode : bool
        whether to run the environment in frame mode or not
    _seq : Numpy Array
        contains _n_games random sequences (reshaped to board size * board size)
        to quickly take the max and find the position to place the next food
        all the _n_games sequences are different to introduce maximum
        stochasticity into the environment, allowing for variety of states
    _body_random : Numpy Array
        contains all possible positions of the snake, assuming it is lying
        somewhere on the board in a straight line, in any possible direction.
        The environment can pick any one of these orientations when initializing
        or doing a soft reset.
    _head_random : Numpy Array
        contains the all possible positions of the snake head, in accordance with
        the snake positions defined in _body_random. Both the _body_random and
        _head_random arrays are of same size, and when selectin a snake
        configuration, the same index must be chosen from both
    _direction_random : Numpy Array
        size is same as _body_random.shape[0] and contains the direction corresponding
        to the snake defined in _body_random. The starting index should be same in
        the three arrays _body_random, _head_random and _direction_random
    _border : Numpy Array
        the current borders (with obstacles) applied to games in play
    _border_random : Numpy Array
        an array containing all possible borders that can be currently used
        in the environment. This can either contain just static borders
        or contain obstacle boards read from a file
    _food : Numpy Array
        of size _n_games * board size * board size and thus contains the position
        of the food in each of the games. The position is flagged by 1 (all other
        values in the array are zero)
    _body : Numpy Array
        of shape _n_games * board size * board size, keeps track of the snake
        body in all the games. Snake body is represented as a sequence of
        integers, with tail always taking the value 1. This allows easy
        simulation of snake movement (refer _move_snake function)
        eg-
        [ 0 0 0 0 0 0] 
        [ 0 0 0 0 0 0] 
        [ 0 1 0 0 0 0] 
        [ 0 2 0 0 0 0] 
        [ 0 3 0 0 0 0] 
        [ 0 0 0 0 0 0] 
    _head : Numpy Array
        of shape _n_games * _board_size * _board_size, keeps track of the snake
        head in all games, the position of head is indicated by 1
        eg-
        [ 0 0 0 0 0 0] 
        [ 0 0 0 0 0 0] 
        [ 0 0 0 0 0 0] 
        [ 0 0 0 0 0 0] 
        [ 0 0 1 0 0 0] 
        [ 0 0 0 0 0 0]
        Thus the complete snake configuration will be (direction is 0 or right)
        [ 0 0 0 0 0 0] 
        [ 0 0 0 0 0 0] 
        [ 0 1 0 0 0 0] 
        [ 0 2 0 0 0 0] 
        [ 0 3 h 0 0 0] 
        [ 0 0 0 0 0 0]        
    _snake_direction : Numpy Array
        of size _n_games, keeps track of current snake direction in all games 
    _snake_length : Numpy Array
        of size _n_games, keeps track of snake length in all games
    _count_food : Numpy Array
        of size _n_games, keeps track of no of times food eaten in all games
    _time : Numpy Array
        of size _n_games, keeps track of total time elapsed in all games
    _done : Numpy Array
        of size _n_games, keeps track of whether a game has terminated
    _cumul_rewards : Numpy Array
        of size _n_games, keeps track of total rewards accumulated in a game
    _obstacles : Bool
        whether to read obstacles board from file or generate static boards
    version : str
        model version string
    """
    def __init__(self, board_size=10, frames=2, games=10, start_length=2, seed=42,
                 max_time_limit=298, frame_mode=False, obstacles=False, version=''):
        """Initialization function for the environment. Not all the attributes
        are populated here. Some will be populated in the reset function so
        that the same environment can be used multiple times
        
        Parameters
        ----------
        board_size : int
            size of one edge of the board (including borders)
        frames : int
            total frames of game kept in history
        games : int
            total games represented at a time in an environment state
            can also be rephrased as total games being run in parallel
        start_length : int
            starting length of the snake including the head
        seed : int
            seed value for Numpy (not used in the script)
        max_time_limit : int
            maximum time to run any game for (-1 means forever)
        frame_mode : bool
            whether to run the environment in a frame mode or not
        """
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
        # define the convolutions for movement operations (total 4 actions)
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
        # whether frame mode or game mode, in former, environment
        # does a soft reset every time any board ends
        self._frame_mode = frame_mode
        # whether to have obstacles
        self._obstacles = obstacles
        self._version = version

    def _queue_to_board(self):
        """Convert the current queue of frames to a tensor by stacking all
        the frames together

        Returns
        -------
        board : Numpy Array
            of size num games * baord size * board size * frames
        """
        board = np.stack([x for x in self._board], axis=3)
        return board.copy()

    def _random_seq(self):
        """Creates num games boards of size board size * board size
        and fills them with sequence running from 1 to board size ** 2,
        shuffled in a random order for each game. After selecting the nos
        at those positions where the board does not have border/snake/head
        we can take the maximum of the numbers and place the food there.
        
        The function is called in the reset function. soft_reset function
        also calls this function, but with a very small probability to
        reduce the runtime needed to generate these sequences.
        """
        seq = np.arange(1,1+self._board_size**2, dtype=np.uint16)
        self._seq = np.zeros((self._n_games,self._board_size,self._board_size))
        for i in range(self._n_games):
            np.random.shuffle(seq)
            self._seq[i] = seq.copy().reshape((1,self._board_size,self._board_size))

    def _random_snake(self):
        """Generates all the possible templates for snake spawn
        The snake length is the start_length and the directions can be
        facing right, left, up and down. The position can be anywhere within
        the board, provided it is not colliding with the walls. It is possible
        that the adjacent cell to the snake is a wall.

        This code populates three variables, _body_random, _head_random and
        _snake_direction. The three variables together define a complete
        snake configuration. Whenevery a snake configuration is to be selected,
        same index in all three variables must be chosen.
        """

        # calculate total possible positions in which the snake can
        # be spawned
        strides = self._board_size - 2 - self._start_length + 1
        total_boards = strides * (self._board_size-2) * 4
        # create the array, uint16 chosen as snake length can be > 256
        self._body_random = np.zeros((total_boards,
                                      self._board_size, self._board_size), 
                                      dtype=np.uint16)

        self._head_random = self._body_random.copy()
        self._direction_random = np.zeros((total_boards,), dtype=np.uint8)
        # snake pointing towards right
        # with each i, snake moves one step right
        # the same is copied from top to bottom inside the loop
        for i in range(strides):
            # idx1 is the indexes in the template we are going to populate
            # this will keep changing as we move from one orientation to another
            idx1 = np.arange(0+i*(self._board_size-2),0+(i+1)*(self._board_size-2), dtype=np.uint8)
            # idx2 is all the rows we are going to populate
            # which is all the rows except the border
            idx2 = np.arange(1,self._board_size-1, dtype=np.uint8)
            self._body_random[idx1,idx2,i+1:i+1+self._start_length-1] = (np.arange(self._start_length-1, dtype=np.uint16)+1)
            # head is at the end after the snake sequence
            self._head_random[idx1,idx2,i+1+self._start_length-1] = 1
            # direction is already 0

        # mirror image (snake pointing towards left)
        # idx1 is the indexes in the template we are modifying
        idx1 = np.arange(total_boards//4, (total_boards//4)*2)
        # idx2 is the indexes in the template we are copying from
        idx2 = np.arange(total_boards//4)
        self._body_random[idx1,:,::-1] = self._body_random[idx2,:,:].copy()
        self._head_random[idx1,:,::-1] = self._head_random[idx2,:,:].copy()
        # direction is opposite from 2
        self._direction_random[idx1] = 2
        
        # snake pointing down
        # idx1 is the indexes in the template we are copying from
        # idx1 + (total_boards//4) is the indexes we are modifying
        idx1 = np.arange(total_boards//4, (total_boards//4)*2)
        for i in idx1:
            self._body_random[i+(total_boards//4),:,:] = self._body_random[i,::-1,:].copy().T
            self._head_random[i+(total_boards//4),:,:] = self._head_random[i,::-1,:].copy().T
        self._direction_random[idx1 + (total_boards//4)] = 3

        # snake pointing up (mirror image of above)
        # idx1 is the indexes in the template we are modifying
        idx1 = np.arange((total_boards//4)*3, (total_boards//4)*4)
        # idx2 is the indexes in the template we are copying from
        idx2 = np.arange((total_boards//4)*2, (total_boards//4)*3)
        self._body_random[idx1,::-1,:] = self._body_random[idx2,:,:].copy()
        self._head_random[idx1,::-1,:] = self._head_random[idx2,:,:].copy()
        self._direction_random[idx1] = 1

    def _random_board(self):
        """Generates the boards with static borders or reads
        obstacle boards from a file
        """
        if(not self._obstacles):
            # generate the boards ourselves
            self._border_random = self._value['board'] * np.ones((self._board_size-2,self._board_size-2), 
                                                          dtype=np.uint8)
            # make board borders
            self._border_random = np.pad(self._border_random, 1, mode='constant',
                                  constant_values=self._value['border'])\
                              .reshape(1,self._board_size,self._board_size)
            self._border_random = np.zeros((self._n_games, self._board_size, self._board_size)) \
                            + self._border_random
        else:
            with open('models/{:s}/obstacles_board'.format(self._version), 'rb') as f:
                self._border_random = pickle.load(f)
            self._border_random *= self._value['border']
            # self._border_random[1:,:,:] = self._border_random[0,:,:]

    def _calculate_board_wo_food(self):
        """Combines all elements together to get the board without food"""
        board = self._border + (self._body > 0)*self._value['snake'] + \
                self._head*self._value['head']
        return board.copy()

    def _calculate_board(self):
        """Combines all elements together to get the board which is the
        representation fed into the agent for training
        (body + head + borders) + food
        """
        board = self._calculate_board_wo_food() + self._food*self._value['food']
        return board.copy()

    def _weighted_sum(self, w, x1, x2):
        """Calculates weighted sum of vectors x1 and x2
        weighted by w
        Useful when need to modify only those boards
        that have not terminated

        Parameters
        ----------
        x1 : Numpy Array
            the first array to multiply with w
        x2 : Numpy Array
            the second array to multiply with 1-w

        Returns
        -------
        weighted_sum : Numpy Array
            the weighted sum
        """
        w = w.reshape(-1,1,1)
        return (w*x1 + (1-w)*x2).copy()

    def _set_first_frame(self):
        """Calculates board using current head, body food and borders
        to determine the lates frame of the _board deque (index 0)
        """
        board = self._calculate_board()
        self._board[0] = self._weighted_sum((1-self._done), board, self._board[0])

    def _reset_frames(self, f):
        """Reset old frames only for games where the game has terminated,
        since it is possible that they got modified during some calculation
        we modify all the frames from index 1 onwards to contain the same
        value as index 0
        
        Parameters
        ----------
        f : Numpy Array
            a filter/mask specifying which game indices to reset
        """

        board = self._calculate_board_wo_food()
        for i in range(1, len(self._board)):
            self._board[i][f] = board[f]

    def print_game(self):
        """Prints the current state (board)"""
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
        """Returns board_size
        
        Returns
        -------
        board_size : int
            the board size
        """
        return self._board_size

    def get_n_frames(self):
        """Returns total frames kept in history

        Returns
        -------
        total_frames : int
            count of frames kept in environment history
        """
        return self._n_frames

    def get_head_value(self):
        """Returns value of snake head in board representation

        Returns
        -------
        value : int
            the value of the snake head
        """
        return self._value['head']

    def get_values(self):
        """Returns the dictionary containing numeric values of
        different game components, see init for definition
        
        Returns
        -------
        _value : dict
            contains numeric mapping for all the game elements
        """
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
        """Resets the environment and initializes the relevant variables

        Parameters
        ----------
        stateful : bool
            whether to maintain the current environment state
            or reset everything to the start
        
        Returns:
            board : the current board state
        """
        # check whether to hard reset everything or not
        if(stateful and len(self._board)>0):
            return self._queue_to_board()
        # random generations
        # random number seq for food
        self._random_seq()
        # random boards for snake position (all horizontal)
        self._random_snake()

        # set the random boards (with/without obstacles)
        self._random_board()
        random_indices = np.random.choice(self._border_random.shape[0], self._n_games)
        self._border = self._border_random[random_indices].copy()

        # initialize snake
        self._food = np.zeros((self._n_games, self._board_size, self._board_size), dtype=np.uint8)
        if(not self._obstacles):
            random_indices = np.random.choice(self._body_random.shape[0], self._n_games)
        else:
            # remove those snake positions that overlap with the obstacles
            # individually for each game
            random_indices = np.zeros((self._n_games,), dtype=np.int16)
            for i in range(self._n_games):
                random_indices_mask = ((self._body_random + self._head_random) * self._border[i])\
                                        .sum(axis=(1,2)) == 0
                # convert to probabilities for the random choice function
                random_indices_mask = random_indices_mask/random_indices_mask.sum()
                random_indices[i] = int(np.random.choice(np.arange(self._body_random.shape[0]), 
                                                  1, p=random_indices_mask))
        # random_indices = np.ones((self._n_games), dtype=np.uint8) * ((self._board_size-2)//2)
        self._body, self._head, self._snake_direction = \
                                self._body_random[random_indices].copy(),\
                                self._head_random[random_indices].copy(),\
                                self._direction_random[random_indices].copy()

        # uint16 since snake length can be > 255
        self._snake_length = self._start_length * np.ones((self._n_games), dtype=np.uint16)
        self._count_food = np.zeros((self._n_games), dtype=np.uint16)
        # first view of the board
        board = self._calculate_board()
        # initialize the queue
        for _ in range(self._n_frames):
            self._board.append(board.copy())
        
        # modify the food position on the board, after board queue initialized
        self._get_food()
        
        # set time elapsed, done and cumulative rewards to 0
        self._time = np.zeros((self._n_games), dtype=np.uint16)
        self._done = np.zeros((self._n_games,), dtype=np.uint8)
        self._cumul_rewards = np.zeros((self._n_games,), dtype=np.int16)
        # set first frame
        self._set_first_frame()
        return self._queue_to_board()

    def _soft_reset(self):
        """Function to do soft reset of the game which is used when playing
        in frame mode, will reset all the boards where the game
        has ended and then initialize them to start again
        This is useful when training Q Learning algorithm, since we do not
        need the game to terminate to collect training samples
        """
        f = (self._done == 1)
        fsum = self._done.sum()
        # reset food where terminated
        self._food[f] = np.zeros((fsum, self._board_size,self._board_size),
                                 dtype=np.uint8)

        random_indices = np.random.choice(np.arange(self._border_random.shape[0]), fsum)
        self._border[f] = self._border_random[random_indices].copy()

        # initialize snake
        if(not self._obstacles):
            random_indices = np.random.choice(np.arange(self._body_random.shape[0]), fsum)
        else:
            # remove those snake positions that overlap with the obstacles
            # individually for each game
            random_indices = np.zeros((fsum,), dtype=np.int16)
            i = 0
            for i1 in range(self._done.shape[0]):
                if(self._done[i1] == 1):
                    random_indices_mask = ((self._body_random + self._head_random) * self._border[i1])\
                                            .sum(axis=(1,2)) == 0
                    # convert to probabilities for the random choice function
                    random_indices_mask = random_indices_mask/random_indices_mask.sum()
                    random_indices[i] = int(np.random.choice(np.arange(self._body_random.shape[0]), 
                                                  1, p=random_indices_mask))
                    i += 1

        # reset body head and direction where terminated
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
        # copy the first frame (without food) to all the remaining
        # frames in history
        self._reset_frames(f)

        # reshuffle all the sequences for random food generation
        # keep the probability small so that the function is not
        # called frequently
        if(np.random.random() < 0.01):
            self._random_seq()

    def get_num_actions(self):
        """Returns get total count of actions available in environment

        Returns
        -------
        _n_actions : int
            total actions available in environment
        """
        return self._n_actions

    def _action_map(self, action):
        """Converts integer actions using internal  mapping

        Parameters
        ----------
        action : int
            the action to take

        Returns
        -------
        action : int
            mapped value
        """
        return self._actions[action]

    def _get_food(self):
        """Find the coordinates of the point to put the food at one of the
        places which are occupied by the board
        """
        # food_pos = (self._board[0] == self._value['board']) * self._seq
        # collect positions where food can be placed (which is not border or body or head)
        food_pos = ((self._border + self._body + self._head) == self._value['board']) * self._seq
        # get the position where to place food, which is max of random nos from seq
        # axis 0 corresponds to _n_games, we need to calculate the maximum value
        # for each game individually
        m = food_pos.max((1,2)).reshape(self._n_games,1,1)
        # the second condition below is used in place no position is available 
        # to place food since the max will be 0 above, which is the value 
        # everywhere
        food_pos = ((food_pos == m) & (food_pos > self._value['board']))
        # if _food is already populated, do not populate again
        self._food = self._weighted_sum(1-self._food.max((1,2)), food_pos, self._food).astype(np.uint8)

    def _get_new_direction(self, action, current_direction):
        """Get the new direction after taking the specified action.
        New direction is same as action, unless action is directly
        opposite to current direction in which case current direction
        is maintained

        Parameters
        ----------
        action : Numpy Array
            contains the list of actions for all of the games
        current_direction : Numpy Array
            contains the current direction of snake in all the games
        
        Returns
        -------
        direction : int
            the new snake direction in all the games
        """
        new_dir = current_direction.copy()
        f = (np.abs(action - current_direction) != 2) & (action != -1)
        new_dir[f] = action[f]
        return new_dir.copy()

    def _get_new_head(self, action, current_direction):
        """Gets the position for the new head through the action
        first do convolution operations for all actions, then use
        one hot encoded actions for each game to get the final position of 
        the new head

        how does convolution provide the new head position
        [0 0 0 0]   [0 1 0]    [0 0]    reapply    [0 0 0 0]
        [0 1 0 0] * [0 0 0] -> [1 0] -> padding -> [0 0 0 0]
        [0 0 0 0]   [0 0 0]                        [0 1 0 0]
        [0 0 0 0]                                  [0 0 0 0]

        movement operation
        1) expand the given head array to create sub arrays that can
           be directly multiplied to perform convolutions (consider 2 games)
           to determine the strides argument in stride_tricks, first determine
           the output shape, and then each value in stride, corresponds to how
           many bytes need to be moved in order to get from one element to
           the next element along that axis
           For the following, output shape is (2, 2, 2, 3, 3)
           * we need to move 16 bytes/elements to get from first game to next
             first row of (2,) array to second
           * we need to move a total of 4 bytes/elements in order to get from
             one row to next row of sub 2x2 array
           * we need to move a total of 4 bytes/elements in order to get from
             one column to next column of a sub (,2,2,,) array
           * we need to move a total of 4 bytes/elements in order to get from 
             one row to next row of a sub (,,3,3) array
           * we need to move a total of 1 bytes/elements in order to get from 
             one column to next column of a sub (,,3,3) array

        [[[0 0 0 0]                    | [[0 0 0]  [0 0 0]
          [0 1 0 0]                    |  [0 1 0]  [1 0 0]
          [0 0 0 0]   strides for      |  [0 0 0]  [0 0 0]
          [0 0 0 0]], 3x3 convolutions | 
                      ---------------->|  [0 1 0]  [1 0 0]
          [0 0 0 0]                    |  [0 0 0]  [0 0 0]
          [0 0 0 0]                    |  [0 0 0]  [0 0 0],
          [0 0 1 0]                    | 
          [0 0 0 0]]]                  |  [0 0 0]  [0 0 0]
                                       |  [0 0 0]  [0 0 0]
                                       |  [0 0 1]  [0 1 0]
                                       | 
                                       |  [0 0 0]  [0 0 0]
                                       |  [0 0 1]  [0 1 0]
                                       |  [0 0 0]  [0 0 0]]

        2) now, we do a convolution operation with all possible actions
           in one go (2,2,2,3,3) * (3,3,4) -> (2,2,2,4)
           and multiply this with the action mask (2,4) to get output of
           shape (2,2,2,4) which has all zeros except the action we want to
           take. Hence, summing along 3rd axis gives us the required new
           position of the head with the corresponding action in each game.

        3) since we have reduced the dimension using convolution operation,
           we use a new zeros array and fill it with the new head posision
           to return the output array to the original head shape 
        
        Parameters
        ----------
        action : Numpy Array
            the actions to take for all the games
        current_direction : Numpy Array
            the current direction for all the games

        Returns:
        new_head : Numpy Array
            new head position for all the games
        """
        # new direction will determine where the snake will point next
        # this is to account for all possible moves, valid or invalid
        action = self._get_new_direction(action, current_direction)
        # convert action integers to one hot vectors
        one_hot_action = np.zeros((self._n_games,1,1,self._n_actions), dtype=np.uint8)
        one_hot_action[np.arange(self._n_games), :, :, action] = 1
        # calculate strides tuple (no of bytes along each axis that need to 
        # be travelled to reach the next element on that axis)
        hstr = self._head.strides
        # expand the head array in order to perform convolution operation
        new_head = np.lib.stride_tricks.as_strided(self._head, 
                       shape=(self._n_games,self._board_size-3+1,self._board_size-3+1,3,3),
                       strides=(hstr[0],hstr[1],hstr[2],hstr[1],hstr[2]),
                       writeable=False)
                       # strides determine how much steps are needed to reach the next element
                       # in that direction, to decide strides for the function, visualize
                       # with the expected output
        # where conv is (3,3,4) and sum along last axis
        new_head = (np.tensordot(new_head,self._action_conv) * one_hot_action).sum(3)
        # since we have reduced the dimensions, we first initialize an empty array
        # and then fill it up with the new calculated head, this is similar to padding
        # but faster
        new_head1 = np.zeros(self._head.shape, dtype=np.uint8)
        new_head1[:,1:self._board_size-1,1:self._board_size-1] = new_head
        return new_head1.copy()

    def step(self, action):
        """Takes an action and performs one time step of the game, returns updated
        board

        Parameters
        ----------
        action : Numpy Array 
            list of actions for all of the games
        
        Returns
        -------
        board : Numpy Array
            updated board after taking the action
        reward : Numpy Array
            Reward received for all of the games
        done : Numpy Array
            whether the game is over or not (1 or 0, for all games)
        info : dict
            any auxillary game information (time elapsed, food eaten, 
            termination_reason, current length, cumul_rewards, 
            all Numpy Arrays of size _n_games)
        next_legal_moves : Numpy Array
            mask containing the legal moves for the updated board
        """
        # assert action in list(range(self._n_actions)), "Action must be in " + list(range(self._n_actions))
        # assert action in self._actions, "Action must be in " + [k for k in self._actions]
        # check if the current action is feasible and if food can be eaten
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
        # whether to do a soft reset or not depending on frame mode
        if(self._frame_mode):
            self._soft_reset()
        # get legal moves for updated board
        next_legal_moves = self.get_legal_moves()
        return self._queue_to_board(), reward.copy(), done_copy.copy(),\
                info, next_legal_moves.copy()

    def _get_food_reward(self, f):
        """Calculate and return the reward for eating food
        try different rewards schemes for when food is eaten
        
        Parameters
        ----------
        f : numpy Array
            mask containing games for which calculation needs to be done

        Returns
        -------
        reward : int (float if allowed in init function)
            the reward value, can be static or dependent on snake length
        """
        # return((self._snake_length[f] - self._start_length + 1) * self._rewards['food'])
        return self._rewards['food']

    def _get_death_reward(self, f):
        """Calculate and return the reward for termination
        try different rewards schemes for termination

        Parameters
        ----------
        f : numpy Array
            mask containing games for which calculation needs to be done
        
        Returns
        -------
        reward : int (float if allowed in init function)
            the reward value, can be static or dependent on snake length
        """
        # return (self._snake_length[f] - self._start_length+1)*self._rewards['out']
        return self._rewards['out']

    def _check_if_done(self, action):
        """Checks if the game has ended or if food has been taken for all the 
        games. The checks are done by first calculating the new position of the head
        and then sequentially performing the following checks
        1) if no position is available for food 
            (game has ended without negative reward) possible
            when board size **2 equals snake length for any game
        2) collision check with borders
            occurs when new head is not populated with any 1
            (since if head bumping into wall, conv operation will return
            all zeros due to size reduction by the conv operation)
        3) collision with self (except tail)
            possible when max(new head * body) > 0 (since they are overlappping
            in that case) but collision with tail is allowed (new head * body 
            sum will be 1 in that case (since tail is 1 always))
            all these calculations are for any game
        4) check for food
            max(new head * food) for any game == 1
        5) check for time up
            if time > max time limit (except when max time limit is -1)
            separate reward is also specified if game terminates here
            without the snake eating anything (probably going around in loops)
        no updates are made in any other case (which is normal snake movement)

        Parameters
        ----------
        action : Numpy Array
            actions to take in each of the games

        Returns
        -------
        reward : Numpy Array
            reward for the current action in all the games
        done : Numpy Array
            1 if ended else 0 for all the games
        can_eat_food : Numpy Array
            whether the current action leads to eating food in all games
        """
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
        # snake is colliding with border/obstacles, conv returns board-2 size matrix
        # hence in case of collision with borders, the whole matrix will be 0
        # otherwise new_head and _border will overlap
        f2 = ((new_head.sum((1,2)) == 0) | \
                ((new_head * self._border).sum((1,2)) > 0))
        f = f2 & ~f1
        self._done[f] = 1
        reward[f] = self._get_death_reward(f)
        termination_reason[f] = 2
        #####################################
        # collision with self, collision with tail is allowed
        # the tail is defined to be equal to 1 in reset function
        body_head_sum = (self._body * new_head).sum((1,2))
        f3 = (body_head_sum > 0) & ~(body_head_sum == 1)
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
        """Moves the snake using the given action
        and updates the board accordingly
        
        Parameters
        ----------
        action : Numpy Array
            the actions to take in all of the games
        can_eat_food : Numpy Array
            1 if food is eaten with given action, otherwise 0 (populated
            for all of the games)
        new_head : Numpy Array
            new head array for all of the games
        """
        # update snake
        new_body = self._body.copy()
        # max value of body for all of the games
        body_max = self._body.max((1,2))
        # for movement, reduce all values except 0 by 1
        # by this operation, snake has effectively moved 1 step
        # except the block just after head (neck) which needs to be calculated
        new_body[self._body>0] -= 1
        # the three arrays added together are as follows
        # 1) done * current body (since we only update games which have not ended
        #    (soft reset done in the step function))
        # 2) if food can be eaten and game not ended, the new "neck" is simply
        #    current position of head * max(body)+1 since snake length has increased
        # 3) if fodd cannot be eaten and game not ended, the new "neck" is simply
        #    current position of head * max(body) since snake length is same
        self._body = (self._done).reshape(-1,1,1)*self._body + \
                     ((1-self._done)*can_eat_food).reshape(-1,1,1)*(self._body+(body_max+1).reshape(-1,1,1)*self._head) +\
                     ((1-self._done)*(1-can_eat_food)).reshape(-1,1,1)*(new_body+body_max.reshape(-1,1,1)*self._head)
        # update head where game not ended
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
