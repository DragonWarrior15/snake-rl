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
    def __init__(self, board_size=10, frames=2, games=10, start_length=5, seed=42,
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
        self._start_length = min(start_length, (board_size-2)//2)
        # set numpy seed for reproducible results
        # np.random.seed(seed)
        # time limit to contain length of game, -1 means run till end
        self._max_time_limit = max_time_limit
        # other variables that can be quickly reused across multiple games
        self._border = self._value['board'] * np.ones((self._board_size-2,self._board_size-2))
        # make board borders
        self._border = np.pad(self._border, 1, mode='constant',
                              constant_values=self._value['border'])\
                          .reshape(1,self._board_size,self._board_size)
        self._border = np.zeros((self._n_games, self._board_size, self._board_size)) \
                        + self._border
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

    def _calculate_board(self):
        ''' combine all elements together to get the board '''
        board = self._border + (self._body > 0)*self._value['snake'] + \
                self._head*self._value['head'] + self._food*self._value['food']
        return board.copy()

    def _weighted_sum(self, w, x1, x2):
        w = w.reshape(-1,1,1)
        return (w*x1 + (1-w)*x2).copy()

    def _set_first_frame(self):
        board = self._calculate_board()
        self._board[0] = self._weighted_sum((1-self._done), board, self._board[0])

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

    def reset(self):
        '''
        reset the environment
        Returns:
            board : the current board state
        '''
        # initialize snake, head takes the value 1 always
        self._body = np.zeros((self._n_games, self._board_size, self._board_size), dtype=np.uint16)
        self._food, self._head = self._body.copy().astype(np.uint8), self._body.copy().astype(np.uint8)
        self._snake_length = self._start_length * np.ones((self._n_games), dtype=np.uint16)
        self._count_food = np.zeros((self._n_games), dtype=np.uint16)
        # assume snake is just head + 1 body initially, place randomly across games
        self._body[:,self._board_size//2, 1:self._start_length] = \
            np.arange(1,self._start_length).reshape(1,1,-1)
        self._head[:, self._board_size//2, self._start_length] = 1
        self._snake_direction = np.zeros((self._n_games,), dtype=np.uint8)
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
        self._head[f] = np.zeros((fsum, self._board_size,self._board_size),
                                 dtype=np.uint8)
        self._body[f] = np.zeros((fsum, self._board_size,self._board_size),
                                 dtype=np.uint8)
        # assign the body
        self._body[f,self._board_size//2, 1:self._start_length] = \
            np.arange(1,self._start_length).reshape(1,1,-1)
        self._head[f, self._board_size//2, self._start_length] = 1
        self._snake_direction[f] = 0
        self._snake_length[f] = self._start_length
        self._time[f] = 0
        self._done[f] = 0
        self._set_first_frame()

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
        board = self._board[0]
        seq = np.arange(0,(self._n_games * (self._board_size**2)))
        np.random.shuffle(seq)
        food_pos = (board == self._value['board']) * seq.reshape(self._n_games,self._board_size,self._board_size)
        m = food_pos.max((1,2)).reshape(self._n_games,1,1)
        food_pos = ((food_pos == m) & (food_pos > self._value['board']))
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
                       strides=(hstr[0],hstr[1],hstr[2],hstr[1],hstr[2]))
                       # strides determine how much steps are needed to reach the next element
                       # in that direction, to decide strides for the function, visualize
                       # with the expected output
        new_head = np.tensordot(new_head, self._action_conv) # where conv is (3,3,4)
        new_head = np.pad((new_head * one_hot_action).sum(3),
                        1,mode='constant', constant_values=0)[1:-1] # sum along last axis
        return new_head.copy()

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
        # info contains time elapsed etc
        info = {'time':self._time, 'food':self._count_food,
                'termination_reason':termination_reason}
        done_copy = self._done.copy()
        if(self._frame_mode):
            self._soft_reset()
        return self._queue_to_board(), reward, done_copy, info

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
        reward, can_eat_food, termination_reason = \
                            self._rewards['time'] * np.ones((self._n_games,), dtype=np.int16),\
                            np.zeros((self._n_games,), dtype=np.uint8),\
                            np.zeros((self._n_games), dtype=np.uint8)
        # get the new head
        #####################################
        new_head = self._get_new_head(action, self._snake_direction)
        # check if no position available for food
        f1 = ((self._board[0] == self._value['board']).sum((1,2)) == 0) & \
               ((self._board[0] == self._value['food']).sum((1,2)) == 0)
        self._done[f1] = 1
        reward[f1] += self._get_food_reward()
        termination_reason[f1] = 1
        #####################################
        # snake is colliding with border/obstacles, conv returns board-2 size matrix, hence
        # following logic is not valid since head will be all zeros if running into wall
        # f2 = ((new_head + self._border) == (self._value['border'] + 1)).sum((1,2)).astype(np.bool)
        f2 = (new_head.sum((1,2)) == 0)
        f = f2 & ~f1
        self._done[f] = 1
        reward[f] = self._get_death_reward()
        termination_reason[f] = 2
        #####################################
        # collision with self, collision with tail is allowed
        f3 = ((self._body + new_head).min((1,2)) > 1) & \
            ~((new_head*self._body).max((1,2)) == self._body.max((1,2)))
        f = f3 & ~f2 & ~f1
        self._done[f] = 1
        reward[f] = self._get_death_reward()
        termination_reason[f] = 3
        #####################################
        # check if food
        f4 = ((self._food * new_head).sum((1,2)) == 1)
        f = f4 & ~f3 & ~f2 & ~f1
        reward[f] += self._get_food_reward()
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
        # print(f1, f2, f3, f4, f5)
        return reward, can_eat_food, termination_reason, new_head

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
