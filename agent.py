'''
store all the agents here
'''
from replay_buffer import ReplayBuffer
from keras.models import Model, Sequential, load_model
from keras.layers import (Input, Conv2D, Dense,
        Flatten, Concatenate, Multiply, Lambda)
from keras.optimizers import Adam, SGD, RMSprop
import keras.backend as K
import numpy as np
import time

def huber_loss(y_true, y_pred, delta=1):
    ''' keras implementation for huber loss '''
    error = (y_true - y_pred)
    if(K.abs(error) < delta):
        # quadratic error
        return K.mean(K.square(error), axis=-1)
    else:
        # linear error
        return K.mean(delta*(K.abs(error) - 0.5*delta), axis=-1)

class DeepQLearningAgent():
    '''
    this agent learns the game via q learning
    model outputs everywhere refers to q values
    Attributes:
        board_size (int): side length of the board
        frames (int): no of frames available in one board state
        buffer_size (int): size of the replay buffer
        buffer (queue): replay buffer
        n_actions (int): no of actions available in the action space
        gamma (float): for reward discounting
        use_target_net (bool): if to keep two networks for learning
        input_shape (tuple): shape of input tensor
    '''
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma = 0.99, n_actions=3, use_target_net=True):
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        self.reset_buffer()
        self.reset_models()

    def get_gamma(self):
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        ''' reset current buffer '''
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBuffer(self._buffer_size)

    def reset_models(self):
        ''' reset all the current models '''
        self._model = self._agent_model()
        if(self._use_target_net):
            self._target_net = self._agent_model()
            self.update_target_net()

    def _get_model_outputs(self, board, model=None):
        ''' get action value '''
        if model is None:
            model = self._model
        if(board.ndim == 3):
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board.copy())
        model_outputs = model.predict_on_batch(board)
        return model_outputs

    def _normalize_board(self, board):
        ''' normalize the board before input to the network '''
        return board.copy()
        # return((board/128.0 - 1).copy())

    def move(self, board):
        ''' get the action using agent policy '''
        model_outputs = self._get_model_outputs(board, self._model)
        return int(np.argmax(model_outputs))

    def _agent_model(self):
        '''
        returns the model which evaluates q values for a given state input
        '''
        input_board = Input((self._board_size, self._board_size, self._n_frames,))
        # total rows + columns + diagonals is total units
        x = Conv2D(16, (4,4), activation = 'relu', data_format='channels_last')(input_board)
        x = Conv2D(32, (4,4), activation = 'relu', data_format='channels_last')(x)
        x = Flatten()(x)
        x = Dense(64, activation = 'relu')(x)
        out = Dense(self._n_actions, activation = 'linear', name = 'action_values')(x)

        model = Model(inputs = input_board, outputs = out)
        model.compile(optimizer = RMSprop(0.0005), loss = 'mean_squared_error')

        return model

    def save_model(self, file_path = '', iteration = None):
        ''' save the current models '''
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.save("{}/model_{:04d}.h5".format(file_path, iteration))
        if(self._use_target_net):
            self._target_net.save("{}/model_{:04d}_target.h5".format(file_path, iteration))

    def load_model(self, file_path = '', iteration = None):
        ''' load any existing models, if available '''
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        try:
            self._model = load_model("{}/model_{:04d}.h5".format(file_path, iteration))
            if(self._use_target_net):
                self._target_net  = load_model("{}/model_{:04d}_target.h5".format(file_path, iteration))
        except FileNotFoundError:
            print("Couldn't locate models at {}, check provided path".format(file_path))

    def print_models(self):
        ''' print the current models '''
        print('Training Model')
        print(self._model.summary())
        print('Target Network')
        print(self._target_net.summary())

    def add_to_buffer(self, board, action, reward, next_board, done):
        '''
        add current game step to the replay buffer
        also maps action back to one hot encoded version
        '''
        one_hot_action = np.zeros((1, self._n_actions))
        one_hot_action[0, action] = 1
        self._buffer.add_to_buffer([board, one_hot_action, reward, next_board, done])

    def train_agent(self, batch_size=32):
        '''
        train the model by sampling from buffer and return the error
        Returns:
            error (float) : the current mse
        '''
        s, a, r, next_s, done = self._buffer.sample(batch_size)
        # calculate the discounted reward, and then train accordingly
        current_model = self._target_net if self._use_target_net else self._model
        next_model_outputs = self._get_model_outputs(next_s, current_model)
        discounted_reward = r + (self._gamma * np.max(next_model_outputs, axis = 1).reshape(-1, 1)) * (1-done)
        # create the target variable, only the column with action has different value
        target = self._get_model_outputs(s)
        target = (1-a)*target + a*discounted_reward
        # fit
        loss = self._model.train_on_batch(self._normalize_board(s), target)
        return loss

    def update_target_net(self):
        '''
        this network is static for a while and serves as "ground truth"
        target network outputs is what we try to predict
        '''
        if(self._use_target_net):
            self._target_net.set_weights(self._model.get_weights())

    def copy_weights_from_agent(self, agent_for_copy):
        ''' to update weights between competing agents
        useful in parallel training '''
        assert isinstance(agent_for_copy, Agent), "Agent type is required for copy"

        self._model.set_weights(agent_for_copy._model.get_weights())
        self._target_net.set_weights(agent_for_copy._model_pred.get_weights())

class PolicyGradientAgent(DeepQLearningAgent):
    '''
    this agent is only different from dqn agent in terms of the algorithm
    for calculating the best actions in a state, we will use policy policy gradient
    for the same
    '''
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma = 0.99, n_actions=3, use_target_net=False):
        DeepQLearningAgent.__init__(self, board_size=board_size, frames=frames,
                                buffer_size=buffer_size, gamma=gamma,
                                n_actions=n_actions, use_target_net=False)
        self._update_function = self._policy_gradient_updates()

    def _agent_model(self):
        '''
        returns the model which evaluates probability values for given state input
        '''
        input_board = Input((self._board_size, self._board_size, self._n_frames,))
        # total rows + columns + diagonals is total units
        x = Conv2D(16, (4,4), activation = 'relu', data_format='channels_last')(input_board)
        x = Conv2D(32, (4,4), activation = 'relu', data_format='channels_last')(x)
        x = Flatten()(x)
        x = Dense(64, activation = 'relu')(x)
        out = Dense(self._n_actions, activation = 'linear', name = 'action_values')(x)

        model = Model(inputs = input_board, outputs = out)
        # do not compile the model here, but rather use the outputs separately
        # in a training function to create any custom loss function
        # model.compile(optimizer = RMSprop(0.0005), loss = 'mean_squared_error')

        return model

    def _policy_gradient_updates(self):
        ''' a custom function for policy gradient losses '''
        a = K.placeholder(name='a', shape=(None, self._n_actions))
        discounted_reward = K.placeholder(name='r', shape=(None, 1))
        # calculate policy
        policy = K.softmax(self._model.output)
        log_policy = K.log(policy)
        # calculate loss
        J = K.mean(K.sum(log_policy * a, axis=1) * discounted_reward)
        entropy = -K.sum(K.sum(policy * log_policy, axis=1))
        loss = -J - 0.001*entropy
        # fit
        optimizer = RMSprop(0.005)
        updates = optimizer.get_updates(loss, self._model.trainable_weights)
        model = K.function([self._model.input, a, discounted_reward], [loss], updates=updates)
        return model

    def train_agent(self, batch_size=32):
        '''
        train the model by sampling from buffer and return the error
        Returns:
            error (float) : the current loss
        '''
        # in policy gradient, only one complete episode is used for training
        s, a, discounted_reward, _, _ = self._buffer.sample(self._buffer.get_current_size())
        # unlike DQN, the discounted reward is not estimated but true one
        # we have defined custom policy graident loss function above
        # use that to train to agent model
        loss = self._update_function([self._normalize_board(s.copy()), a, discounted_reward])

        return loss[0]

    def get_action_proba(self, s):
        '''
        returns the action probability values as policy graident is on policy
        '''
        model_outputs = self._get_model_outputs(board, self._model)
        return K.softmax(model_outputs)
