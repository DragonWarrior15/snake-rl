'''
store all the agents here
'''
from replay_buffer import ReplayBuffer
from keras.models import Model, Sequential
from keras.layers import (Input, Conv2D, Dense,
        Flatten, Concatenate, Multiply, Lambda)
from keras.optimizers import Adam, SGD
import keras.backend as K
import numpy as np

def huber_loss(y_true, y_pred, delta=1):
    ''' keras implementation for huber loss '''
    error = (y_true - y_pred)
    if(K.abs(error) < delta):
        # quadratic error
        return K.mean(K.square(error), axis=-1)
    else:
        # linear error
        return K.mean(delta*(K.abs(error) - 0.5*delta), axis=-1)

class QLearningAgent():
    '''
    this agent learns the game via q learning
    Attributes:
        board_size (int): side length of the board
        frames (int): no of frames available in one board state
        buffer_size (int): size of the replay buffer
        buffer (queue): replay buffer
        n_actions (int): no of actions available in the action space
        actions (list): list of possible actions that can be taken
        epsilon (float): for epsilon greedy policy
        gamma (float): for reward discounting
        use_target_net (bool): if to keep two networks for learning
        input_shape (tuple): shape of input tensor
    '''
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 epsilon=0.01, gamma = 0.9, n_actions=3, use_target_net=True):
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._actions = [-1, 0, 1]
        self._epsilon = epsilon
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        self.reset_buffer()
        self.reset_models()

    def reset_buffer(self, buffer_size=None):
        ''' reset current buffer '''
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBuffer(self._buffer_size)

    def reset_models(self):
        ''' reset all the current models '''
        self._model_train, self._model_pred = self._agent_models()
        if(self._use_target_net):
            _, self._target_net = self._agent_models()
            self.update_target_net()

    def _action_map(self, action):
        ''' convert integer output to -1, 0 or 1 '''
        assert action in [0, 1, 2], "error! network output size should be 3"
        return self._actions[action]

    def _inverse_action_map(self, action):
        ''' convert action (-1, 0, 1) to action index (network output) '''
        assert action in self._actions
        for index, i in enumerate(self._actions):
            if(i == action):
                return index

    def get_epsilon(self):
        return self._epsilon

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon

    def _get_qvalues(self, board, model=None):
        ''' get action value '''
        if model is None:
            model = self._model_pred
        if(board.ndim == 3):
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board.copy())
        q_values = model.predict(board)
        return q_values

    def _normalize_board(self, board):
        ''' normalize the board before input to the network '''
        return((board/128.0 - 1).copy())

    def move(self, board):
        ''' get the action using epsilon greedy policy '''
        if(np.random.random() <= self._epsilon):
            action = int(np.random.choice(list(range(self._n_actions)), 1)[0])
        else:
            q_values = self._get_qvalues(board.reshape((1,) + self._input_shape),
                                        self._model_pred)
            action = int(np.argmax(q_values))
        action = self._action_map(action)
        return action

    def _agent_models(self):
        '''
        returns the trainin and prediction models, training model will use action
        mask as well to get single output for a given state action combination
        prediction network gets values for all the possible actions
        '''
        input_board = Input((self._board_size, self._board_size, self._n_frames,))
        input_action = Input((self._n_actions,))
        # total rows + columns + diagonals is total units
        x = Conv2D(64, (3,3), activation = 'relu')(input_board)
        x = Conv2D(128, (3,3), activation = 'relu')(x)
        x = Flatten()(x)
        x = Dense(64, activation = 'relu')(x)
        x = Dense(self._n_actions, activation = 'linear', name = 'action_values')(x)
        x = Multiply()([input_action, x])
        out = Lambda(lambda x: K.sum(x, axis = 1), output_shape = (1,))(x)

        model_train = Model(inputs = [input_board, input_action], outputs = out)
        model_train.compile(optimizer = Adam(1e-4, clipvalue=1.0), loss = 'mean_squared_error')

        model_pred = Model(inputs = input_board,
                           outputs = model_train.get_layer('action_values').output)

        return model_train, model_pred

    def save_model(self, file_path = '', iteration = None):
        ''' save the current models '''
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model_pred.save("{}/model_{:04d}_prediction.h5".format(file_path, iteration))
        self._model_train.save("{}/model_{:04d}_train.h5".format(file_path, iteration))
        self._target_net.save("{}/model_{:04d}_target.h5".format(file_path, iteration))

    def load_model(self, file_path = '', iteration = None):
        ''' load any existing models, if available '''
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        try:
            self._model_pred  = load_model("{}/model_{:04d}_prediction.h5".format(file_path, iteration))
            self._model_train = load_model("{}/model_{:04d}_train.h5".format(file_path, iteration))
            self._target_net  = load_model("{}/model_{:04d}_target.h5".format(file_path, iteration))
        except FileNotFoundError:
            print("Couldn't locate models at {}, check provided path".format(file_path))

    def print_models(self):
        ''' print the current models '''
        print('Training Model')
        print(self._model_train.summary())
        print('Prediction Model')
        print(self._model_pred.summary())
        print('Target Network')
        print(self._target_net.summary())

    def add_to_buffer(self, board, action, reward, next_board, done):
        '''
        add current game step to the replay buffer
        also maps action back to one hot encoded version
        '''
        one_hot_action = np.zeros((1, len(self._actions)))
        one_hot_action[0, self._inverse_action_map(action)] = 1
        self._buffer.add_to_buffer([board, one_hot_action, reward, next_board, done])

    def train_agent(self, sample_size=1000, epochs=10, verbose=0):
        '''
        train the model by sampling from buffer and return the error
        Returns:
            error (float) : the current mse
        '''
        s, a, r, next_s, done = self._buffer.sample(sample_size)
        # calculate the discounted reward, and then train accordingly
        not_done = 1 - done
        current_model = self._target_net if self._use_target_net else self._model_pred
        discounted_reward = r + (self._gamma * np.max(self._get_qvalues(next_s, current_model), axis = 1).reshape(-1, 1)) * not_done
        # normalize and train the model
        s = self._normalize_board(s.copy())
        self._model_train.fit([s, a], discounted_reward, epochs=epochs, verbose=verbose)
        return self._model_train.evaluate([s, a], discounted_reward, verbose=verbose)

    def update_target_net(self):
        '''
        this network is static for a while and serves as "ground truth"
        target network outputs is what we try to predict
        '''
        if(self._use_target_net):
            self._target_net.set_weights(self._model_pred.get_weights())

    def copy_weights_from_agent(self, agent_for_copy):
        ''' to update weights between competing agents
        useful in parallel training '''
        assert isinstance(agent_for_copy, Agent), "Agent type is required for copy"

        self._model_train.set_weights(agent_for_copy._model_train.get_weights())
        self._target_net.set_weights(agent_for_copy._model_pred.get_weights())
