from collections import deque
import numpy as np

class ReplayBuffer:
    '''
    this class stores the replay buffer from which data can be sampled for
    training the model for q learning
    Attributes:
        buffer size (int) : maximum data to store in buffer
    '''
    def __init__(self, buffer_size = 10000):
        ''' initialize the buffer with given size '''
        self._buffer = deque(maxlen = 10000)
        self._buffer_size = buffer_size

    def add_to_buffer(self, data):
        ''' update the buffer by adding data '''
        self._buffer.append(data)

    def sample(self, size = 1000):
        '''
        sample data from buffer and return in easily ingestible form
        Returns:
            s (nd array) : the state matrix for input
            a (int) : list of actions taken
            r (int) : list of rewards
            done (int) : if the game was completed
        '''
        sample_indices = np.random.choice(list(range(len(self._buffer))), size)
        s, a, r, done = [], [], [], []
        # sample size will be smaller than buffer size, hence traverse queue once
        for i in sample_indices:
            d =
