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
        self._buffer = deque(maxlen = buffer_size)
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
        buffer_size = len(self._buffer)
        size = min(size, buffer_size)
        p = (1.0*size)/buffer_size
        # sample size will be smaller than buffer size, hence traverse queue once
        sample_data = []
        for x in self._buffer:
            if(np.random.random() < p):
                sample_data.append(x)
        np.random.shuffle(sample_data)
        s, a, r, done = [], [], [], []
        for x in sample_data:
            s.append(x[0])
            a.append(x[1])
            r.append(x[2])
            done.append(x[3])
        s = np.array(s)
        a = np.concatenate(a, axis=0)
        r = np.array(r).reshape(-1, 1)
        done = np.array(done).reshape(-1, 1)

        return s, a, r, done
