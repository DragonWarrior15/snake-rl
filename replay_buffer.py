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

    def get_current_size(self):
        ''' get current buffer size '''
        return len(self._buffer)

    def sample(self, size=1000, replace=False, shuffle=False):
        '''
        sample data from buffer and return in easily ingestible form
        Returns:
            s (nd array) : the state matrix for input
            a (int) : list of actions taken
            r (int) : list of rewards
            next_s (nd array) : the next state matrix for input
            done (int) : if the game was completed
        '''
        buffer_size = len(self._buffer)
        size = min(size, buffer_size)
        sample_data_idx = set(np.random.choice(range(buffer_size), \
                                    size=size, replace=replace))
        # sample size will be <= buffer size, hence traverse queue once
        sample_data = [val for index, val in enumerate(self._buffer) if index in sample_data_idx]
        if(shuffle):
            np.random.shuffle(sample_data)
        s, a, r, next_s, done = [], [], [], [], []
        for x in sample_data:
            s.append(x[0])
            a.append(x[1])
            r.append(x[2])
            next_s.append(x[3])
            done.append(x[4])
        s = np.array(s)
        a = np.concatenate(a, axis=0)
        r = np.array(r).reshape(-1, 1)
        next_s = np.array(next_s)
        done = np.array(done).reshape(-1, 1)

        return s, a, r, next_s, done
