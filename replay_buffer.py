import numpy as np
from collections import deque

class ReplayBuffer:
    '''
    this class stores the replay buffer from which data can be sampled for
    training the model for q learning
    Attributes:
        buffer size (int) : maximum data to store in buffer
    '''
    def __init__(self, buffer_size=1000, board_size=6, frames=2, actions=4):
        ''' initialize the buffer with given size '''
        self._buffer = deque(maxlen = buffer_size)
        self._buffer_size = buffer_size
        self._n_actions = actions

    def add_to_buffer(self, s, a, r, next_s, done):
        ''' update the buffer by adding data '''
        self._buffer.append([s, a, r, next_s, done])

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
        s, a1, r, next_s, done = [], [], [], [], []
        for x in sample_data:
            s.append(x[0])
            a1.append(x[1])
            r.append(x[2])
            next_s.append(x[3])
            done.append(x[4])
        s = np.array(s)
        a1 = np.array(a1).reshape(-1)
        a = np.zeros((a1.shape[0], self._n_actions), dtype=np.uint8)
        a[np.arange(a1.shape[0]), a1] = 1
        r = np.array(r).reshape(-1, 1)
        next_s = np.array(next_s)
        done = np.array(done).reshape(-1, 1)

        return s, a, r, next_s, done

class ReplayBuffer2:
    '''
    this class stores the replay buffer from which data can be sampled for
    training the model for q learning
    Attributes:
        buffer size (int) : maximum data to store in buffer
    '''
    def __init__(self, buffer_size=1000, board_size=6, frames=2, actions=4):
        ''' initialize the buffer with given size '''
        self._s = np.zeros((buffer_size, board_size, board_size, frames), dtype=np.uint8)
        self._next_s = self._s.copy()
        self._a = np.zeros((buffer_size,), dtype=np.uint8)
        self._done = self._a.copy()
        self._r = np.zeros((buffer_size,), dtype=np.int16)
        self._buffer_size = buffer_size
        self._current_buffer_size = 0
        self._pos = 0
        self._n_actions = actions

    def add_to_buffer(self, s, a, r, next_s, done):
        ''' update the buffer by adding data '''
        if(s.ndim == 3):
            # single board is supplied
            l = 1
        else:
            l = s.shape[0]
        idx = np.arange(self._pos, self._pos+l)%self._buffer_size
        self._s[idx] = s
        self._a[idx] = a
        self._r[idx] = r
        self._next_s[idx] = next_s
        self._done[idx] = done
        self._pos = (self._pos+l)%self._buffer_size
        self._current_buffer_size = max(self._current_buffer_size, self._pos)

    def get_current_size(self):
        ''' get current buffer size, redundant '''
        # return len(self._buffer)
        return self._buffer_size

    def sample(self, size=1000, replace=False):
        '''
        sample data from buffer and return in easily ingestible form
        since we use np.random.choice, this is by default shuffled
        Returns:
            s (nd array) : the state matrix for input
            a (int) : list of actions taken
            r (int) : list of rewards
            next_s (nd array) : the next state matrix for input
            done (int) : if the game was completed
        '''
        size = min(size, self._current_buffer_size)
        idx = np.random.choice(np.arange(self._current_buffer_size), \
                                    size=size, replace=replace)

        s = self._s[idx]
        a = np.zeros((idx.shape[0],self._n_actions))
        a[np.arange(idx.shape[0]),self._a[idx]] = 1
        r = self._r[idx].reshape((-1, 1))
        next_s = self._next_s[idx]
        done = self._done[idx].reshape(-1, 1)

        return s, a, r, next_s, done
