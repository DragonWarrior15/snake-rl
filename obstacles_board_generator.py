"""Simple script to generate random boards with obstacles
such that the path formed by the cells available for movement
is connected. The script is separate from environment as checking
connectivity and running the generator again and again will consume
too much time. Also, mirror/rotated versions of the board are also added
to the final array to generate more possibilities and reduce time

Assumptions
-----------
the script returns possible obstacle locations marked by 1
the script assumes the board to already have a border which
will also be generated in this script

Parameters
----------
board_size : int
    the board size to generate
version : str
    the model version string, generated boards saved in that folder
total_boards : int
    total boards to generate using the script
set_seed : bool
    whether to reset the seed before generation
total_obstacles : int
    total obstacle cells to keep in the board
    a very high value may slow the script and not generate 
    enough possibilities
"""

import numpy as np
import pickle
import time
from tqdm import tqdm

board_size = 10
version = 'v17.1'
total_boards = 40
track_index = 1000
set_seed = True
total_obstacles = 8

def time_calculate(start, end):
    total_time = round(end_time - start_time, 3)
    hrs = int(total_time)//3600
    mins = int(total_time)//60 - (hrs * 60)
    secs = int(total_time) - (hrs * 3600) - (mins * 60)
    msecs = total_time - int(total_time)
    return (hrs, mins, secs, msecs)

# set seed
if(set_seed):
    np.random.seed(42)

# final master board, 12 added as safety margin for adding duplicates
obstacles_board = np.zeros((total_boards+12, board_size, board_size), dtype=np.uint8)

start_time = time.time()
# start generating the random boards
prev_index = 0
index = 0
while (index < total_boards):
    # print(index)
    if(index//track_index > prev_index//track_index):
        end_time = time.time()
        hrs, mins, secs, msecs = time_calculate(start_time, end_time)
        print('Generated a total of {:d} boards in {:3d}hrs {:2d}mins {:2d}secs {:.3f}ms.'\
              .format(index, hrs, mins, secs, msecs))
    prev_index = index

    # if trying to generate the connected board
    # more than some fixed number of times, break
    trial_counter = 0
    while(trial_counter < board_size**2):
        # generate a random sequence
        seq = np.arange(1, board_size**2 + 1, dtype=np.uint16)
        np.random.shuffle(seq)
        seq = seq.reshape((board_size, board_size))

        # put borders as necessary
        board = np.ones((board_size, board_size), dtype=np.uint8)
        board[:, [0, board_size-1]] = 0
        board[[0, board_size-1], :] = 0

        for _ in range(total_obstacles):
            # get available positions
            m = seq * board
            m1 = (m == m.max())
            # add the obstacle
            board[m1] = 0
            m[m1] = 0

        # check connectivity in the board
        """
        moving one step in horizontal/vertical directions
        should give a total of more than 2 zeros, since that will
        allow formation of a path
        """
        connected = True
        for i in range(0, board_size):
            for j in range(0, board_size):
                if(board[i, j] == 1):
                    adj_ones = 0
                    for del_x, del_y in [[-1,0], [1,0], [0,1], [0,-1]]:
                        if(board[i+del_x, j+del_y] == 1):
                            adj_ones += 1
                    if(adj_ones < 2):
                        connected = False
                if(not connected):
                    break
            if(not connected):
                break

        trial_counter += 1
        if(not connected):
            continue
        else:
            # check if board already generated earlier
            duplicate = False
            if(index > 0):
                if((obstacles_board[:index,:,:] == board).sum((1,2)).max() == board_size**2):
                    duplicate = True
            if(duplicate):
                continue
            else:
                break
    print(trial_counter, connected)
    if(trial_counter == board_size**2 and not connected):
        # maximum tries reached, do not go ahead
        print('Maximum tries reached at index : {:d}, breaking'.format(index))
        break
    else:
        # add the board and rotations/mirrors to master board
        board_list = []
        for i in range(4):
            # do rotations and take their horizontal and vertical mirrors
            for j in range(3):
                if(j == 0):
                    if(i == 0):
                        board_temp = board.copy()
                    else:
                        # rotation = transpose then mirror
                        board_temp = board.copy().T
                        board_temp[:, :] = board_temp[:, ::-1]
                elif(j == 1):
                    # mirror along vertical axis
                    board_temp = board.copy()
                    board_temp[:, :] = board_temp[:, ::-1]
                else:
                    # mirror along horizontal axis
                    board_temp = board.copy()
                    board_temp[:, :] = board_temp[::-1, :]
                # check for duplicates due to symmetry
                duplicate = False
                for i in range(len(board_list)-1):
                    if((board_temp == i).all()):
                        # duplicate, do not add
                        duplicate = True
                        break
                if(not duplicate):
                    board_list.append(board_temp.copy())

            # step one rotation forward
            board = board.T
            board[:, :] = board[:, ::-1]
            board = board.copy()

        for i in board_list:
            obstacles_board[index, :, :] = i.copy()
            index += 1

end_time = time.time()
hrs, mins, secs, msecs = time_calculate(start_time, end_time)
print('Generated a total of {:d} boards in {:3d}hrs {:2d}mins {:2d}secs {:.3f}ms. Saving to disk'\
      .format(index, hrs, mins, secs, msecs))

with open('models/{:s}/obstacles_board'.format(version), 'wb') as f:
    pickle.dump(1-obstacles_board[:index, :, :], f)
