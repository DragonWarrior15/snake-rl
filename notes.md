PolicyGradientAgent
Model Architecture
'''python
input_board = tf.keras.layers.Input((self._board_size, self._board_size, self._n_frames,))
x = tf.keras.layers.Conv2D(16, (4,4), activation = 'relu', data_format='channels_last')(input_board)
x = tf.keras.layers.Conv2D(32, (4,4), activation = 'relu', data_format='channels_last')(x)
x = tf.keras.layers.Conv2D(64, (4,4), activation = 'relu', data_format='channels_last')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation = 'relu')(x)
out = tf.keras.layers.Dense(self._n_actions, activation = 'linear', name = 'action_logits')(x)

model = tf.keras.Model(inputs = input_board, outputs = out)
'''
Loss Function
'''python
policy = tf.nn.softmax(self._model.output)
log_policy = tf.nn.log_softmax(self._model.output)
# calculate loss
J = tf.tensordot(target, log_policy, axes=2)/num_games
entropy = -tf.tensordot(policy, log_policy, axes=2)/num_games
loss = -(1-beta)*J - beta*entropy
'''
experiment results with different learning rates
(-10/10 reward changing, no food -10 reward, rewards normalized, 16 games for one update, RMSprop)
lr 0.1 -> convergence immediately to choosing right action
lr 0.01 -> convergence immediately to choosing right action
lr 0.001 -> convergence almost immediately to choosing left action
lr 0.0001 -> 50 steps to converge to choosing right action (95% of time)
lr 0.00001 -> no convergence, all actions remain almost equally probable, no increment in reward
lr 0.000001 -> no convergence, all actions remain almost equally probable (slight bias), no increment in reward
lr 0.0000001 -> no convergence, all actions remain almost equally probable (slight bias), no increment in reward

(-10/10 reward changing, no food -10 reward, rewards normalized, 32 games for one update, RMSprop)
lr 0.1 -> convergence immediately to choosing left action
lr 0.01 -> convergence immediately to choosing nothing action
lr 0.001 -> 8 steps to converge to choosing right action (95% of time)
lr 0.0001 -> 50 steps to converge to choosing left action (95% of time)
lr 0.00001 -> no convergence, all actions remain almost equally probable (slight bias), no increment in reward

(-10/10 reward changing, no food -10 reward, rewards normalized, 16 games for one update, RMSprop, one more conv layer)
lr 0.1 -> convergence immediately to choosing right action
lr 0.01 -> convergence immediately to choosing right action
lr 0.001 -> 8 steps to converge to choosing left action (95% of time)
lr 0.0001 -> 50 steps to converge to choosing left action (95% of time)
lr 0.00001 -> no convergence, all actions remain almost equally probable (slight bias), no increment in reward
lr 0.00005 -> no convergence, all actions remain almost equally probable (slight bias), no increment in reward

(-10/10 reward changing, no food -10 reward, rewards not normalized, 16 games for one update, RMSprop, one more conv layer)
lr 0.00005 -> no convergence, all actions remain almost equally probable (slight bias), no increment in reward
