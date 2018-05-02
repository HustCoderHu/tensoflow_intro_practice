import numpy as np

import tensorflow as tf
from tensorflow import layers as tl
from tensorflow import nn as nn
import tensorflow.contrib as tc

class RNN:
  def __init__(self, rnntype, n_hidden, n_classes):
    self.training = tf.placeholder(tf.bool, [])
    self.rnntype = rnntype
    self.n_hidden = n_hidden
    self.n_classes = n_classes

  def __call__(self, inputs, batch_sz):
    pr_shape = lambda var : print(var.shape)

    if self.rnntype == "GRU" :
      print("rnntype: " + self.rnntype)
      cell = nn.rnn_cell.GRUCell(self.n_hidden)
    else :
      print("rnntype: " + self.rnntype)
      cell = nn.rnn_cell.LSTMCell(self.n_hidden)
    
    initial_state = cell.zero_state(batch_sz, tf.float32)
    # initial_state = cell.zero_state(inputs.shape[0], tf.float32)

    # dynamic_rnn inputs shape = [batch_size, max_time, ...]
    # outs shape = [batch_size, max_time, cell.output_size]
    # states shape = [batch_size, cell.state_size]
    # n_step = int(inputs.shape[1]) # n_step

    outs, states = nn.dynamic_rnn(cell, inputs, initial_state=initial_state, 
        dtype=tf.float32)
    print('outs shape: ')
    pr_shape(outs) # (batch_sz, max_time, n_hidden)
    # final_state = states[-1] # 
    print('states shape: ')
    pr_shape(states) # (batch_sz, n_hidden)

    FC = tl.Dense(self.n_classes, use_bias=True, 
        kernel_initializer=tc.layers.xavier_initializer(tf.float32))
    outs = FC(states)

    return outs

if __name__ == '__main__':
  model = RNN('GRU', n_hidden=20, n_classes=3)
  x = tf.constant(0.0, shape=[10, 30, 128])

  pr_shape = lambda var : print(var.shape)
  logits = model(x)
  print(logits.shape)