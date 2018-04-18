import numpy as np

import tensorflow as tf
from tensorflow import layers as tl
from tensorflow import nn as nn
import tensorflow.contrib as tc

class Simplest:
  def __init__(self, num_classes, data_format='NHWC', dtype=tf.float32):
    self.training = tf.placeholder(tf.bool, [])
    
    self.w_initer=tc.layers.xavier_initializer(tf.float32)
    self.data_format = data_format
    self.dtype = dtype
    self.layer_data_format = 'channels_last' if data_format == 'NHWC' \
        else 'channels_first'
    
    # self.conv3d = tl.Conv3D(filters=64, kernel_size=(1,7,7), strides=(1,2,2),
    #     padding=self.layer_data_format, use_bias=False, kernel_initializer=self.w_initer)

    axis = 1 if self.data_format=="NCHW" else -1
    self.bn = tl.BatchNormalization(axis=axis, scale=False, fused=True)
    # out = self.bn(in, training=True) False for eval

    return
  
  def Conv2D(self, filters, kernel_size, strides=1, padding='valid', use_bias=False):
    return tl.Conv2D(filters, kernel_size, strides,
        padding=padding, data_format=self.layer_data_format, use_bias=use_bias, 
        kernel_initializer=self.w_initer)
  def FC(self, units, use_bias=False):
    return tl.Dense(units, use_bias=use_bias, kernel_initializer=self.w_initer)
  def Pool2D(self, pool_size, strides, pool_type):
    if pool_type == 'max':
      return tl.MaxPooling2D(pool_size, strides, data_format=self.layer_data_format)
    else :
      return tl.AveragePooling2D(pool_size, strides, data_format=self.layer_data_format)
  
  def global_avg_pool(self, x):
    if self.data_format == 'NHWC':
      out = tf.reduce_mean(x, (1, 2))
    else :
      out = tf.reduce_mean(x, (2, 3))
    return out

  def __call__(self, inputs):
    # self.training
    pr_shape = lambda var : print(var.shape)
    
    # CNN
    out = self.Conv2D(32, 5)(inputs)
    out = tf.nn.relu(out)
    out = self.Pool2D(2, 2, 'max')(out)

    out = self.Conv2D(64, 5)(out)
    out = tf.nn.relu(out)
    out = self.Pool2D(2, 2, 'max')(out)
    # out = self.bn(out, training=self.training)

    out = tl.Flatten()(out)
    out = self.FC(128)(out)
    out = tl.Dropout(0.5)(out, training=self.training)
    out = self.FC(32)(out)
    pr_shape(out)

    # RNN
    # n_step = 50
    n_hidden = 6

    # if out.shape[0] < n_step:
    #   sequence_length = (out.shape[0])
    
    out = tf.reshape(out, [1, -1, out.shape[-1]])
    cell = nn.rnn_cell.GRUCell(n_hidden)
    initial_state = cell.zero_state(out.shape[0], tf.float32)

    # dynamic_rnn inputs shape = [batch_size, max_time, ...]
    # outs shape = [batch_size, max_time, cell.output_size]
    # states shape = [batch_size, cell.state_size]
    
    n_step = int(out.shape[1]) # n_step
    # print(type(n_step))
    # pr_shape(out)
    outs, states = nn.dynamic_rnn(cell, out, sequence_length=(n_step, ),
      initial_state=initial_state, dtype=tf.float32)
    pr_shape(states)

    final_state = states[-1] # list len = n_steps
    pr_shape(states) # (n_hidden,)

    return final_state

class CNN:
  def __init__(self, num_classes, data_format='NHWC', dtype=tf.float32):
    self.training = tf.placeholder(tf.bool, [])
    
    self.w_initer=tc.layers.xavier_initializer(tf.float32)
    self.data_format = data_format
    self.dtype = dtype
    self.layer_data_format = 'channels_last' if data_format == 'NHWC' \
        else 'channels_first'
    
    # self.conv3d = tl.Conv3D(filters=64, kernel_size=(1,7,7), strides=(1,2,2),
    #     padding=self.layer_data_format, use_bias=False, kernel_initializer=self.w_initer)

    axis = 1 if self.data_format=="NCHW" else -1
    self.bn = tl.BatchNormalization(axis=axis, scale=False, fused=True)
    self.feature2rnn = 0
    # out = self.bn(in, training=True) False for eval

    return
  
  def Conv2D(self, filters, kernel_size, strides=1, padding='valid', use_bias=False):
    return tl.Conv2D(filters, kernel_size, strides,
        padding=padding, data_format=self.layer_data_format, use_bias=use_bias, 
        kernel_initializer=self.w_initer)
  def FC(self, units, use_bias=False):
    return tl.Dense(units, use_bias=use_bias, kernel_initializer=self.w_initer)
  def Pool2D(self, pool_size, strides, pool_type):
    if pool_type == 'max':
      return tl.MaxPooling2D(pool_size, strides, data_format=self.layer_data_format)
    else :
      return tl.AveragePooling2D(pool_size, strides, data_format=self.layer_data_format)
  
  def global_avg_pool(self, x):
    if self.data_format == 'NHWC':
      out = tf.reduce_mean(x, (1, 2))
    else :
      out = tf.reduce_mean(x, (2, 3))
    return out
  
  def __call__(self, inputs):
    # self.training
    pr_shape = lambda var : print(var.shape)
    
    # CNN
    out = self.Conv2D(32, 5)(inputs)
    out = tf.nn.relu(out)
    out = self.Pool2D(2, 2, 'max')(out)

    out = self.Conv2D(64, 5)(out)
    out = tf.nn.relu(out)
    out = self.Pool2D(2, 2, 'max')(out)
    # out = self.bn(out, training=self.training)

    out = tl.Flatten()(out)
    out = self.FC(128)(out)
    
    self.feature2rnn = out

    out = tl.Dropout(0.5)(out, training=self.training)
    out = self.FC(32)(out)
    pr_shape(out)
    
    return out

if __name__ == '__main__':
  model = Simplest(8)
  x = tf.constant(0.0, shape=[32, 250, 250, 3])

  pr_shape = lambda var : print(var.shape)
  logits = model(x)
  print(logits.shape)

# CNN Long Short-Term Memory Networks
# https://machinelearningmastery.com/cnn-long-short-term-memory-networks/