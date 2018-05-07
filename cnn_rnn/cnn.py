import numpy as np

import tensorflow as tf
from tensorflow import layers as tl
from tensorflow import nn as nn
import tensorflow.contrib as tc



class CNN:
  def __init__(self, data_format='NHWC', dtype=tf.float32):
    self.training = tf.placeholder(tf.bool, [])
    
    self.w_initer=tc.layers.xavier_initializer(dtype)
    self.data_format = data_format
    self.dtype = dtype
    self.layer_data_format = 'channels_last' if data_format == 'NHWC' \
        else 'channels_first'
    
    # self.conv3d = tl.Conv3D(filters=64, kernel_size=(1,7,7), strides=(1,2,2),
    #     padding=self.layer_data_format, use_bias=False, kernel_initializer=self.w_initer)

    axis = 1 if self.data_format=="NCHW" else -1
    self.BN = lambda : tl.BatchNormalization(axis=axis, scale=False, fused=True)
    self.feature2rnn = 0
    # out = self.BN(in, training=True) False for eval
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
  
  def __call__(self, inputs, castFromUint8=True):
    # self.training
    pr_shape = lambda var : print(var.shape)
    
    # CNN
    # with tf.name_scope('')
    if self.data_format == 'NCHW':
      inputs = tf.transpose(inputs, [0, 3, 1, 2])
    # print(inputs.shape.dims)
    if castFromUint8:
      inputs = tf.cast(inputs, self.dtype)
    out = self.Conv2D(32, 5)(inputs)
    # out = self.BN()(out, training=self.training)
    out = tf.nn.relu(out)
    out = self.Pool2D(2, 2, 'max')(out)

    out = self.Conv2D(64, 5)(out)
    # out = self.BN()(out, training=self.training)
    out = tf.nn.relu(out)
    out = self.Pool2D(2, 2, 'max')(out)
    # print(self.layer_data_format)

    out = self.Conv2D(64, 5)(out)
    # out = self.BN()(out, training=self.training)
    out = tf.nn.relu(out)
    out = self.Pool2D(4, 4, 'max')(out)

    # out = self.bn()(out, training=self.training)

    out = tl.Flatten()(out)
    out = self.FC(128)(out)
    self.feature2rnn = out

    out = tl.Dropout(0.5)(out, training=self.training)
    out = self.FC(2)(out)
    # pr_shape(out)
    
    return out

if __name__ == '__main__':
  model = CNN(data_format='NCHW')
  x = tf.constant(0, dtype=tf.uint8, shape=[32, 250, 250, 3])

  pr_shape = lambda var : print(var.shape)
  logits = model(x)
  # print(logits.shape)

# CNN Long Short-Term Memory Networks
# https://machinelearningmastery.com/cnn-long-short-term-memory-networks/