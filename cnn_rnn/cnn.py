import numpy as np

import tensorflow as tf
from tensorflow import layers as tl
from tensorflow import nn as nn
import tensorflow.contrib as tc


class CNN:
  def __init__(self, data_format='NHWC', dtype=tf.float32):
    self.is_training = tf.placeholder(tf.bool, [])
    
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
    self.channelAxis = 1 if self.data_format=="NCHW" else -1
    self.i = 0
    return
  
  def Conv2D(self, filters, kernel_size, strides=1, padding='same', use_bias=False):
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
    # self.is_training
    pr_shape = lambda var : print(var.shape)
    
    # CNN
    # with tf.name_scope('')
    if self.data_format == 'NCHW':
      inputs = tf.transpose(inputs, [0, 3, 1, 2])
    # print(inputs.shape.dims)
    if castFromUint8:
      inputs = tf.cast(inputs, self.dtype)
    
    with tf.variable_scope("init_conv") :
      out = self.Conv2D(32, 3, strides=2)(inputs)
      # pr_shape(out)
      out = self.BN()(out, training=self.is_training)
      out = tf.nn.relu6(out)
    with tf.variable_scope("body") :
      out = self._inverted_bottleneck(out, 6, 16, stride=1)
      # pr_shape(out)
      # out = tf.nn.relu6(out)
      out = self._inverted_bottleneck(out, 6, 24, stride=2)
      # pr_shape(out)
      # out = tf.nn.relu6(out)
      out = self._inverted_bottleneck(out, 6, 32, stride=2)
      # out = tf.nn.relu6(out)
      out = self._inverted_bottleneck(out, 6, 64, stride=2)
      # out = tf.nn.relu6(out)
    out = self.Pool2D(4, 3, 'max')(out)
    pr_shape(out)

    # residualParam = []
    # param = {'filters': 32, 'kernel_sz': 5, 'strides': 2}
    # residualParam.append(param)
    # param = {'filters': 48, 'kernel_sz': 3, 'strides': 1}
    # residualParam.append(param)
    # with tf.variable_scope("res1"):
    #   out = self.residual(inputs, residualParam)
    # out = tf.nn.relu6(out)

    # with tf.variable_scope("conv1_relu"):
    #   out0 = self.Conv2D(48, 5)(inputs)
    #   out0 = self.BN()(out0, training=self.is_training)
    #   out0 = tf.nn.relu6(out0)

    # with tf.variable_scope("conv1_relu"):
    # with tf.variable_scope("pool1"):
    #   out = self.Pool2D(3, 2, 'max')(out)

    # with tf.variable_scope("conv2_relu"):
    #   out = self.Conv2D(48, 5)(out)
    #   out = self.BN()(out, training=self.is_training)
    #   out = tf.nn.relu6(out)
    # with tf.variable_scope("pool2"):
    #   out = self.Pool2D(3, 3, 'max')(out)

    # with tf.variable_scope("conv3_relu"):
    #   out = self.Conv2D(48, 5)(out)
    #   out = self.BN()(out, training=self.is_training)
    #   out = tf.nn.relu6(out)
    # with tf.variable_scope("pool3"):
    #   out = self.Pool2D(3, 3, 'max')(out)

    with tf.variable_scope('fc1'):
      out = tl.Flatten()(out)
      out = self.FC(128)(out)

    self.feature2rnn = out
    
    with tf.variable_scope("dropout"):
      out = tl.Dropout(0.5)(out, training=self.is_training)
    with tf.variable_scope('fc2'):
      out = self.FC(2)(out)
    # pr_shape(out)
    
    return out
  
  # layerParam = [] item 
  # {'filters': int, 'kernel_sz': int, 'strides': int}
  def residual(self, inputs, layerParam):
    shortcut = inputs
    # 第一个
    param = layerParam[0]
    out0 = self.Conv2D(param['filters'], param['kernel_sz'], param['strides'])(inputs)
    out0 = self.BN()(out0, training=self.is_training)
    out0 = tf.nn.relu6(out0)

    # 2 ~ n-1
    for idx in range(1, len(layerParam)-1):
      param = layerParam[idx]
      out0 = self.Conv2D(param['filters'], param['kernel_sz'], param['strides'])(out0)
      out0 = self.BN()(out0, training=self.is_training)
      out0 = tf.nn.relu6(out0)

    # 第n个
    param = layerParam[-1]
    out0 = self.Conv2D(param['filters'], param['kernel_sz'], param['strides'])(out0)
    out0 = self.BN()(out0, training=self.is_training)
    
    conv1x1_stride = 1
    for param in layerParam:
      conv1x1_stride *= param['strides']

    if shortcut.shape[self.channelAxis] != out0.shape[self.channelAxis] \
      or conv1x1_stride != 1:
      param = layerParam[-1]
      print('conv1x1 to adjust channel')
      shortcut = self.Conv2D(param['filters'], 1, conv1x1_stride)(out0)

    # tf.variable_scope("conv1_relu"):
    return out0 + shortcut
  
  def _inverted_bottleneck_orig(self, x, rate_channels, channels, stride=1):
    init = tf.contrib.layers.xavier_initializer(dtype=tf.float32)

    in_shape = x.get_shape().as_list()
    if self.data_format == "NCHW":
      x_channels = in_shape[1]
      strides_4d = [1, 1, stride, stride]
    else :
      x_channels = in_shape[-1]
      strides_4d = [1, stride, stride, 1]

    # print('type(x_channels)')
    # print(type(x_channels)) # int

    with tf.variable_scope("inverted_bottleneck_{}_t{}_s{}".format(
        self.i, rate_channels, stride) ) :

      # --- 1x1 pointwise
      n_out = rate_channels * n_in
      branch_1 = self.Conv2D(n_out, 1)(x)
      branch_1 = self.BN()(branch_1, training=self.is_training)
      branch_1 = tf.nn.relu6(branch_1)

      # --- 3x3 depthwise
      n_in = n_out
      # print(n_in)
      # print(n_out)
      channel_multiplier = 1
      w3x3 = tf.get_variable("w3x3_br1", [3, 3, n_in, channel_multiplier],
                             initializer=init)
      # b3x3 = b1x1 = tf.get_variable("b3x3", [n_out], initializer=init)
      branch_1 = tf.nn.depthwise_conv2d(branch_1, w3x3, strides=strides_4d,
                                      padding="SAME", data_format=self.data_format)
      branch_1 = self.BN()(branch_1, training=self.is_training)
      branch_1 = tf.nn.relu6(branch_1)

      # --- 1x1
      branch_1 = self.Conv2D(channels, 1)(branch_1)
      branch_1 = self.BN()(branch_1, training=self.is_training)
      # print(self.i)
      if stride == 1 and x_channels==channels:
        out = branch_1+branch_0
      else:
        out = branch_1
      # out = tf.add(branch_0, branch_1)
    self.i += 1
    return out

  def _inverted_bottleneck(self, x, rate_channels, channels, stride=1):
    init = tf.contrib.layers.xavier_initializer(dtype=tf.float32)

    in_shape = x.get_shape().as_list()
    if self.data_format == "NCHW":
      x_channels = in_shape[1]
      strides_4d = [1, 1, stride, stride]
    else :
      x_channels = in_shape[-1]
      strides_4d = [1, stride, stride, 1]

    with tf.variable_scope("inverted_bottleneck_{}_t{}_s{}".format(
        self.i, rate_channels, stride) ) :
      channel_multiplier = 1

      if x_channels == channels and stride==1:
        branch_0 = x
      else:
        if stride == 2:
          branch_0 = self.Conv2D(channels, 3, stride)(x)
        else :
          branch_0 = self.Conv2D(channels, 1, stride)(x)
        branch_0 = self.BN()(branch_0, training=self.is_training)

      # --- 1x1 pointwise
      n_out = rate_channels * x_channels
      branch_1 = self.Conv2D(n_out, 1)(x)
      branch_1 = self.BN()(branch_1, training=self.is_training)
      branch_1 = tf.nn.relu6(branch_1)

      # --- 3x3 depthwise
      n_in = n_out
      # print(n_in)
      # print(n_out)
      w3x3 = tf.get_variable("w3x3_br1", [3, 3, n_in, channel_multiplier],
                             initializer=init)
      # b3x3 = b1x1 = tf.get_variable("b3x3", [n_out], initializer=init)
      branch_1 = tf.nn.depthwise_conv2d(branch_1, w3x3, strides=strides_4d,
                                      padding="SAME", data_format=self.data_format)
      branch_1 = self.BN()(branch_1, training=self.is_training)
      branch_1 = tf.nn.relu6(branch_1)

      # --- 1x1
      branch_1 = self.Conv2D(channels, 1)(branch_1)
      branch_1 = self.BN()(branch_1, training=self.is_training)

      out = tf.add(branch_0, branch_1)
    self.i += 1
    return out


if __name__ == '__main__':
  model = CNN(data_format='NCHW')
  x = tf.constant(0, dtype=tf.uint8, shape=[1, 240, 320, 3])

  pr_shape = lambda var : print(var.shape)
  logits = model(x)
  print(logits.shape)

# CNN Long Short-Term Memory Networks
# https://machinelearningmastery.com/cnn-long-short-term-memory-networks/