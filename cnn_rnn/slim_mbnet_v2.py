import sys
import os
import os.path as path
from os.path import join as pj
from pprint import pprint

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.slim as slim

model_slim = r'D:\github_repo\models\research\slim'
model_slim = r'/home/hzx/github-repo/models/research/slim'
sys.path.append(model_slim)
from nets.mobilenet import mobilenet_v2

class MyNetV2():
  def __init__(self, n_classes, data_format='NHWC', dtype=tf.float32):
    self.is_training = tf.placeholder(tf.bool, [])
    self.n_classes = n_classes
    self.w_initer=tc.layers.xavier_initializer(dtype)
    self.data_format = data_format
    self.dtype = dtype
    self.layer_data_format = 'channels_last' if data_format == 'NHWC' \
        else 'channels_first'
    
    self.channelAxis = 1 if self.data_format=="NCHW" else -1

    self.variables_to_restore = None
    self.variables_to_save = None
    self.variables_to_train = []
    return

  def __call__(self, inputs, castFromUint8=True):
    pr_shape = lambda var : print(var.shape)
    if castFromUint8:
      inputs = tf.cast(inputs, self.dtype)
    # print(inputs.dtype)

    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(
        is_training=self.is_training)):
      # print(inputs.dtype)
      global_pool, endpoints = mobilenet_v2.mobilenet(inputs, num_classes=None)
    self.variables_to_restore = slim.get_variables() # len 260
    # 后加两层fc
    dropout_keep_prob = 0.5
    weight_decay = 0.05
    with tf.variable_scope('additional', 'fc'):
      # flatten = tf.flatten(endpoints['global_pool'])
      flatten = slim.flatten(global_pool)
      with slim.arg_scope([slim.fully_connected],
          weights_regularizer=slim.l2_regularizer(weight_decay),
          weights_initializer = tc.layers.xavier_initializer(tf.float32),
          # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
          activation_fn=None) as sc:
        net = slim.fully_connected(flatten, 128, activation_fn=None, scope='fc1')
        net = slim.dropout(net, dropout_keep_prob, is_training=self.is_training, scope='dropout')
        logits = slim.fully_connected(net, self.n_classes, activation_fn=None, scope='fc2')
    # 多出来的4个参数保存 共264
    self.variables_to_save = slim.get_variables()

    for var in self.variables_to_save:
      if var in self.variables_to_restore:
        continue
      self.variables_to_train.append(var)
    # pr_shape(out)
    return logits

  def port2myNet(self, standardV2 = r'', myNetCkpt = r''):
    x = tf.placeholder(dtype=tf.uint8, shape=[None, 240, 320, 3])
    labels = tf.placeholder(tf.int32, [None])
    logits = self(x)

    init_fn0 = slim.assign_from_checkpoint_fn(standardV2, self.variables_to_restore)

    with tf.name_scope('cross_entropy'):
      loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  
    with tf.name_scope('optimizer'):
      optimizer = tf.train.AdamOptimizer(1e-4)
    # 每个参数有 Adam Adam_1 也要保持
    # 否则 tf.train.MonitoredSession 出错 Adam not found in checkpoint
    variables_to_train = []
    trainable_variables = tf.trainable_variables()
    for var in trainable_variables:
      if var in model.variables_to_restore:
        continue
      variables_to_train.append(var)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step(),
          variables_to_train)
    saver = tf.train.Saver()

    op_init1 = tf.variables_initializer(tf.global_variables())
    op_init2 = tf.variables_initializer(tf.local_variables())
    op_group = tf.group(op_init1, op_init2)
    
    sess_conf = tf.ConfigProto()
    sess_conf.gpu_options.allow_growth = True
    with tf.Session(config= sess_conf) as sess:
      sess.run(op_group) # fc参数需要初始化
      init_fn0(sess)
      # restorer.save(sess, saveto)
      saver.save(sess, myNetCkpt)
    print('finish port2myNet')
    return

if __name__ == '__main__':
  model = MyNetV2(2)
  standardV2 = r'D:\Lab408\tfslim\mobileNetV2\mobilenet_v2_1.0_224.ckpt'
  myNetCkpt = r'D:\Lab408\tfslim\mynetv2\mynetv2.ckpt'
  model.port2myNet(standardV2, myNetCkpt)