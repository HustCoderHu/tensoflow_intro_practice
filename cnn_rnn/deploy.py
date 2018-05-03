import numpy as np
import random as rd

import sys
import os
import os.path as path
from os.path import join as pj
import tensorflow as tf
import cv2 as cv

import cnn
import rnn

LOW_FREQ = {
  'interval': 5,
  'batchsz': 10
}

HIGH_FREQ = {
  'interval': 2,
  'batchsz': 25
}

frameInterval = 5
cnn_batchsz = 10

# cnn 在 batch中检测到事件(比如火)的比例
# 达到此值就触发rnn的事件剧烈程度(比如火势)分析
trigger_rnn = 0.8

# tensor
t_clip = None
t_feature2rnn = None

t_pred_vec = None
t_pred_avg = None
t_rnn_pred = None
cnn_model = None
rnn_model = None

# clip shape (seqLen, h, w, 3)
def main(videoPath):
  global t_clip
  global t_feature2rnn

  global t_pred_vec
  global t_pred_avg
  global t_rnn_pred
  global cnn_model
  global rnn_model

  t_clip = tf.placeholder(tf.float32, [None, 250, 250, 3], name='clip_input')
  with tf.name_scope('CNN'):
    cnn_model = cnn.CNN(data_format='NHWC')
    cnn_out = cnn_model(t_clip)
    # pred_vec = tf.argmax(cnn_out, axis=1, output_type=tf.int32) # (batch,)
    t_pred_vec = tf.argmax(cnn_out, axis=1, output_type=tf.float32) # (batch,)
    t_pred_avg = tf.reduce_mean(t_pred_vec)

  t_feature2rnn = tf.placeholder(tf.float32, [1, 128], name='rnn_input')
  with tf.name_scope('RNN'):
    rnn_model = rnn.RNN('GRU', n_hidden=10, n_classes=2)
    logits = rnn_model(t_feature2rnn, batch_sz=1)
    t_rnn_pred = tf.nn.softmax(logits) # 可信度
    # 0:a, 1:1-b, 2:1-a-b

  sess_conf = tf.ConfigProto()
  sess_conf.gpu_options.allow_growth = True
  sess_conf.gpu_options.per_process_gpu_memory_fraction = 0.75
  with tf.Session(config= sess_conf) as sess:
    handelVideo(sess, videoPath)
  
  return

# [] item (seqLen, h, w, 3)
def genClips(cap, maxCnt=4):
  return 

def handelVideo(sess, videoPath):
  cap = cv.VideoCapture(video_path)
  frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
  cap.read()

  freq = LOW_FREQ
  iframe = 0
  l = []
  nExample = 0
  while(cap.isOpened):
    ret, frame = cap.read()
    if ret == False:
      break
    iframe +=1
    batchFrame = None
    if iframe % freq['interval'] == 0:
      l.append(frame[np.newaxis, :, :, :])
      nExample += 1
      if nExample == freq['batchsz']:
        batchFrame = np.concatenate(l)
        feature2rnn, pred_avg = sess.run(
            [cnn_model.feature2rnn, t_pred_avg],
            feed_dict={t_clip: batchFrame, cnn_model.training: False,
               rnn_model.training: False})
        if freq == HIGH_FREQ:
          rnn_pred = sess.run(t_rnn_pred,
            feed_dict={t_feature2rnn: feature2rnn, cnn_model.training: False,
               rnn_model.training: False})
        # 触发高频采样
        if pred_avg >= trigger_rnn:
          freq = HIGH_FREQ

  for idx in range(frame_count-seqLen+1):
    pass

  _pred = sess.run(predicts, feed_dict={
    # clip: ,
    cnn_model.training: False,
    rnn_model.training: False
  })
  # if 
  cap.release()

def tst():
  # nda0 = np.zeros([250, 250, 3], np.uint8)
  # nda1 = np.ones([250, 250, 3], np.uint8)
  # nda2 = np.zeros([250, 250, 3], np.uint8)

  l = []
  for i in range(25):
    nda = np.zeros([250, 250, 3], np.uint8)
    l.append(nda[np.newaxis,:,:,:])
  cclip = np.concatenate(l)
  print(cclip.shape)
  return


if __name__ == '__main__':
  tst()
  # main()
