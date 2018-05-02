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

frameInterval = 5
seqLen = 10

# clip shape (seqLen, h, w, 3)
def handelVideo(videoPath):
  clip = tf.placeholder(tf.float32, [None, 250, 250, 3], name='clip_input')
  with tf.name_scope('CNN'):
    cnn_model = cnn.CNN(data_format='NHWC')
    _t = cnn_model(clip)
  with tf.name_scope('RNN'):
    rnn_model = rnn.RNN('GRU', n_hidden=20, n_classes=2)
    logits = rnn_model(cnn_model.feature2rnn, batch_sz=1)

  predicts = tf.nn.softmax(logits)

  sess_conf = tf.ConfigProto()
  sess_conf.gpu_options.allow_growth = True
  sess_conf.gpu_options.per_process_gpu_memory_fraction = 0.75
  with tf.Session(config= sess_conf) as sess:
    cap = cv.VideoCapture(video_path)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.read()
    for idx in range(frame_count-seqLen+1):

      pass

    _pred = sess.run(predicts, feed_dict={
      clip: ,
      cnn_model.training: False,
      rnn_model.training: False
    })
  cap.release()

# [] item (seqLen, h, w, 3)
def genClips(cap, maxCnt=4):
  
  

  
