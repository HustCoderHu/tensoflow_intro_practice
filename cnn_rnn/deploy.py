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

# 采样频率及长度
LOW_FREQ = {
  'interval': 5,
  'batchsz': 15
}
HIGH_FREQ = {
  'interval': 2,
  'batchsz': 25
}

# cnn 在 batch中检测到事件(比如火)的比例
# 达到此值就触发rnn的事件剧烈程度(比如火势)分析
trigger_rnn = {
  'HIGH': 0.85,
  'LOW': 0.15
}

cnn_label = {
  'nofire': 0,
  'fire': 1
}

cnn_ckpt = r'D:\Lab408\monitored_sess_log_all_two_4.17\ckpts\model.ckpt-25700'
rnn_ckpt = r'D:\Lab408\cnn_rnn\monsess_log-0502-hidden10\ckpts\model.ckpt-11200'
# cnn_ckpt = r'/home/hzx/tensorflow_intro_practice/cnn_rnn/cnn_fire_ckpt/model.ckpt-25700'
# rnn_ckpt

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
  
  # build graph
  cnn_graph = tf.Graph()
  with cnn_graph.as_default():
    t_clip = tf.placeholder(tf.float32, [None, 250, 250, 3], name='clip_input')
    with tf.name_scope('CNN'):
      cnn_model = cnn.CNN(data_format='NHWC')
      cnn_out = cnn_model(t_clip) # (batch, 2)
      print(cnn_out.shape)
      # pred_vec = tf.argmax(cnn_out, axis=1, output_type=tf.int32) # (batch,)
      cnn_out_avg = tf.reduce_sum(cnn_out, axis=0) # (classes, ) batch维度上累加
      # print(cnn_out_avg.shape)
      t_pred_avg = tf.nn.softmax(cnn_out_avg)
      # print(t_pred_avg.shape)
      # print(t_pred_avg.dtype)
      # return
      # t_pred_vec = tf.argmax(cnn_out, axis=1, output_type=tf.float32) # (batch,)
      # t_pred_avg = tf.reduce_mean(t_pred_vec)
  
  rnn_graph = tf.Graph()
  with rnn_graph.as_default():
    t_feature2rnn = tf.placeholder(tf.float32, [1, None, 128], name='rnn_input')
    with tf.name_scope('RNN'):
      rnn_model = rnn.RNN('GRU', n_hidden=10, n_classes=2)
      logits = rnn_model(t_feature2rnn, batch_sz=1)
      t_rnn_pred = tf.nn.softmax(logits) # 可信度 (1, 2)
      # 0:a, 1:1-b, 2:1-a-b
  
  # create sess & restore param
  sess_conf = tf.ConfigProto()
  sess_conf.gpu_options.allow_growth = True
  # sess_conf.gpu_options.per_process_gpu_memory_fraction = 0.75
  cnn_sess = tf.Session(graph=cnn_graph, config=sess_conf)
  rnn_sess = tf.Session(graph=rnn_graph, config=sess_conf)

  with cnn_graph.as_default():
    saver = tf.train.Saver()
    saver.restore(cnn_sess, cnn_ckpt)
  with rnn_graph.as_default():
    saver = tf.train.Saver()
    saver.restore(rnn_sess, rnn_ckpt)
  # saver.restore(cnn_sess, cnn_ckpt)
  # saver.restore(rnn_sess, rnn_ckpt)
  
  # work on video
  videoPath = r'D:\Lab408\cnn_rnn\src_dir\5.mp4'
  # videoPath = r'D:\Lab408\cnn_rnn\src_dir\fire-smoke-small(13).avi'
  # videoPath = r'D:\Lab408\cnn_rnn\src_dir\NIST Re-creation of The Station Night Club fire   without sprinklers (1).mp4'
  handelVideo(cnn_sess, rnn_sess, videoPath)
  
  cnn_sess.close()
  rnn_sess.close()
  return

def handelVideo(cnn_sess, rnn_sess, videoPath):
  cap = cv.VideoCapture(videoPath)
  frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
  fps = cap.get(cv.CAP_PROP_FPS) # float
  print(type(fps))
  height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
  width = cv.CAP_PROP_FRAME_WIDTH
  prePolicy = FramePreprocessPolicy((height, width), (250, 250))
  print('frame_count: {}'.format(frame_count))
  print('fps: {}'.format(fps))

  freq = LOW_FREQ
  frameIdx = -1 # 最后读取帧的位置
  while(cap.isOpened):
    # 拼成 batch 处理
    batchFrame, frameOffset = stackFrames(cap, prePolicy, freq)
    if frameOffset == 0:
      break
    frameIdx += frameOffset
    feature2rnn, pred_avg = cnn_sess.run(
        [cnn_model.feature2rnn, t_pred_avg],
        feed_dict={t_clip: batchFrame, cnn_model.training: False})
    # 触发高频采样
    fire_confidence = pred_avg[cnn_label['fire']]
    if fire_confidence >= trigger_rnn['HIGH']:
      freq = HIGH_FREQ
    # 低频
    elif fire_confidence <= trigger_rnn['LOW']:
      freq = LOW_FREQ
    # 目前处在火势分析流程中
    if freq == HIGH_FREQ:
      rnn_pred = rnn_sess.run(t_rnn_pred,
          feed_dict={t_feature2rnn: [feature2rnn], rnn_model.training: False})
      rnn_pred = rnn_pred[0]
    else:
      rnn_pred = None
    # 
    showAnalysis(frameIdx, fps, fire_confidence, rnn_pred)

  cap.release()


# [] item (seqLen, h, w, 3)
# def genClips(cap, maxBatch):

# 预处理策略
# freq: 采样参数
# retval: 帧batch, 总读取帧数量
def stackFrames(cap, prePolicy, freq):
  frameOffset = 0 # 已读帧
  nFrame = 0 # 采样帧
  interval = freq['interval']
  batchsz = freq['batchsz']
  l = []
  while nFrame < batchsz:
    ret, frame = cap.read()
    if ret == False:
      break
    frameOffset += 1
    # 帧采样
    if frameOffset % interval == 0:
      frame = prePolicy(frame)
      l.append(frame[np.newaxis, :, :, :])
      nFrame += 1
  # 拼成 batch
  # batchFrame = np.concatenate(l)
  if len(l) == 0:
    return None, 0
  else:
    return np.concatenate(l), frameOffset

def showAnalysis(frameIdx, fps, fire_confidence, rnn_pred):
  # http://blog.xiayf.cn/2013/01/26/python-string-format/
  if rnn_pred is None:
    # < 左对齐
    # 20 min 50 fps 最大帧数 30000
    # 对齐宽度 5
    content = '{:5.1f}s fire {:.3f}'.format(
        frameIdx/fps, fire_confidence)
  else:
    content = '{:5.1f}s fire {:.3f}\n  趋势 变大:{:.3f} 变小:{:.3f}'.format(
        frameIdx/fps, fire_confidence, rnn_pred[0], rnn_pred[1])
    if rnn_pred[0]>rnn_pred[1]:
     content += ' 大'
  print(content)
  return

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

  # i = 0
  for i in range(0, 5):
    if i == 3:
      break
  print('i: %d' % i)
  showAnalysis(3, [0.7, 0.3], None)
  showAnalysis(3, [0.7, 0.3], [0.88, 0.12])
  return

class FramePreprocessPolicy(object):
  def __init__(self, fromSize, toSize):
    self.ops = []
    op = lambda frame : cv.resize(frame, toSize)
    self.ops.append(op)
    return

  def __call__(self, frame):
    for op in self.ops:
      frame = op(frame)
    return frame

# END class FramePreprocessPolicy

if __name__ == '__main__':
  # tst()
  main('')
