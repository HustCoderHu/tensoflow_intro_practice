import numpy as np
import random as rd
import multiprocessing as mp
import time

import sys
import os
import os.path as path
from os.path import join as pj
import tensorflow as tf

import ForwardDataset

print(os.getcwd())
cwd = r'/home/tensorflow_intro_practice'
cwd = r'E:\github_repo\tensorflow_intro_practice'
sys.path.append(pj(cwd, 'cnn_rnn'))

import cnn

# a = '011'
# b = int(a)
# print(b)

# dir = r'D:\Lab408\cnn_rnn'

# os.chdir(dir)

# for i in range(80, 101):
#   with open('%06d.txt' % i, 'w') as f:
#     f.write(str(i))
# l = os.listdir(os.getcwd())
# print(l)

t_logits = 0
model = 0

def main():
  videoIdList = range(1, 78)
  videoIdList = range(1, 5)
  videoIdList = range(5, 9)
  videoIdList = range(9, 13)
  videoIdList = range(13, 31)
  videoIdList = range(31, 78)
  for videoId in videoIdList:
    p = mp.Process(target=handleAvideo, args=(videoId,))
    p.start()
    p.join()
  
  print('--- end')

def handleAvideo(videoId):
  global t_logits
  global model
  
  srcDir = r'/home/all_data'
  dstDir = r'/home/all_data/77featureVectorNpy'
  srcDir = r'D:\Lab408\cnn_rnn\src_dir'
  dstDir = r'D:\Lab408\cnn_rnn\dst_dir'

  # videoIdList = range(11, 12)
  if not path.exists(dstDir):
    os.mkdir(dstDir)
  
  dataset = ForwardDataset.FD(srcDir, videoId)
  model = cnn.CNN(data_format = 'NHWC')
  # model = cnn.CNN(data_format = 'NCHW')

  inputx, t_absName = dataset(batch_sz=30, prefetch_batch=3)
  # print('--- inputx shape: %s' % inputx.shape)
  # print('--- t_absName shape: %s' % t_absName.shape)
  t_logits = model(inputx)
  # print(logits.shape.dims)
  # print(model.feature2rnn.shape.dims)
  # return 
  
  saver = tf.train.Saver()
  
  ckptDir = r'/home/tensorflow_intro_practice/cnn_rnn/cnn_fire_ckpt'
  # ckptDir = r'D:\Lab408\monitored_sess_log_all_two_4.17\monitored_sess_log\ckpts'
  ckptDir = r'D:\Lab408\monitored_sess_log_all_two_4.17\ckpt'

  # state = tf.train.get_checkpoint_state(ckptDir)
  # print(type(state))
  # if ckpt and ckpt.model_checkpoint_path:

  latestCkpt = tf.train.latest_checkpoint(ckptDir)
  # print(latestCkpt)
  # return

  sess_conf = tf.ConfigProto()
  sess_conf.gpu_options.allow_growth = True
  sess_conf.gpu_options.per_process_gpu_memory_fraction = 0.9
  
  with tf.Session(config= sess_conf) as sess:
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, latestCkpt)
    concat = sess.run(model.feature2rnn)
    while True:
      try:
        feature2rnn, fname = sess.run([model.feature2rnn, t_absName])
        concat = np.concatenate((concat, feature2rnn))
        print(fname[-1])

        # fname = sess.run(t_absName)
        # print('fname.shape[0]: %d' % fname.shape[0])
        # for i in range(fname.shape[0]):
        #   print(fname[i])

      except tf.errors.OutOfRangeError as identifier:
        print(identifier.message)
        break
    
    print('concat shape: %d' % concat.shape[0])
    np.save(pj(dstDir, '%03d' % videoId), concat)
  return

def tst():
  a = np.zeros([128], np.float32)
  b = np.ones([128], np.float32)
  c = np.concatenate(([a],[b]))
  print(c.shape) # (2, 128)
  print(c[0, 0])
  print(c[1, 0])
  c = np.concatenate((c, [a]))
  c = np.concatenate((c, [b]))
  print(c.shape) # (4, 128)P
  print(c[2, 0])
  print(c[3, 0])
  return

  batch_vec = np.zeros((16,128), np.float)
  f = r'D:\Lab408\cnn_rnn\src_dir\011\000000.jpg'
  img = cv.imread(f)
  print(img.shape)

  f = r'D:\Lab408\cnn_rnn\dst_dir\011\000054.npy'
  nda = np.load(f)
  print(nda)
  print(nda.shape) # (128,)
  

  return

if __name__ == '__main__':
  # tst()
  # handleAvideo(11)
  main()