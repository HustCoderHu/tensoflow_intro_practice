import os
import os.path as path
from os.path import join as pj
from concurrent.futures import ProcessPoolExecutor

import json
import numpy as np
import random as rd

import tensorflow as tf

labelMap = {
  'raging': 0,
  'weaking': 1,
  'invariant': 2
}
labelList = ['raging', 'weaking', 'invariant']

videoNpyRoot = r'D:\Lab408\cnn_rnn\77featureVectorNpy'
jsonFile = r'D:\Lab408\cnn_rnn\label.json'
recordDir = r'D:\Lab408\cnn_rnn'

videoNpyRoot = r'/home/hzx/all_data/77featureVectorNpy'
jsonFile = r'/home/hzx/all_data/label.json'
recordDir = r'/home/hzx/all_data'

seqLen = 32
step = 1
recordFile = pj(recordDir, 'seqlen-%d-step-%d.tfrecord' % (seqLen, step))

def main():
  exampleList = genExampleList()
  # buildRecord(exampleList, recordFile)
  mp_buildRecord(exampleList, recordFile)

def genExampleList():
  with open(jsonFile) as f:
    dat = json.load(f)
  dat = dat['fire_clips']

  resultList= []
  max_workers = 1
  max_workers = os.cpu_count()  # ProcessPoolExecutor
  # 多进程 切片
  with ProcessPoolExecutor(max_workers) as executor:
    for itr in dat:
      itr['video_root'] = videoNpyRoot
      # res = executor.map()
      res = executor.submit(worker_perVideoNpy, itr)
      resultList.append(res)
      # break
    executor.shutdown()
  print('--- worker_perVideoNpy executor shutdown()')

  exampleList = []
  for res in resultList:
    # result = res.result()
    # print(type(result))
    exampleList.extend(res.result())
  print('--- genExampleList len: %d' % len(exampleList)) # 82115
  return exampleList

# 切片进程
# return [] # item {'vec': xxx, 'label': xxx}
def worker_perVideoNpy(param_dict):
  it = param_dict
  video_root = it['video_root']
  vedio_dir = '%03d' % int(it['vedio_dir'])
  fire_begin_frame = int(it['fire_begin_frame'])
  fire_biggest_frame = int(it['fire_biggest_frame'])
  fire_over_frame = int(it['fire_over_frame'])
  
  print(vedio_dir)
  print(' fire_begin_frame: %d' % fire_begin_frame)
  print(' fire_biggest_frame: %d' % fire_biggest_frame)
  print(' fire_over_frame: %d' % fire_over_frame)
  
  mVideoFV = np.load(pj(video_root, vedio_dir)+'.npy')
  print(" shape: " + str(mVideoFV.shape))

  exampleList = [] # item {'vec': xxx, 'label': xxx}
  # 火变大 label 0
  border = fire_biggest_frame + 1
  for i in range(fire_begin_frame, border, step):
    end = i+seqLen
    if end > border: # 分片越界
      break
    # aSeq = mVideoFV[np.newaxis, i:end, :]
    aSeq = mVideoFV[i:end, :]
    exampleList.append({'vec': aSeq, 'label': 0})
  # print('label 0: %d' % len(exampleList))

  # 变小 label 1
  border = fire_over_frame + 1
  for i in range(fire_biggest_frame, border, step):
    end = i+seqLen
    if end > border: # 分片越界
      break
    # aSeq = mVideoFV[np.newaxis, i:end, :]
    aSeq = mVideoFV[i:end, :]
    exampleList.append({'vec': aSeq, 'label': 1})

  return exampleList

# 单进程 构造 tfexample 并写入
# exampleList: [{'vec': xxx, 'label': xxx}, {}, ..]
# recordFile: /path/to/xxx.tfrecord
def buildRecord(exampleList, recordFile):
  rd.shuffle(exampleList)
  # val 必须是 list
  float_lda = lambda val : tf.train.Feature(
      float_list=tf.train.FloatList(value=val) )
  int64_lda = lambda val : tf.train.Feature(
      int64_list=tf.train.Int64List(value=val) )
  # byte_lda = lambda val : tf.train.Feature(
  #     bytes_list=tf.train.BytesList(value=[val]) )  
  compression_type = tf.python_io.TFRecordCompressionType.NONE
  # compression_type = tf.python_io.TFRecordCompressionType.ZLIB
  # compression_type = tf.python_io.TFRecordCompressionType.GZIP
  options_ = tf.python_io.TFRecordOptions(compression_type)
  cnt = 0
  with tf.python_io.TFRecordWriter(recordFile, options_) as writer:
    for exp in exampleList:
      cnt += 1

      seq_shape = exp['vec'].shape
      label = exp['label']
      flatten = np.ravel(exp['vec'])
      feature_map = {
          'seq': float_lda(flatten),
          'seq_shape': int64_lda(seq_shape),
          'label': int64_lda([label])
        }
      features_ = tf.train.Features(feature = feature_map)
      example = tf.train.Example(features = features_)
      writer.write(example.SerializeToString())

      if cnt%50==0:
        print('cnt %d' % cnt)
  print('--- buildRecord() end')
  return
# YJango：TensorFlow中层API Datasets+TFRecord的数据导入
# https://cloud.tencent.com/developer/article/1088751

# 多进程构造 tfexample
# 单进程写入 tfrecord
def mp_buildRecord(exampleList, recordFile):
  rd.shuffle(exampleList)
  TFexampleList = genTFexampleList(exampleList)

  evalSize = 8000
  # evalSize = 40
  trainSize = len(exampleList) - evalSize
  # 随机选测试样本
  eval_idx_list = rd.sample(range(len(exampleList)), evalSize)
  eval_idx_list.sort()

  eval_list = []
  train_list = []
  start = 0
  for idx in eval_idx_list:
    eval_list.append(TFexampleList[idx])
    # 剩下的即为训练样本
    for i in range(start, idx):
      train_list.append(TFexampleList[i])
    start = idx+1
  # 最后的区间
  for i in range(start, len(exampleList)):
    train_list.append(TFexampleList[i])

  train_record = recordFile+'.train'
  eval_record = recordFile+'.eval'

  compression_type = tf.python_io.TFRecordCompressionType.NONE
  options_ = tf.python_io.TFRecordOptions(compression_type)

  cnt = 0
  with tf.python_io.TFRecordWriter(train_record, options_) as writer:
    for TFexample in train_list:
      writer.write(TFexample.SerializeToString())
      cnt += 1
      if cnt%400==0:
        print('cnt %d' % cnt)
  print('--- train_record end')
  
  with tf.python_io.TFRecordWriter(eval_record, options_) as writer:
    for TFexample in eval_list:
      cnt += 1
      writer.write(TFexample.SerializeToString())
      if cnt%400==0:
        print('cnt %d' % cnt)
  print('--- eval_record end')
      
  print('--- mp_buildRecord() end')
  return

# 进程池构造 tfexample
def genTFexampleList(exampleList):
  resultList = []
  max_workers = os.cpu_count()  # ProcessPoolExecutor
  max_workers = max_workers + (max_workers>>1)
  with ProcessPoolExecutor(max_workers) as executor:
    for exp in exampleList:
      res = executor.submit(worker_TFexample, exp)
      resultList.append(res)
      # break
    executor.shutdown()
  print('--- worker_TFexample executor shutdown()')
  TFexampleList = [r.result() for r in resultList]
  return TFexampleList

# item in exampleList
def worker_TFexample(exp):
  float_lda = lambda val : tf.train.Feature(
      float_list=tf.train.FloatList(value=val) )
  int64_lda = lambda val : tf.train.Feature(
      int64_list=tf.train.Int64List(value=val) )
  seq_shape = exp['vec'].shape
  label = exp['label']
  flatten = np.ravel(exp['vec'])
  feature_map = {
      'seq': float_lda(flatten),
      'seq_shape': int64_lda(seq_shape),
      'label': int64_lda([label])
    }
  features_ = tf.train.Features(feature = feature_map)
  example = tf.train.Example(features = features_)
  return example

def tst_perVideoNpy():
  with open(jsonFile) as f:
    dat = json.load(f)
  dat = dat['fire_clips']
  for itr in dat:
    itr['video_root'] = videoNpyRoot
    # res = executor.map()
    perVideoNpy(itr)
    break
  return


if __name__ == '__main__':
  # tst_perVideoNpy()
  main()