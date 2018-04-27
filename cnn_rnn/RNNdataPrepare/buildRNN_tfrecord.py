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
  'invariant': 1,
  'weaking': 2
}
labelList = ['raging', 'invariant', 'weaking']
videoNpyRoot = r'D:\Lab408\cnn_rnn\77featureVectorNpy'

def main():
  recordFile = r''
  exampleList = genExampleList()
  buildRecord(exampleList, recordFile)

def genExampleList():
  with open('.json') as f:
    dat = json.load(f)
  dat = dat['fire_clips']

  resultList= []
  max_workers = 1
  max_workers = os.cpu_count()  # ProcessPoolExecutor
  with ProcessPoolExecutor(max_workers) as executor:
    for it in dat:
      it['video_root'] = videoNpyRoot
      # res = executor.map()
      res = executor.submit(perVideoNpy, it)
      resultList.append(res)
      
    executor.shutdown()
  print('------ executor shutdown()')
  exampleList = [] # {'vec': xxx, 'label': xxx}
  exampleList.append(res.result() for res in resultList)


def perVideoNpy(param_dict):
  video_root = it['video_root']
  vedio_dir = '%03d' % int(it['vedio_dir'])
  fire_begin_frame = it['fire_begin_frame']
  fire_biggest_frame = it['fire_biggest_frame']

  seqLen = 32
  step = 1

  # 火变大
  seqList = []
  border = fire_biggest_frame + 1
  for i in range(fire_begin_frame, border, step):
    end = i+seqLen
    if end > border: # 分片越界
      break
    # aSeq = mVideoFV[np.newaxis, i:end, :]
    aSeq = mVideoFV[i:end, :]
    seqList.append(aSeq)

  # fire_over_frame = it['fire_over_frame']
  # over_frame = it['over_frame']
  # formatted = '%03d' % int(vedio_dir)
  # srcFramesDir = pj(srcDir, formatted)

  exampleList = []
  
  return exampleList

# exampleList: [{'vec': xxx, 'label': xxx}, {}, ..]
# recordFile: /path/to/xxx.tfrecord
def buildRecord(exampleList, recordFile):
  rd.shuffle(exampleList)
  # val 必须是 list
  float_lda = lambda val : tf.train.Feature(
      int64_list=tf.train.FloatList(value=val) )
  int64_lda = lambda val : tf.train.Feature(
      int64_list=tf.train.Int64List(value=val) )
  # byte_lda = lambda val : tf.train.Feature(
  #     bytes_list=tf.train.BytesList(value=[val]) )  
  compression_type = tf.python_io.TFRecordCompressionType.NONE
  # compression_type = tf.python_io.TFRecordCompressionType.ZLIB
  # compression_type = tf.python_io.TFRecordCompressionType.GZIP
  options_ = tf.python_io.TFRecordOptions(compression_type)
  with tf.python_io.TFRecordWriter(recordFile, options_) as writer:
    for exp in exampleList:
      seq_shape = exp['vec'].shape
      label = exp['label']
      flatten = np.ravel(exp['vec'])
      feature_map = {
          'seq': float_lda(flatten),
          'seq_shape': int64_lda(seq_shape)
          'label': int64_lda([label]),
        }
      features_ = tf.train.Features(feature = feature_map)
      example = tf.train.Example(features = features_)
      writer.write(example.SerializeToString())
  return
# YJango：TensorFlow中层API Datasets+TFRecord的数据导入
# https://cloud.tencent.com/developer/article/1088751

if __name__ == '__main__':
  main()