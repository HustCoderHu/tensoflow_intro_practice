import numpy as np
import random as rd

import os
import os.path as path
from os.path import join as pjoin
import tensorflow as tf

class MyDataset():
  def __init__(self, train_tfrecord, eval_tfrecord):
    if not path.exists(train_tfrecord):
      raise SystemError('file not exist: {}'.format(
        train_tfrecord ))
    if not path.exists(eval_tfrecord):
      raise SystemError('file not exist: {}'.format(
        eval_tfrecord ))

    self.train_tfrecord = train_tfrecord
    self.eval_tfrecord = eval_tfrecord
    self.output_types = None

    self.dict_name_id = {}
    labels = os.listdir(train_tfrecord)
    for idx, name in enumerate(labels):
      print(name + ": " + idx)
      self.dict_name_id[name] = idx
  
  def train_iter(self, batch_sz = 30, prefetch_batch=None):
    print('train ----------')
    return self.get_iter(self.train_tfrecord, batch_sz, prefetch_batch)
  
  def eval_iter(self, batch_sz = 50, prefetch_batch=None):
    print('eval ===========')
    return self.get_iter(self.eval_tfrecord, batch_sz, prefetch_batch)
    # return self.get_iter(self.eval_tfrecord, batch_sz, prefetch_batch,
        # repeat=False)


  def get_iter(self, tfrecord, batch_sz, prefetch_batch=None, repeat=True):
    dataset = tf.data.TFRecordDataset(tfrecord)
    dset = dataset.map(_parse_function, num_parallel_calls=os.cpu_count())
    # dset = dset.shuffle(len(npy_list), reshuffle_each_iteration=False)
    
    if prefetch_batch == None:
      dset = dset.cache()
    if repeat:
      dset = dset.repeat()
    dset = dset.batch(batch_sz)
    if prefetch_batch != None:
      dset = dset.prefetch(prefetch_batch)
    # img, label, filename
    # shape (batch_sz, )
    self.output_types = dset.output_types
    return dset.make_one_shot_iterator()
    # return dset.make_one_shot_iterator().get_next()
    # _iter = dset.make_one_shot_iterator()
    # next_one = _iter.get_next()
    # return next_one
  
  # end get_iter()

def _parse_function(example_proto):
  features = {"seq": tf.FixedLenFeature((), tf.float32, default_value=0.0),
              "seq_shape": tf.FixedLenFeature((), tf.int64, default_value=0),
              "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["seq"], parsed_features["seq_shape"], parsed_features["label"]