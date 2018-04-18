import numpy as np
import random as rd

import os
import os.path as path
from os.path import join as pjoin
import tensorflow as tf


class MyDataset():
  def __init__(self, train_dir, eval_dir):
    if not path.exists(train_dir):
      raise SystemError('directory not exist: {}'.format(
        train_dir ))
    if not path.exists(eval_dir):
      raise SystemError('directory not exist: {}'.format(
        eval_dir ))

    self.train_dir = train_dir
    self.eval_dir = eval_dir

    self.dict_name_id = {}
    labels = os.listdir(train_dir)
    for idx, name in enumerate(labels):
      print(name + ": " + idx)
      self.dict_name_id[name] = idx
  
  def train(self, batch_sz = 10, prefetch_batch=None):
    print('train ----------')
    return self._get(self.train_dir, batch_sz, prefetch_batch)
  
  def eval(self, batch_sz = 10, prefetch_batch=None):
    print('eval ===========')
    return self._get(self.eval_dir, batch_sz, prefetch_batch)
    # return self._get(self.eval_dir, batch_sz, prefetch_batch,
        # repeat=False)


  def _get(self, subset_dir, batch_sz, prefetch_batch=None, repeat=True):
    _labels = os.listdir(subset_dir)
    labels = []
    for lb in _labels:
      if path.isdir(pjoin(subset_dir, lb)):
        labels.append(lb)

    npy_list = []
    label_list = []

    aExample = {'npy': 0, 'label': 'fire'}
    example_list = []#[aExample]

    for class_name in labels:
      per_class_dir = pjoin(subset_dir, class_name)
      per_class_npylist = os.listdir(per_class_dir)
      print('{}: total {}'.format(class_name, len(per_class_npylist)))
      _list = [pjoin(per_class_dir, i) for i in per_class_npylist]
      for _npy in _list:
        example_list.append({'npy': _npy, 
            'label': self.dict_name_id[class_name]})
    # shuffle and split to npy set and label set
    rd.shuffle(example_list)
    for _dict in example_list:
      npy_list.append(_dict['npy'])
      label_list.append(_dict['label'])

    print('npys num: {}'.format(len(npy_list)))
    t_npylist = tf.constant(npy_list)
    t_lbllist = tf.constant(label_list, dtype=tf.int32)
    dataset = tf.data.Dataset().from_tensor_slices((t_npylist, t_lbllist))
    dset = dataset.map(self._mapfn, num_parallel_calls=os.cpu_count())
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
    return dset.make_one_shot_iterator()
    # return dset.make_one_shot_iterator().get_next()
    # _iter = dset.make_one_shot_iterator()
    # next_one = _iter.get_next()
    # return next_one
  
  # end _get()

  def _mapfn(self, filename, label):
    with tf.device('/cpu:0'):
      dat = np.load(filename)
    return dat, label, filename