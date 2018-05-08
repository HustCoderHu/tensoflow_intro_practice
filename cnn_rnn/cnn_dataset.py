import os
import os.path as path
from os.path import join as pj
import random as rd
import json

import tensorflow as tf

class MyDataset():
  # evalSet: 各种类别里作为测试的文件集合
  # 比如 [4, 5, 6]
  def __init__(self, videoRoot, labeljson, evalSet, resize=None):
    if not path.exists(videoRoot):
      raise SystemError('directory not exist: {}'.format(
        videoRoot ))

    self.videoRoot = videoRoot
    self.labeljson = labeljson
    self.evalSet = evalSet
    self.resize=resize

    self.label_id = {'fire': 0, 'fireless': 1}

    # labels = os.listdir(videoRoot)
    # labels.sort()
    # for idx, name in enumerate(labels):
    #   self.dict_name_id[name] = idx
    return

  def setTrainParams(self, batchsz, prefetch=None, repeat=True):
    self.trainBatchsz = batchsz
    self.trainPrefetch = prefetch
    self.trainRepeat = repeat
    return
  def setEvalParams(self, batchsz, prefetch=None, repeat=True):
    self.evalBatchsz = batchsz
    self.evalPrefetch = prefetch
    self.evalRepeat = repeat
    return

  def makeTrainIter(self):
    with open(self.labeljson) as f:
      dat = json.load(f)
    dat = dat['fire_clips']

    includeVideoLabel = []
    for it in dat:
      # 排除测试用的视频帧
      if int(it['video_dir']) in self.evalSet:
        continue
      includeVideoLabel.append(it)

    print('-- train')
    img_list, label_list = self.genFrames_PathLabels(includeVideoLabel)
    print('images num: {}'.format(len(img_list)))

    t_imglist = tf.constant(img_list, dtype=tf.string)
    t_lbllist = tf.constant(label_list, dtype=tf.int32)
    dataset = tf.data.Dataset().from_tensor_slices((t_imglist, t_lbllist))
    if self.resize == None:
      dset = dataset.map(self._mapfn, num_parallel_calls=os.cpu_count())
    else :
      dset = dataset.map(self._mapfn_resize, num_parallel_calls=os.cpu_count())
    dset = dset.shuffle(len(img_list), reshuffle_each_iteration=False)
    
    if self.trainPrefetch == None:
      dset = dset.cache()
    if self.trainRepeat:
      dset = dset.repeat()
    dset = dset.batch(self.trainBatchsz)
    if self.trainPrefetch != None:
      dset = dset.prefetch(self.trainPrefetch)
    # img, label, filename
    # shape (batch_sz, )
    self.output_types = dset.output_types
    return dset.make_one_shot_iterator()
    # return dset.make_one_shot_iterator().get_next()
    # _iter = dset.make_one_shot_iterator()
    # next_one = _iter.get_next()
    # return next_one
  # end makeIter()

  def makeEvalIter(self):
    with open(self.labeljson) as f:
      dat = json.load(f)
    dat = dat['fire_clips']

    includeVideoLabel = []
    for it in dat:
      # 测试用的视频帧
      if int(it['video_dir']) in self.evalSet:
        includeVideoLabel.append(it)
    print('-- eval')
    img_list, label_list = self.genFrames_PathLabels(includeVideoLabel)

    print('images num: {}'.format(len(img_list)))
    t_imglist = tf.constant(img_list, dtype=tf.string)
    t_lbllist = tf.constant(label_list, dtype=tf.int32)
    dataset = tf.data.Dataset().from_tensor_slices((t_imglist, t_lbllist))
    if self.resize == None:
      dset = dataset.map(self._mapfn, num_parallel_calls=os.cpu_count())
    else :
      dset = dataset.map(self._mapfn_resize, num_parallel_calls=os.cpu_count())
    dset = dset.shuffle(len(img_list), reshuffle_each_iteration=False)
    
    if self.evalPrefetch == None:
      dset = dset.cache()
    if self.evalRepeat:
      dset = dset.repeat()
    dset = dset.batch(self.evalBatchsz)
    if self.evalPrefetch != None:
      dset = dset.prefetch(self.evalPrefetch)
    # img, label, filename
    # shape (batch_sz, )
    self.output_types = dset.output_types
    return dset.make_one_shot_iterator()

  def genFrames_PathLabels(self, includeVideoLabel):
    allPath = []
    for param_dict in includeVideoLabel:
      # 逐个处理每个视频
      video_idx = int(param_dict['video_dir'])
      param_dict['frames_dir'] = pj(self.videoRoot, '{:0>3d}'.format(video_idx))
      exampleList = self.handelFrames(param_dict)
      allPath.extend(exampleList)

    # 打乱所有
    rd.shuffle(allPath)

    pathList = []
    labelList = []
    for it in allPath:
      pathList.append(it['imgPath'])
      labelList.append(it['label'])
    return pathList, labelList
  
  # return [] # item {'imgPath': xxx, 'label': xxx}
  def handelFrames(self, param_dict):
    fire_begin_frame = int(param_dict['fire_begin_frame'])
    # fire_biggest_frame = int(param_dict['fire_biggest_frame'])
    fire_over_frame = int(param_dict['fire_over_frame'])
    over_frame = int(param_dict['over_frame'])

    frames_dir = param_dict['frames_dir']
    if not path.exists(frames_dir):
      return []

    flist = os.listdir(frames_dir)
    print('len(flist): %d' % len(flist))
    exampleList = []
    for f in flist:
      if path.splitext(f)[1] != '.jpg':
        continue
      # 每帧对应标签
      frameIdx = int(path.splitext(f)[0])
      labelid = self.label_id['fireless']
      if fire_begin_frame <= frameIdx and frameIdx <= fire_over_frame:
        labelid = self.label_id['fire']
      
      example = {'imgPath': pj(frames_dir, f), 'label': labelid}
      exampleList.append(example)
    return exampleList

  def _mapfn(self, filename, label):
    with tf.device('/cpu:0'):
      # <https://www.tensorflow.org/performance/performance_guide>
      img_raw = tf.read_file(filename)
      decoded = tf.image.decode_jpeg(img_raw)
      # decoded = tf.cast(decoded, tf.float32)
    return decoded, label, filename
  
  def _mapfn_resize(self, filename, label):
    with tf.device('/cpu:0'):
      # <https://www.tensorflow.org/performance/performance_guide>
      img_raw = tf.read_file(filename)
      decoded = tf.image.decode_jpeg(img_raw)
      _h, _w = self.resize
      resized = tf.image.resize_images(decoded, [_h, _w])
    return resized, label, filename

def tst():
  videoRoot = r'D:\Lab408\cnn_rnn\src_dir'
  labeljson = r'D:\Lab408\cnn_rnn\label.json'
  evalSet = [4, 5, 6]

  dataset = MyDataset(videoRoot, labeljson, evalSet, resize=(250, 250))
  dataset.setTrainParams(50)
  dataset.setEvalParams(200)
  trainIter = dataset.makeTrainIter()
  evalIter = dataset.makeEvalIter()
  return 

  resized, filename = dataset(4, prefetch_batch=None)
  sess_conf = tf.ConfigProto()
  sess_conf.gpu_options.allow_growth = True
  sess_conf.gpu_options.per_process_gpu_memory_fraction = 0.9
  with tf.Session(config= sess_conf) as sess:
    while True:
      try:
        fname = sess.run(filename)
        print(type(fname))
        print(fname.shape)
        for i in range(fname.shape[0]):
          print(fname[i])
      except tf.errors.OutOfRangeError as identifier:
        print(identifier.message)
        break

if __name__ == '__main__':
  tst()