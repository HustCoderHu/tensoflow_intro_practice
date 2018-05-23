import os
import os.path as path
from os.path import join as pj
import random as rd
import json
from pprint import pprint
import cv2 as cv

import tensorflow as tf
from tensorflow.python.ops import random_ops

import dataPreProcess as preProcess

class MyDataset():
  label_id = {'fire': 0, 'fireless': 1}
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
    _h, _w = resize
    self.aspect_ratio_range = [0.85, 1.15]
    # self.aspect_ratio_range = float(_w)/float(_h)

    self.labeldict = preProcess.decodeLabel(labeljson)
    # pprint(self.labeldict)
    self.categoryInfo = preProcess.info(self.labeldict)
    pprint(self.categoryInfo)
    self.bboxes = tf.constant([[[0., 0., 1., 1.]]])  

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
    part_labeldict = {}
    for video_idx, spans in self.labeldict.items():
      # 排除测试用的视频帧
      if video_idx in self.evalSet:
        continue
      part_labeldict[video_idx] = spans
    print('-- train')
    # keys = list(part_labeldict.keys())
    # keys.sort()
    # print(keys)
    # print(len(part_labeldict))
    # left = []
    # for k in range(1, 79):
    #   if k in keys:
    #     continue
    #   left.append(k)
    # print(left)
    # pprint(includeVideoLabel)
    img_list, label_list = self.genFrames_PathLabels(part_labeldict)
    print('images num: {}'.format(len(img_list)))

    # t_imglist = tf.constant(img_list, dtype=tf.string)
    # t_lbllist = tf.constant(label_list, dtype=tf.int32)
    # dataset = tf.data.Dataset().from_tensor_slices((t_imglist, t_lbllist))
    dataset = tf.data.Dataset().from_tensor_slices((img_list, label_list))
    if self.resize == None:
      dset = dataset.map(self._mapfn, num_parallel_calls=os.cpu_count())
    else :
      dset = dataset.map(self._mapfn_resize, num_parallel_calls=os.cpu_count())
    # dset = dset.shuffle(len(img_list), reshuffle_each_iteration=False)
    
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
    part_labeldict = {}
    for video_idx, spans in self.labeldict.items():
      # 测试用的视频帧
      if video_idx in self.evalSet:
        part_labeldict[video_idx] = spans
    print('-- eval')
    # print(len(part_labeldict))
    # pprint(includeVideoLabel)
    img_list, label_list = self.genFrames_PathLabels(part_labeldict)
    print('images num: {}'.format(len(img_list)))

    t_imglist = tf.constant(img_list, dtype=tf.string)
    t_lbllist = tf.constant(label_list, dtype=tf.int32)
    dataset = tf.data.Dataset().from_tensor_slices((t_imglist, t_lbllist))
    if self.resize == None:
      dset = dataset.map(self._mapfn, num_parallel_calls=os.cpu_count())
    else :
      dset = dataset.map(self._mapfn_resize_noDataAug, num_parallel_calls=os.cpu_count())
    # dset = dset.shuffle(len(img_list), reshuffle_each_iteration=False)
    
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
  
  # 逐个处理每个视频
  def genFrames_PathLabels(self, part_labeldict):
    allPath = []
    # export2file_list = []
    for video_idx, spans in part_labeldict.items():
      spans = part_labeldict[video_idx]
      spans['frames_dir'] = pj(self.videoRoot, '{:0>3d}'.format(video_idx))
      exampleList = MyDataset.handelFrames(spans)

      # export2file_list.append({'video_idx': video_idx, 'frame':exampleList})
      allPath.extend(exampleList)

    # handelFrames = pj(self.videoRoot, 'handelFrames')
    # if not path.exists(handelFrames):
    #   os.mkdir(handelFrames)
    # for it in export2file_list:
    #   vidx = it['video_idx']
    #   with open(pj(handelFrames, '{}.txt'.format(vidx)), 'w') as f:
    #     json.dump(it, f)
    # 打乱所有
    rd.shuffle(allPath)

    pathList = []
    labelList = []
    for it in allPath:
      pathList.append(it['imgPath'])
      labelList.append(it['label'])
    
    # 两类数量统计
    labeldict = {-1: 0, 0: 0, 1: 0}
    for lid in labelList:
      labeldict[lid] += 1
    pprint(labeldict)
    return pathList, labelList
  
  # 每个视频所有帧构成 list
  # spans: {'frames_dir': xx, fire': A, 'fireless': B}
  # A, B: [[0, x], [y, z], [z, .], ...]
  # return [] # item {'imgPath': xxx, 'label': xxx}
  @staticmethod
  def handelFrames(spans):
    frames_dir = spans['frames_dir']
    if not path.exists(frames_dir):
      return []
    
    flist = os.listdir(frames_dir)
    flist = [f for f in flist if path.splitext(f)[1] == '.jpg']
    flist.sort()
    # print('len(flist) jpg: %d' % len(flist))
    exampleList = []
    for f in flist:
      # 每帧对应标签
      frameIdx = int(path.splitext(f)[0])
      labelid = preProcess.judgeLabel(spans, frameIdx)
      if labelid < 0:
        continue
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
      # 随机的截取图片中一个块
      # begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
      #     tf.shape(decoded), self.bboxes, min_object_covered=0.85,
      #     aspect_ratio_range=self.aspect_ratio_range, max_attempts=10)
      # distorted_image = tf.slice(decoded, begin, size)
      # print('distorted_image')
      # print(distorted_image.dtype) # uint8
      # random_ops.random_uniform([], lower, upper, seed=seed)
      image_random_saturation = tf.image.random_saturation(decoded,
          0.85, 1.2)
      # print('image_random_saturation')
      # print(image_random_saturation.dtype) # uint8
      _h, _w = self.resize
      # resized = tf.image.resize_image_with_crop_or_pad(decoded, _h, _w)
      flipped = tf.image.random_flip_left_right(image_random_saturation)
      # method=tf.image.ResizeMethod.AREA
      resized = tf.image.resize_images(flipped, [_h, _w], tf.image.ResizeMethod.AREA)
      # ResizeMethod.AREA 缩小
      # resized float32
    return resized, label, filename
  
  # 无调整，eval 用
  def _mapfn_resize_noDataAug(self, filename, label):
    with tf.device('/cpu:0'):
      # <https://www.tensorflow.org/performance/performance_guide>
      img_raw = tf.read_file(filename)
      decoded = tf.image.decode_jpeg(img_raw)
      _h, _w = self.resize
      resized = tf.image.resize_images(decoded, [_h, _w], tf.image.ResizeMethod.AREA)
    return resized, label, filename

  
  # 挑选作为 测试集的视频帧
  # categoryInfo: returned by info()
  @staticmethod
  def selectEvalSet(categoryInfo):
    return []

def tst():
  videoRoot = r'D:\Lab408\cnn_rnn\src_dir'
  labeljson = r'D:\Lab408\cnn_rnn\label.json'

  videoRoot = r'W:\hzx\all_data'
  # videoRoot = r'/home/hzx/all_data/'
  # labeljson = r'/home/hzx/all_data/label.json'

  evalSet = [47, 48, 49, 50, 27, 33, 21, 32]

  dataset = MyDataset(videoRoot, labeljson, evalSet, resize=(240, 320))
  dataset.setTrainParams(50, prefetch=10)
  dataset.setEvalParams(200, prefetch=5)
  # trainIter = dataset.makeTrainIter()
  # evalIter = dataset.makeEvalIter()
  # inputx, labels, filename = trainIter.get_next()
  # print(inputx.dtype)
  # print(inputx.shape)
  return

  resized, filename = dataset(4, prefetch_batch=None)
  sess_conf = tf.ConfigProto()
  sess_conf.gpu_options.allow_growth = True
  # sess_conf.gpu_options.per_process_gpu_memory_fraction = 0.9
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

def tst_adjust_saturation():
  filename = r'D:\Lab408\cnn_rnn\src_dir\068\000324.jpg'
  img_raw = tf.read_file(filename)
  t_decoded = tf.image.decode_jpeg(img_raw)
  _h = 240
  _w = 320
  t_resized = tf.image.resize_images(t_decoded, (_h, _w))
  t_resized = tf.image.convert_image_dtype(t_resized, tf.uint8)
  t_flipped = tf.image.flip_left_right(t_decoded)
  t_image_saturation = tf.image.adjust_saturation(t_decoded,
          1.4)

  sess_conf = tf.ConfigProto()
  sess_conf.gpu_options.allow_growth = True
  with tf.Session(config= sess_conf) as sess:
    decoded, resized, flipped, image_saturation = sess.run(
      [t_decoded, t_resized, t_flipped, t_image_saturation]
    )

  # https://stackoverflow.com/questions/38890525/visualize-with-opencv-image-read-by-tensorflow
  cvtColor = cv.cvtColor(decoded, cv.COLOR_RGB2BGR)
  cv.imshow('decoded', cvtColor)

  cvtColor = cv.cvtColor(resized, cv.COLOR_RGB2BGR)
  cv.imshow('resized', cvtColor)

  cvtColor = cv.cvtColor(flipped, cv.COLOR_RGB2BGR)
  cv.imshow("flipped", cvtColor)
  
  cvtColor = cv.cvtColor(image_saturation, cv.COLOR_RGB2BGR)
  cv.imshow("image_saturation", cvtColor)

  cv.waitKey()

if __name__ == '__main__':
  # size = 16*240*320*3*4 / 1024
  # print(size)
  # l = list(range(1,10))
  # rd.shuffle(l)
  # print(l)
  # print(162563+14375)
  tst()
  # tst_adjust_saturation()