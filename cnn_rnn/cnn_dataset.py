import os
import os.path as path
from os.path import join as pj
import random as rd
import json
from pprint import pprint

import tensorflow as tf

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

    self.labeldict = MyDataset.decodeLabel(labeljson)
    # pprint(self.labeldict)
    self.categoryInfo = MyDataset.info(self.labeldict)
    pprint(self.categoryInfo)
    self.bboxes = tf.constant([[[0., 0., 1., 1.]]])  

    # labels = os.listdir(videoRoot)
    # labels.sort()
    # for idx, name in enumerate(labels):
    #   self.dict_name_id[name] = idx
    return
  # 解析标签文件
  # retval {video_idx (int): 范围}
  # 范围: {'fire': A, 'fireless': B}
  # A, B: [[0, x], [y, z], [z, .], ...]
  @staticmethod
  def decodeLabel(labeljson):
    with open(labeljson) as f:
      dat = json.load(f)
    dat = dat['fire_clips'] # []

    labeldict = {}
    for it in dat:
      video_idx = int(it['video_dir'])
      begin_frame = int(it['begin_frame'])
      fire_begin_frame = int(it['fire_begin_frame'])
      # fire_biggest_frame = int(it['fire_biggest_frame'])
      fire_over_frame = int(it['fire_over_frame'])
      over_frame = int(it['over_frame'])

      spans = None
      if video_idx in labeldict.keys():
        spans = labeldict[video_idx]
      else:
        spans = {'fire': [], 'fireless': []}
        labeldict[video_idx] = spans
      if begin_frame <= fire_begin_frame-1:
        spans['fireless'].append(tuple([begin_frame, fire_begin_frame-1]))
      if fire_over_frame+1 <= over_frame:
        spans['fireless'].append(tuple([fire_over_frame+1, over_frame]))
      if fire_begin_frame < fire_over_frame:
        spans['fire'].append(tuple([fire_begin_frame, fire_over_frame]))
    return labeldict

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

    t_imglist = tf.constant(img_list, dtype=tf.string)
    t_lbllist = tf.constant(label_list, dtype=tf.int32)
    dataset = tf.data.Dataset().from_tensor_slices((t_imglist, t_lbllist))
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
    with open(self.labeljson) as f:
      dat = json.load(f)
    dat = dat['fire_clips']

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
      dset = dataset.map(self._mapfn_resize, num_parallel_calls=os.cpu_count())
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
    for video_idx, spans in part_labeldict.items():
      spans = part_labeldict[video_idx]
      spans['frames_dir'] = pj(self.videoRoot, '{:0>3d}'.format(video_idx))
      exampleList = MyDataset.handelFrames(spans)
      allPath.extend(exampleList)

    # 打乱所有
    rd.shuffle(allPath)

    pathList = []
    labelList = []
    for it in allPath:
      pathList.append(it['imgPath'])
      labelList.append(it['label'])
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
    # print('len(flist) jpg: %d' % len(flist))
    exampleList = []
    for f in flist:
      # 每帧对应标签
      frameIdx = int(path.splitext(f)[0])
      labelid = MyDataset.label_id['fireless']
      if MyDataset.judgeLabel(spans, frameIdx):
        labelid = MyDataset.label_id['fire']
      example = {'imgPath': pj(frames_dir, f), 'label': labelid}
      exampleList.append(example)

    return exampleList

  @staticmethod
  def judgeLabel(spans, frameIdx):
    spansFire = spans['fire']
    fire = False
    for span in spansFire:
      if span[0] <= frameIdx and frameIdx <= span[1]:
        fire = True
        break
    return fire
    # spansFireless = spans['fireless']


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
      begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
          tf.shape(decoded), self.bboxes, min_object_covered=0.85,
          aspect_ratio_range=self.aspect_ratio_range, max_attempts=10)
      distorted_image = tf.slice(decoded, begin, size)
      image_random_saturation = tf.image.random_saturation(distorted_image,
          0.1, 0.9)
      _h, _w = self.resize
      # resized = tf.image.resize_image_with_crop_or_pad(decoded, _h, _w)
      resized = tf.image.resize_images(image_random_saturation, [_h, _w])
      flipped = tf.image.random_flip_left_right(resized)
    return flipped, label, filename
  
  # 统计每个视频有火和无火的帧数
  # and all
  # labeldict: returned by decodeLabel()
  @staticmethod
  def info(labeldict):
    categoryInfo = {}
    totalFire = 0
    totalFireless = 0
    for video_idx, spans in labeldict.items():
      # 有火帧数
      spanFire = spans['fire']
      nFire = 0
      for span in spanFire:
        nFire += (span[1]-span[0]+1)
      # 无火帧数
      spanFireless = spans['fireless']
      nFireless = 0
      for span in spanFireless:
        nFireless += (span[1]-span[0]+1)
      # 累计
      totalFire += nFire
      totalFireless += nFireless
      categoryInfo[video_idx] = {'fire': nFire, 'fireless': nFireless}
      # print('video {}:'.format(video_idx))
      # pprint({'fire': nFire, 'totalFireless': nFireless})
    
    categoryInfo['total:'] = {'totalfire': totalFire, 'totalFireless': totalFireless}
    # print('total:')
    # print('fire: {}'.format(totalFire))
    # print('fireless: {}'.format(totalFireless))
    return categoryInfo
  # 挑选作为 测试集的视频帧
  # categoryInfo: returned by info()
  @staticmethod
  def selectEvalSet(categoryInfo):
    return []
def tst():
  videoRoot = r'D:\Lab408\cnn_rnn\src_dir'
  labeljson = r'D:\Lab408\cnn_rnn\label.json'
  videoRoot = r'/home/hzx/all_data/'
  labeljson = r'/home/hzx/all_data/label.json'

  evalSet = [47, 48, 49, 50, 27, 33, 21, 32]

  dataset = MyDataset(videoRoot, labeljson, evalSet, resize=(250, 250))
  dataset.setTrainParams(50, prefetch=16)
  dataset.setEvalParams(200, prefetch=5)
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