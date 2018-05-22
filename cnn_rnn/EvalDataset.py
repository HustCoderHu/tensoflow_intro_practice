import os
import os.path as path
from os.path import join as pj
from pprint import pprint

import tensorflow as tf

import dataPreProcess as preProcess

class EvalDataset():
  def __init__(self, videoRoot, videoIdx, labeljson, resize):
    self.videoRoot = videoRoot
    labeldict = preProcess.decodeLabel(labeljson)
    # pprint(labeldict)
    spans = labeldict[videoIdx]
    spans['frames_dir'] = pj(videoRoot, '{:0>3d}'.format(videoIdx))
    self.spans = spans

    self.resize = resize
    # self.categoryInfo = preProcess.info(self.labeldict)
    self.output_types = None    
    return

  # 每个视频所有帧构成 list
  # spans: {'frames_dir': xx, fire': A, 'fireless': B}
  # A, B: [[0, x], [y, z], [z, .], ...]
  # return [] # item {'imgPath': xxx, 'label': xxx}
  @staticmethod
  def handleFrames(spans):
    frames_dir = spans['frames_dir']
    if not path.exists(frames_dir):
      return []
    
    flist = os.listdir(frames_dir)
    flist = [f for f in flist if path.splitext(f)[1] == '.jpg']
    flist.sort()
    # print('len(flist) jpg: %d' % len(flist))
    # exampleList = []
    pathList = []
    labelList = []

    for f in flist:
      # 每帧对应标签
      frameIdx = int(path.splitext(f)[0])
      labelid = preProcess.judgeLabel(spans, frameIdx)
      if labelid < 0:
        continue
      pathList.append(pj(frames_dir, f))
      labelList.append(labelid)
    return pathList, labelList
  
  def setEvalParams(self, batchsz, prefetch=None, cacheFile=None):
    self.batchsz = batchsz
    self.prefetch = prefetch
    self.cacheFile = cacheFile
    return

  def makeIter(self):
    pathList, labelList = EvalDataset.handleFrames(self.spans)

    t_imglist = tf.constant(pathList, dtype=tf.string)
    t_lbllist = tf.constant(labelList, dtype=tf.int32)
    dataset = tf.data.Dataset().from_tensor_slices((t_imglist, t_lbllist))
    dset = dataset.map(self._mapfn_resize_noDataAug, num_parallel_calls=os.cpu_count())
  
    if self.cacheFile != None:
      if self.cacheFile == 'mem':
        dset = dset.cache()
      else :
        dset = dset.cache(cacheFile)
    dset = dset.batch(self.batchsz)
    if self.prefetch != None:
        dset = dset.prefetch(self.prefetch)
    self.output_types = dset.output_types
    return dset.make_one_shot_iterator()

  def _mapfn_resize_noDataAug(self, filename, labels):
    with tf.device('/cpu:0'):
      # <https://www.tensorflow.org/performance/performance_guide>
      img_raw = tf.read_file(filename)
      decoded = tf.image.decode_jpeg(img_raw)
      _h, _w = self.resize
      resized = tf.image.resize_images(decoded, [_h, _w], tf.image.ResizeMethod.AREA)
    return resized, labels, filename

if __name__ == '__main__':
  labeljson = r'D:\Lab408\cnn_rnn\label.json'
  dset = EvalDataset('', 1, labeljson, resize=(240, 320))