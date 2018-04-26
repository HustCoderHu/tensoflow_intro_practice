import os
import os.path as path
from os.path import join as pj
import tensorflow as tf

class FD():
  def __init__(self, srcDir, videoId=''):
    self.srcDir = srcDir
    self.videoId = videoId
    self.resize = (250, 250)
  
  def __call__(self, batch_sz, prefetch_batch=None):
    video_dir = self.videoId
    # video_dir = '%03d' % self.videoId
    perFramesDir = pj(self.srcDir, video_dir)
    print('--- perFramesDir: ' + perFramesDir)
    jpgBasenameList = os.listdir(perFramesDir)
    jpgBasenameList.sort()
    print('--- len(jpgBasenameList): %d' % len(jpgBasenameList))
    print('--- jpgBasenameList[0:5]: %s' % jpgBasenameList[0:5])
    # jpgBasenameList = jpgBasenameList[0:6]

    img_list = []
    for f in jpgBasenameList:
      abspath = pj(perFramesDir, f)
      if path.isfile(abspath) and path.splitext(f)[1]=='.jpg':
        img_list.append(abspath)

    print('video %s, jpg num: %d' % (self.videoId, len(img_list)) )

    dataset = tf.data.Dataset().from_tensor_slices(img_list)
    dset = dataset.map(self._mapfn_resize, num_parallel_calls=os.cpu_count())
    dset = dset.batch(batch_sz)
    if prefetch_batch != None:
      dset = dset.prefetch(prefetch_batch)

    return dset.make_one_shot_iterator().get_next()
  
  def _mapfn_resize(self, filename):
    with tf.device('/cpu:0'):
      # <https://www.tensorflow.org/performance/performance_guide>
      img_raw = tf.read_file(filename)
      decoded = tf.image.decode_jpeg(img_raw)
      _h, _w = self.resize
      resized = tf.image.resize_images(decoded, [_h, _w])
      # print('--- inputx shape: %s' % resized.shape)
      inputx = tf.reshape(resized, [_h, _w, 3])
    return inputx, filename

def tst():
  srcDir = r'D:\Lab408\cnn_rnn\src_dir'
  fname = r'D:\Lab408\cnn_rnn\src_dir\011\000005.jpg'
  print(path.basename(fname)) # 000005.jpg
  return

  dataset = FD(srcDir, '011')
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
  