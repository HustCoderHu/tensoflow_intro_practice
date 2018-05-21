import time
import os
import os.path as path

import tensorflow as tf

import dataPreProcess as preProcess
import cnn

def main():
  log_dir = r'/home/hzx/fireDetect-hzx/log20180517/train_eval_log'
  ckpt_dir = path.join(log_dir, 'ckpts')

  videoRoot = r'/home/hzx/all_data/'
  labeljson = r'/home/hzx/all_data/label.json'
  # evalSet = [47, 48, 49, 50, 27, 33, 21, 32]
  evalSet = [47, 48, 49, 51, 52, 59, 61, 62, 63, 65]
  # 47: {'fire': 1601, 'fireless': 57},
#  48: {'fire': 3748, 'fireless': 98},
#  49: {'fire': 3714, 'fireless': 40},
#   51: {'fire': 4120, 'fireless': 21},
#  52: {'fire': 4451, 'fireless': 45},
#   59: {'fire': 6911, 'fireless': 70},
#    61: {'fire': 1298, 'fireless': 0},
#  62: {'fire': 3275, 'fireless': 0},
#  63: {'fire': 5055, 'fireless': 0},
#   65: {'fire': 6913, 'fireless': 64},
  if not path.exists(log_dir):
    os.mkdir(log_dir)
  if not path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
  
  batchsz = 100

  inputx = tf.placeholder(tf.uint8, (None, 240, 320, 3))

  # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ build graph \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  model = cnn.CNN('NCHW')
  # model = smodel.Simplest('NHWC')
  logits = model(inputx, castFromUint8=True)
  
  with tf.name_scope('prediction'):
    t_pred_vec = tf.nn.softmax(logits) # (batch, n_classes)
  # with tf.name_scope('cross_entropy'):
    # loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
  # with tf.name_scope('accuracy'):
  #   acc_vec = tf.equal(labels, t_pred_vec)
  #   acc = tf.reduce_mean(tf.cast(acc_vec, tf.float32))
    
  # with tf.name_scope('optimizer'):
  #   optimizer = tf.train.AdamOptimizer(1e-4)
  #   train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

  # ||||||||||||||||||||||||||||||  hooks ||||||||||||||||||||||||||||||
  # >>>  logging
  tf.logging.set_verbosity(tf.logging.INFO)
  # global_step = tf.train.get_or_create_global_step()
  # tf.identity(global_step, 'g_step')
  # tf.identity(loss, 'cross_entropy')
  # tf.identity(acc, 'accuracy')
  # tensor_lr = optimizer._lr_t

  # ////////////////////////////// session config //////////////////////////////
  sess_conf = tf.ConfigProto()
  sess_conf.gpu_options.allow_growth = True
  # sess_conf.gpu_options.per_process_gpu_memory_fraction = 0.9
 
  # ------------------------------  start  ------------------------------
  cap = cv.VideoCapture(videoPath)
  frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
  fps = cap.get(cv.CAP_PROP_FPS) # float
  print('frame_count: {}'.format(frame_count))
  print('fps: {}'.format(fps))

  height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
  width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
  prePolicy = FramePreprocessPolicy((height, width), (240, 320))

  log_dict = {} # item frameIdx:{'pred': label_id, 'loss': int}
  # item  frameIdx:{'fire': confidence, 'firelss': 1-confidence}
  frameBase = 0
  with tf.Session(config= sess_conf) as sess:
    while(cap.isOpened):
      # 拼成 batch 处理
      batchFrame, frameOffset = stackFrames(cap, prePolicy, batchsz)
      batchConfidence = sess.run(t_pred_vec, feed_dict={inputx:batchFrame})

      for idx in range(batchFrame):
        frameConf = batchConfidence[idx]
        fireConfidence = frameConf[preProcess.label_id['fire']]
        alog = {'fire': fireConfidence, 'fireless': 1-fireConfidence}
        log_dict[frameBase+idx] = alog

      frameBase += frameOffset
  return
  
def stackFrames(cap, prePolicy, batchsz):
  frameOffset = 0 # 已读帧
  l = []
  while frameOffset < batchsz:
    ret, frame = cap.read()
    if ret == False:
      break
    frameOffset += 1
    # 帧采样
    frame = prePolicy(frame)
    l.append(frame[np.newaxis, :, :, :])
  # 拼成 batch
  # batchFrame = np.concatenate(l)
  if len(l) == 0:
    return None, frameOffset
  else:
    return np.concatenate(l), frameOffset

if __name__ == '__main__':
  main()

class FramePreprocessPolicy(object):
  def __init__(self, fromSize, toSize):
    self.ops = []
    op = lambda frame : cv.resize(frame, toSize)
    self.ops.append(op)
    return

  def __call__(self, frame):
    for op in self.ops:
      frame = op(frame)
    return frame

class EvalDataset():
  label_id = {'fire': 0, 'fireless': 1}
  def __init__(self, video, video_idx, labeljson, resize):
    self.labeldict = preProcess.decodeLabel(labeljson)
    _h, _w = resize
    # self.categoryInfo = preProcess.info(self.labeldict)
    spans = self.labeldict[video_idx]
    
    frameIdx = 0
    img_list = []
    label_list = []
    # 读取每帧 构造输入
    cap = cv.VideoCapture(videoPath)
    while(cap.isOpened):
      img = cap.read()
      labelid = preProcess.judgeLabel(spans, frameIdx)
      if labelid < 0:
        continue
      img_list.append(img)
      label_list.append(labelid)
    cap.release()

    self.img_list = img_list
    self.label_list = label_list
    self.output_types = None    
    return
  
  def setEvalParams(self, batchsz, prefetch=None, cacheFile=None):
    self.batchsz = batchsz
    self.prefetch = prefetch
    self.cacheFile = cacheFile
    return

  def makeIter(self):
    t_imglist = tf.constant(self.img_list, dtype=tf.string)
    t_lbllist = tf.constant(self.label_list, dtype=tf.int32)
    dataset = tf.data.Dataset().from_tensor_slices((t_imglist, t_lbllist))
    dset = dataset.map(self._mapfn, num_parallel_calls=os.cpu_count())
  
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

  def _mapfn(self):
    with tf.device('/cpu:0'):
      # <https://www.tensorflow.org/performance/performance_guide>
      img_raw = tf.read_file(filename)
      decoded = tf.image.decode_jpeg(img_raw)
      _h, _w = self.resize
      resized = tf.image.resize_images(decoded, [_h, _w], tf.image.ResizeMethod.AREA)
    return resized, label, filename