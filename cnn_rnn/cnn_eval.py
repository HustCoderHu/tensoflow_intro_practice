import time
import os
import os.path as path
from os.path import join as pj
import json
from pprint import pprint
from pprint import pformat
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor

import tensorflow as tf

import var_config as cf
import dataPreProcess as preProcess
from EvalDataset import EvalDataset
import cnn
import slim_mbnet_v2

cwd = cf.cwd
labeljson = cf.labeljson
videoRoot = cf.videoRoot
# labeljson = r'/home/hzx/all_data/label.json'

# videoRoot = r'/home/kevin/data/all_data/'
# labeljson = r'/home/hzx/fireDetect-hzx/label.json'

log_dir = pj(cwd, 'train_eval_log')
lossSummaryRoot = pj(cwd, 'lossSummaryRoot')
pureEval = pj(cwd, 'pureEval')

ckpt_dir = pj(log_dir, 'ckpts')
# ckpt_dir = r'D:\Lab408\cnn_rnn\20180517\ckpts'
ckpt_path = pj(ckpt_dir, 'model.ckpt-1800')
ckpt_path = pj(ckpt_dir, 'model.ckpt-1200')
ckpt_path = pj(ckpt_dir, 'model.ckpt-5600')

model = None

batchsz = 60

def handleAllVideos():
  if not path.exists(pureEval):
    os.mkdir(pureEval)
  
  MAX_PARALLEL = 4
  processSet = []
  for videoIdx in cf.wholeSet:
    p = Process(target=forwardVideoLoss, args=(videoIdx,))
    p.start()
    processSet.append(p)
    if len(processSet) < MAX_PARALLEL:
      continue
    else :
      for p in processSet:
        p.join()
      processSet.clear()
  
  print('finish handleAllVideos')
  return

# 计算每帧的loss
def forwardVideoLoss(videoIdx):
  global model
  if not path.exists(log_dir):
    raise RuntimeError(log_dir+' not exists !')
    # os.mkdir(log_dir)
  if not path.exists(ckpt_dir):
    raise RuntimeError(ckpt_dir+' not exists !')
    # os.mkdir(ckpt_dir)

  # ------------------------------ prepare input ------------------------------
  _h = 240
  _w = 320
  dset = EvalDataset(videoRoot, videoIdx, labeljson, resize=(_h, _w))
  dset.setEvalParams(batchsz, prefetch=10, cacheFile=None)
  iterator = dset.makeIter()
  resized, labels, t_filenames = iterator.get_next()
  inputx = tf.reshape(resized, [-1, _h, _w, 3])
  
  # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ build graph \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  # model = cnn.CNN('NCHW')
  # The channel dimension of the inputs should be defined. Found `None`.
  # logits = model(inputx, castFromUint8=False)
  model = slim_mbnet_v2.MyNetV2(n_classes=2)
  logits = model(inputx, castFromUint8=False)
  
  with tf.name_scope('prediction'):
    t_pred_softmax = tf.nn.softmax(logits) # (batch, n_classes)
    t_pred = tf.argmax(logits, axis=1, output_type=tf.int32)

  with tf.name_scope('cross_entropy'):
    # Weighted loss Tensor of the same type as logits. 
    # If reduction is NONE, this has the same shape as labels; otherwise, it is scalar.
    t_loss = tf.losses.sparse_softmax_cross_entropy(labels, logits, 
        reduction=tf.losses.Reduction.NONE)
  # with tf.name_scope('accuracy'):
  #   acc_vec = tf.equal(labels, t_pred_softmax)
  #   acc = tf.reduce_mean(tf.cast(acc_vec, tf.float32))

  # ||||||||||||||||||||||||||||||  hooks ||||||||||||||||||||||||||||||
  # >>>  logging
  tf.logging.set_verbosity(tf.logging.INFO)

  # ////////////////////////////// session config //////////////////////////////
  sess_conf = tf.ConfigProto()
  sess_conf.gpu_options.allow_growth = True
  # sess_conf.gpu_options.per_process_gpu_memory_fraction = 0.9
 
  # ------------------------------  start  ------------------------------
  saver = tf.train.Saver()
  log2file = pj(pureEval, str(videoIdx)+'.json')
  log2file_pformat = pj(pureEval, str(videoIdx)+'_pformat.json')

  log_dict = {}
  # item  frameIdx:{'loss': float, fire': confidence, 'firelss': 1-confidence}
  with tf.Session(config= sess_conf) as sess:
    saver.restore(sess, ckpt_path)
    while True:
      try:
        loss, batchConfidence, pred, batchFilenames = sess.run(
            [t_loss, t_pred_softmax, t_pred, t_filenames], 
            feed_dict={model.is_training: False})
        nFrames = loss.shape[0]
        for i in range(nFrames):
          frameConf = batchConfidence[i]
          fireConfidence = frameConf[preProcess.label_id['fire']]
          # print(type(loss[i])) # np.float32
          # print(type(fireConfidence))
          alog = {'loss': float(loss[i]), 'pred': int(pred[i]),
              'fire': float(fireConfidence), 'fireless': float(1-fireConfidence)}

          fname = path.basename(batchFilenames[i])
          frameIdx = int(path.splitext(fname)[0])
          # TypeError: 5.3022945e-06 is not JSON serializable
          log_dict[frameIdx] = alog
      except tf.errors.OutOfRangeError as identifier:
        print(identifier.message)
        break

  with open(log2file, 'w') as f:
    json.dump(log_dict, f)
  with open(log2file_pformat, 'w') as f:
    f.write(pformat(log_dict))
  print('finish: ' + str(videoIdx))
  return log_dict

# 计算每个视频精度，以及整体精度 输出到json文件
def acc2file(labeljson):
  labeldict = preProcess.decodeLabel(labeljson)
  # trainset
  tofile = pj(cwd, 'acc.train.json')
  statistic = {}
  totalHit = 0
  totalFrames = 0
  for videoIdx in cf.trainSet:
    hit, frames, acc = computeAccuray(labeldict, videoIdx)
    alog = {'hit': hit, 'frames': frames, 'acc': acc}
    statistic[videoIdx] = alog
    # 累计
    totalHit += hit
    totalFrames += frames
  statistic['total'] = {'hit': totalHit, 'frames': totalFrames, 
      'acc': totalHit / totalFrames}
  pprint(statistic)

  # trainset
  tofile = pj(cwd, 'acc.eval.json')
  statistic = {}
  totalHit = 0
  totalFrames = 0
  for videoIdx in cf.evalSet:
    hit, frames, acc = computeAccuray(labeldict, videoIdx)
    alog = {'hit': hit, 'frames': frames, 'acc': acc}
    statistic[videoIdx] = alog
    # 累计
    totalHit += hit
    totalFrames += frames
  statistic['total'] = {'hit': totalHit, 'frames': totalFrames, 
      'acc': totalHit / totalFrames}
  pprint(statistic)

  # with open(tofile, 'w') as f:
    # json.dump(statistic, f)
  return


def computeAccuray(labeldict, videoIdx):
  lossfile = pj(pureEval, str(videoIdx)+'.json')
  if not path.exists(lossfile):
    return 0, 0, 0
  with open(lossfile, 'r') as f:
    lossdict = json.load(f)

  hit = 0
  for strk, alog in lossdict.items():
    pred = alog['pred']
    frameIdx = int(strk)
    exactLabel = preProcess.judgeLabel_ease(labeldict, videoIdx, frameIdx)
    if pred == exactLabel:
      hit += 1
  frames = len(lossdict.keys())
  acc = hit / frames
  return hit, frames, acc

# 将所给视频有标签帧的loss 添加到summary，方便用tfboard观察
# videoIdxSet 视频id号集合
def framesLoss2tfboard(videoIdxSet):
  # log2file = pj(pureEval, '1.json')
  # with open(log2file, 'r') as f:
  #   log_dict = json.load(f)
  # print(type(log_dict))
  # for k in log_dict.keys():
  #   print(log_dict[k])
  #   break
  # print(log_dict.values()[0])
  # print(type(log_dict.values()))
  # return

  if not path.exists(lossSummaryRoot):
    os.mkdir(lossSummaryRoot)

  loss = tf.placeholder(tf.float32, ())
  fireConfidence = tf.placeholder(tf.float32, ())

  summary_protobuf = {
    'loss': tf.summary.scalar('cross_entropy', loss),
    'fireConfidence': tf.summary.scalar('fireConfidence', fireConfidence)
  }
  merged = tf.summary.merge(list(summary_protobuf.values()))

  sess_conf = tf.ConfigProto()
  sess_conf.gpu_options.allow_growth = True
  with tf.Session(config= sess_conf) as sess:
    for idx in videoIdxSet:
      # 读取每个视频 帧结果
      log2file = pj(pureEval, str(idx)+'.json')
      with open(log2file, 'r') as f:
        log_dict = json.load(f)

      # 写入 summary
      summaryDir = pj(lossSummaryRoot, str(idx))
      summaryWriter = tf.summary.FileWriterCache.get(summaryDir)
      # for k, _dict in log_dict.values():
      # keySet = [k for k in log_dict.keys()]
      # keySet.sort() # 排序不对
      # print(keySet)
      # 下面排序正确
      mapInt2alog = {}
      for k in log_dict.keys():
        mapInt2alog[int(k)] = log_dict[k]
      # print(mapInt2alog.keys())

      for k in mapInt2alog.keys():
        _dict = mapInt2alog[k]
        strMerged = sess.run(merged, feed_dict={loss: _dict['loss'],
            fireConfidence: _dict['fire']})
        frameIdx = int(k)
        summaryWriter.add_summary(strMerged, frameIdx)
      summaryWriter.flush()
      print('finish video: ' + str(idx))

# 将所给视频有标签帧的loss 添加到summary，方便用tfboard观察
# 序列化成 protobuff 是 cpu 计算密集型
# 多进程版
def mp_framesLoss2tfboard(videoIdxSet):
  max_workers = os.cpu_count()
  q = Queue(max_workers)

  paramDict = {'sess': None, 'mergedOp': None, 'loss': None,
      'fireConfidence': None}
  sess_conf = tf.ConfigProto()
  sess_conf.gpu_options.allow_growth = True
  # 准备 cpu核心数 对应的 n 组参数
  for i in range(max_workers):
    graph = tf.Graph()
    with graph.as_default():
      loss = tf.placeholder(tf.float32, ())
      fireConfidence = tf.placeholder(tf.float32, ())
      summary_protobuf = {
          'loss': tf.summary.scalar('cross_entropy', loss),
          'fireConfidence': tf.summary.scalar('fireConfidence', fireConfidence)
      }
      mergedOp = tf.summary.merge(list(summary_protobuf.values()))
      sess = tf.Session(config=sess_conf)
      paramDict['sess'] = sess
      paramDict['mergedOp'] = mergedOp
      paramDict['loss'] = loss
      paramDict['fireConfidence'] = fireConfidence
    # 参数组入队
    q.put(paramDict)
  # 进程池启动
  with ProcessPoolExecutor(max_workers) as executor:
    for videoIdx in videoIdxSet:
      # res = executor.map()
      res = executor.submit(worker_computeAvideo, q, videoIdx)
      # resultList.append(res)
      # break
    executor.shutdown()
  print('--- worker_computeAvideo executor shutdown()')

  # 释放资源
  for i in range(max_workers):
    paramDict = q.get(True)
    paramDict['sess'].close()
  return

def worker_computeAvideo(q, videoIdx):
  # 先获取资源
  paramDict = q.get(True) # True 队列空时阻塞
  sess = paramDict['sess']
  mergedOp = paramDict['mergedOp']
  loss = paramDict['loss']
  fireConfidence = paramDict['fireConfidence']

  log2file = pj(pureEval, str(videoIdx)+'.json')
  with open(log2file, 'r') as f:
    log_dict = json.load(f)
  # 写入 summary
  summaryDir = pj(lossSummaryRoot, str(videoIdx))
  summaryWriter = tf.summary.FileWriterCache.get(summaryDir)
  mapInt2alog = {}
  for k in log_dict.keys():
    mapInt2alog[int(k)] = log_dict[k]

  for k in mapInt2alog.keys():
    _dict = mapInt2alog[k]
    strMerged = sess.run(mergedOp, feed_dict={loss: _dict['loss'],
        fireConfidence: _dict['fire']})
    frameIdx = int(k)
    summaryWriter.add_summary(strMerged, frameIdx)
  summaryWriter.flush()

  q.put(paramDict, True) # 资源归还 True 队列满时阻塞
  print('finish video: ' + str(videoIdx))
  return

def displayJson(jfile):
  with open(jfile, 'r') as f:
    dat = json.load(f)
  pprint(dat)

if __name__ == '__main__':
  # a = {}
  # a[3] = 'aa'
  # a[4] = 'bb'
  # jstr = json.dumps(a)
  # pprint(jstr)
  # b = json.loads(jstr)
  # for key in b.keys():
  #   print(type(key)) # str
  #   print(key)

  # if not path.exists(pureEval):
  #   os.mkdir(pureEval)
  # videoIdxSet = range(83, 111)
  # forwardVideoLoss(2)
  # handleAllVideos()
  acc2file(labeljson)
  # framesLoss2tfboard(videoIdxSet)

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

