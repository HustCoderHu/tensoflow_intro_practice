import shutil as sh
from os.path import join as pj
import os.path as path
import os
import json
from pprint import pprint
from concurrent.futures import ProcessPoolExecutor

import var_config as cf

videoRoot = cf.videoRoot

cwd = cf.cwd
newVideoRoot = pj(cwd, 'all_data')

# newVideoRoot = r'/home/hzx/all_data_reduced'

def cpimg():
  if not path.exists(newVideoRoot):
    os.mkdir(newVideoRoot)
  # videoIdxSet = [1]
  videoIdxSet = list(range(1, 82))
  videoIdxSet.extend(list(range(83, 111)))
  framesPerVideo = 300
  
  # 可迭代参数集合
  paramsCollection = []
  for videoIdx in videoIdxSet:
    paramsCollection.append((videoIdx, framesPerVideo))

  max_workers = os.cpu_count()
  with ProcessPoolExecutor(max_workers) as executor:
    # 保持和multiprocessing.pool的默认chunksize一样
    # chunksize, extra = divmod(len(paramsCollection), executor._max_workers * 4)
    # executor.map(worker_cpVideoFrames, paramsCollection, chunksize=chunksize)
    for videoIdx in videoIdxSet:
      executor.submit(worker_cpVideoFrames, videoIdx, framesPerVideo)
    executor.shutdown()
  print('--- worker_cpVideoFrames executor shutdown()')
  return

# framesPerVideo 每个视频取帧数上限 等间隔取
def worker_cpVideoFrames(videoIdx, framesPerVideo):
  print('start video: ' + str(videoIdx))
  framesDir = pj(videoRoot, '{:0>3d}'.format(videoIdx))
  flist = os.listdir(framesDir)
  flist = [f for f in flist if path.splitext(f)[1] == '.jpg']
  flist.sort()

  nImg = len(flist)
  interval, mod = divmod(nImg, framesPerVideo) # nImg // framesPerVideo
  if mod != 0:
    interval += 1

  dstDir = pj(newVideoRoot, '{:0>3d}'.format(videoIdx))
  if not path.exists(dstDir):
    os.mkdir(dstDir)
  
  # 图像数量小于指定 全部复制
  if nImg <= framesPerVideo:
    for f in flist:
      src = pj(framesDir, f)
      dst = pj(dstDir, f)
      os.symlink(src, dst)
  else :
    for f in flist:
      frameIdx = int(path.splitext(f)[0])
      if frameIdx%interval != 0:
        continue
      src = pj(framesDir, f)
      dst = pj(dstDir, f)
      os.symlink(src, dst)
      # os.copy(src, dst)
      # os.copy2(src, dst)
  print('finish video: ' + str(videoIdx))
  return

def tst_softlink():
  labeljson = r'/home/hzx/all_data/label.json'
  dst = r'/home/hzx/all_data/softlink.json'
  os.symlink(labeljson, dst)
  with open(dst, 'r') as f:
    dat = json.load(f)
  pprint(dat)
  

if __name__ == '__main__':
  # a = 3200
  # b = 300
  # c = a // b
  # print(c) # 10
  # print(37915 / 50)
  # tst_softlink()
  # worker_cpVideoFrames(2, 5)
  cpimg()