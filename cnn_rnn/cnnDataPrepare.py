import shutil as sh
from os.path import join as pj
import os.path as path
import os

videoRoot = r'W:\hzx\all_data'
newVideoRoot = r'W:\hzx\all_data_reduced'
videoRoot = r'/home/hzx/all_data/'
newVideoRoot = r'/home/hzx/all_data_reduced'

def cpimg():
  return

# framesPerVideo 每个视频取帧数 等间隔取
def cpVideoFrames(videoIdx, framesPerVideo):
  framesDir = pj(videoRoot, '{:0>3d}'.format(videoIdx))
  flist = os.listdir(framesDir)
  flist = [f for f in flist if path.splitext(f)[1] == '.jpg']
  

  nImg = len(flist)
  interval = nImg // framesPerVideo

  flist.sort()
  dstDir = pj(newVideoRoot, '{:0>3d}'.format(videoIdx))
  if not path.exists(dstDir):
    os.mkdir(dstDir)

  for f in flist:
    frameIdx = int(path.splitext(f)[0])
    if frameIdx%interval != 0:
      continue
    src = pj(framesDir, f)
    dst = pj(dstDir, f)
    # sh.copyfile()
    os.symlink(src, dst)
  return

if __name__ == '__main__':
  # a = 3200
  # b = 300
  # c = a // b
  # print(c) # 10
  cpVideoFrames(1, 5)
  # cpimg()