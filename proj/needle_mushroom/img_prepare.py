import cv2 as cv
import os
import os.path as path
from os.path import join as pjoin
from pprint import pprint
import random as rd
import sys
import shutil

def cvt_all(_src_dir, _dst_dir, img_sz, 
    src_ext_list=['.bmp', '.jpg'], dst_ext='.bmp'):
  flist = os.listdir(_src_dir)
  imglist = [f for f in flist if path.splitext(f)[1] in src_ext_list]
  print('{} total: {}'.format(src_ext_list, len(imglist)))
  # print(len(imglist))
  # return
  
  for i, f in enumerate(imglist):
    img_path = pjoin(_src_dir, f)
    img = cv.imread(img_path)
    img = cv.resize(img, img_sz, interpolation=cv.INTER_AREA)

    dst_name = str(i) + dst_ext
    dst_path = pjoin(_dst_dir, dst_name)
    cv.imwrite(dst_path, img)

  return

def _divide_train_eval(_src_dir_list, train_dir, eval_dir,
    train_num):
  """
  """
  img_path_list = []
  for img_dir in _src_dir_list:
    imglist = os.listdir(img_dir)
    _path_list = [pjoin(img_dir, f) for f in imglist]
    img_path_list.extend(_path_list)
  total = len(img_path_list)
  print(_src_dir_list)
  print('-- total: {}'.format(total))
  sample_list = rd.sample(range(total), train_num)
  sample_list.sort()

  dir_list = [eval_dir] * total
  for i in sample_list:
    dir_list[i] = train_dir
  
  for i, f in enumerate(img_path_list):
    dst_name = str(i) + path.splitext(f)[1]
    dst_path = pjoin(dir_list[i], dst_name)
    # print(dst_path)
    # break
    shutil.copy2(f, dst_path)

  return

def divide_train_eval(_src_dir, _dst_dir, label, train_num):
  _src_dir_list = [
    pjoin(pjoin(_src_dir, 'train'), label),
    pjoin(pjoin(_src_dir, 'eval'), label)
  ]
  train_dir = pjoin(_dst_dir, 'train')
  eval_dir = pjoin(_dst_dir, 'eval')
  if not path.exists(train_dir):
    os.mkdir(train_dir)
  if not path.exists(eval_dir):
    os.mkdir(eval_dir)
  _dst_label_dir_train = pjoin(train_dir, label)
  _dst_label_dir_eval = pjoin(eval_dir, label)

  if not path.exists(_dst_label_dir_train):
    os.mkdir(_dst_label_dir_train)
  if not path.exists(_dst_label_dir_eval):
    os.mkdir(_dst_label_dir_eval)

  _divide_train_eval(
    _src_dir_list,
    _dst_label_dir_train,
    _dst_label_dir_eval,
    train_num
  )


def stage_1_convert():
  src_dir = r'F:\Lab408\jinzhengu\root'
  dst_dir = r'F:\Lab408\jinzhengu\root\resized'

  phase = ['train', 'eval']

  label_list = os.listdir(pjoin(src_dir, 'train'))

  dir_dict = {}

  for _p in phase:
    _dir = pjoin(dst_dir, _p)
    if not path.exists(_dir):
      os.mkdir(_dir)
    for label in label_list:
      __dir = pjoin(_dir, label)
      if not path.exists(__dir):
        os.mkdir(__dir)
      key = pjoin(pjoin(src_dir, _p), label)
      val = __dir
      dir_dict[key] = val
  # print(dir_dict)
  pprint(dir_dict)

  img_sz = (250, 200)
  for _src_dir, _dst_dir in dir_dict.items():
    print('------ start')
    print(_src_dir + '   to   ' + _dst_dir)
    # cvt_all(_src_dir, _dst_dir, img_sz)
    print('====== end')

  print('stage_1_convert end')
  return

def stage_2_shuffle_divide():
  src_dir = r'F:\Lab408\jinzhengu\root\resized'
  dst_dir = r'F:\Lab408\jinzhengu\root\shuffled_divided'
  divide_train_eval(src_dir, dst_dir, label='a', train_num=870)
  divide_train_eval(src_dir, dst_dir, label='b', train_num=870)
  print('------------------- stage_2_shuffle_divide end')
  return



if __name__ == '__main__':
  print('here')
# stage_1_convert()
stage_2_shuffle_divide()
print('end')
exit(0)
# os.mkdir(_dir) for _dir in phase_dir
# train
# a 894
# b 797
# eval
# a 782
# b 173

