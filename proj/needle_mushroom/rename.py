import cv2 as cv
import os
import os.path as path
from os.path import join as pjoin
from pprint import pprint
import shutil

def rename_all(_src_dir, _dst_dir):
  if not path.isdir(_src_dir):
    return
  flist = os.listdir(_src_dir)
  print(len(flist))
  for i, f in enumerate(flist):
    src_path = pjoin(_src_dir, f)
    dst_name = str(i) + path.splitext(f)[1]
    dst_path = pjoin(_dst_dir, dst_name)
    # print(dst_name)
    # print(dst_path)
    shutil.move(src_path, dst_path)
    # break
  return




src_dir = r'F:\Lab408\jinzhengu\root\resized\train'
label_list = os.listdir(src_dir)
for label in label_list:
  rename_all(pjoin(src_dir, label))
  
print('finish')
