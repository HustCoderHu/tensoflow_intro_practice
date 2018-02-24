import cv2 as cv
import os
import os.path as path
from pprint import pprint

# import sys

"""
dir hierarchy

train
- a
- b

test
- a
- b

"""

# cwd = os.getcwd()
# train_dir = path.join(cwd, "train")
# src_train_a = path.join(train_dir, "a")
# src_train_b = path.join(train_dir, "b")
# eval_dir = path.join(cwd, "test")
# src_eval_a = path.join(eval_dir, "a")
# src_eval_b = path.join(eval_dir, "b")

# resized_train = path.join(cwd, "resized")
# resized_a = path.join(resized_train, "a")
# resized_b = path.join(resized_train, "b")

def resz_allindir(src, dst, dst_sz):
  flist = os.listdir(src)
  bmplist = [f for f in flist if path.splitext(f)[1]==".bmp"]
  print(len(bmplist))
  # return
  for _bmp in bmplist:
    img_path = path.join(src, _bmp)
    img = cv.imread(img_path)
    img = cv.resize(img, img_sz, interpolation=cv.INTER_AREA)
    img_path = path.join(dst, _bmp)
    cv.imwrite(img_path, img)
  return

# print(len(bmplist))


# resz_allindir(src_b, resized_b, img_sz)

phase = ['train', 'eval']
label_list = ['a', 'b']
img_sz = (250, 200)

src_dir = r'F:\Lab408\jinzhengu\root'
resized_dir = r'F:\Lab408\jinzhengu\root\resized'

if not path.exists(src_dir):
  raise SystemError(src_dir + " not exist")
if not path.exists(resized_dir):
  os.mkdir(resized_dir)

dict_src = {}
dict_resized = {}
for _p in phase:
  dict_src[_p] = {}
  dict_resized[_p] = {}
  orgi_phase_dir = path.join(src_dir, _p)
  resized_phase_dir = path.join(resized_dir, _p)
  for label in label_list:
    dict_src[_p][label] = path.join(orgi_phase_dir, label)
    dict_resized[_p][label] = path.join(resized_phase_dir, label)

pprint(dict_src)
pprint(dict_resized)

for _p in phase:
  if _p == 'train':
    continue
  for label in label_list:
    _src_dir = dict_src[_p][label]
    _dst_dir = dict_resized[_p][label]
    if not path.exists(_src_dir):
      raise SystemError(_src_dir + " not exist")
    if not path.exists(_dst_dir):
      os.mkdir(_dst_dir)

    resz_allindir(_src_dir, _dst_dir, img_sz)
  
print('finish')
exit(0)

# print("finish")
# img_path = path.join(a_dir, "2.bmp")
# img = cv.imread(img_path)
# img_sz = (500, 400)
# img_sz = (250, 200)
# img = cv.resize(img, img_sz, interpolation=cv.INTER_AREA)
# cv.imshow("2.bmp", img)
# cv.waitKey()
# cv.imwrite("2.bmp", img)
# flist = os.listdir