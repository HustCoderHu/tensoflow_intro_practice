import json
from pprint import pprint

label_id = {'fire': 0, 'fireless': 1}

# 解析标签文件
  # retval {video_idx (int): 范围}
  # 范围: {'fire': A, 'fireless': B}
  # A, B: [[0, x], [y, z], [z, .], ...]
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

def judgeLabel_ease(labeldict, video_idx, frameIdx):
  spans = labeldict[video_idx]
  # labelid = -1
  spansFire = spans['fire']
  for span in spansFire:
    if span[0] <= frameIdx and frameIdx <= span[1]:
      return label_id['fire']
  spansFireless = spans['fireless']
  for span in spansFireless:
    if span[0] <= frameIdx and frameIdx <= span[1]:
      return label_id['fireless']
  return -1
    # spansFireless = spans['fireless']

def judgeLabel(spans, frameIdx):
  # labelid = -1
  spansFire = spans['fire']
  for span in spansFire:
    if span[0] <= frameIdx and frameIdx <= span[1]:
      return label_id['fire']
  spansFireless = spans['fireless']
  for span in spansFireless:
    if span[0] <= frameIdx and frameIdx <= span[1]:
      return label_id['fireless']
  return -1

# 统计每个视频有火和无火的帧数
# and all
# labeldict: returned by decodeLabel()
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
  
  categoryInfo['totalfire:'] = totalFire
  categoryInfo['totalFireless:'] = totalFireless
  categoryInfo['all:'] = totalFire + totalFireless
  # print('total:')
  # print('fire: {}'.format(totalFire))
  # print('fireless: {}'.format(totalFireless))
  return categoryInfo

def tst():
  labeljson = r'D:\Lab408\cnn_rnn\label.json'

  labeldict = decodeLabel(labeljson)
  videoIdx = 14
  frameIdx = 352
  label_id = judgeLabel_ease(labeldict, videoIdx, frameIdx)
  print(label_id)
  return

if __name__ == '__main__':
  tst()