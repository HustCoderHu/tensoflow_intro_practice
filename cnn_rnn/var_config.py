import os.path as path
from os.path import join as pj

cwd = r'/home/hzx/fireDetect-hzx/log20180517'
cwd = r'/home/hzx/fireDetect-hzx/log20180524'
videoRoot = pj(cwd, 'all_data')
cwd = r'/home/hzx/fireDetect-hzx/log20180529'

# cwd = r'D:\Lab408\cnn_rnn\20180517'
# cwd = r'D:\Lab408\cnn_rnn\20180524'

log_dir = pj(cwd, 'train_eval_log')
ckpt_dir = path.join(log_dir, 'ckpts')

# videoRoot = pj(cwd, 'all_data')
labeljson = r'/home/hzx/all_data/label.json'

evalSet = [47, 48, 49, 50, 27, 33, 21, 32]
wholeSet = list(range(1, 111))
wholeSet.remove(82) # 82 是竖着的
trainSet = wholeSet.copy()
for idx in evalSet:
  trainSet.remove(idx)

if __name__ == '__main__':
  print(trainSet)