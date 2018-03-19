import numpy as np

import tensorflow as tf
from tensorflow import layers as tl
import tensorflow.contrib as tc

import p3d
# def conv_S

def p3d63(pretrained=False, modality='RGB', **kwargs):
  """Construct a P3D63 modelbased on a ResNet-50-3D model.
  """
  model = P3D(Bottleneck, [3, 4, 6, 3], modality=modality, **kwargs)
  if pretrained == True:
    if modality=='RGB':
      pretrained_file=r'path\to\ckpt' # p3d_rgb_199.checkpoint.pth.tar
    elif modality=='Flow':
      pretrained_file=r'path\to\ckpt' # 'p3d_flow_199.checkpoint.pth.tar'
    # weights=torch.load(pretrained_file)['state_dict']
    # model.load_state_dict(weights)
  return model