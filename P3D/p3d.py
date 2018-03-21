import numpy as np

import tensorflow as tf
from tensorflow import layers as tl
from tensorflow import nn
import tensorflow.contrib as tc



class p3d():
  def __init__(self, block, layers, modality='RGB',
      shortcut_type='B', num_classes=400,dropout=0.5,ST_struc=('A','B','C')):
    self.w_initer=tc.layers.xavier_initializer(tf.float32)
    self.data_format='NCHW'
    self.layer_data_format = 'channels_last' if data_format == 'NHWC' \
        else 'channels_first'
    self.is_training=True

    self.inplanes = 64

    self.input_channel = 3 if modality=='RGB' else 2  # 2 is for flow 
    self.ST_struc=ST_struc

    self.conv1_custom = nn.Conv3d(self.input_channel, 64, kernel_size=(1,7,7), stride=(1,2,2),
                                padding=(0,3,3), bias=False)
    self.conv1_custom = tl.Conv3D(filters=64, kernel_size=(1,7,7), strides=(1,2,2), 
        padding=self.layer_data_format, use_bias=False, kernel_initializer=self.w_initer)
    
    self.depth_3d = sum(layers[:3])# C3D layers are only (res2,res3,res4),  res5 is C2D

    axis = 1 if self.data_format=="NCHW" else -1
    self.bn1 = tl.BatchNormalization(axis=axis, scale=False fused=True)
    # out = self.bn(in, training=True) False for eval
    self.cnt = 0
    self.relu = lambda input : nn.relu(input)
    self.maxpool = tl.MaxPooling3D(pool_size=(2, 3, 3), strides=2, padding='valid',
        data_format=self.layer_data_format) # pooling layer for conv1.
    self.maxpool = tl.MaxPooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), padding='valid',
        data_format=self.layer_data_format) # pooling layer for res2, 3, 4.
    
    self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
    self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

    self.avgpool = tl.AveragePooling2D(pool_size=5, strides=1,
        data_format=self.layer_data_format)                              # pooling layer for res5.
    self.dropout=tl.Dropout(dropout)
    self.fc = tl.Dense(num_classes, use_bias=False, kernel_initializer=self.w_initer)
    # self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
        if isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    # some private attribute
    self.input_size=(self.input_channel,16,160,160)       # input of the network
    self.input_mean = [0.485, 0.456, 0.406] if modality=='RGB' else [0.5]
    self.input_std = [0.229, 0.224, 0.225] if modality=='RGB' else [np.mean([0.229, 0.224, 0.225])]

    return
  
  def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
    downsample = None
    stride_p=stride #especially for downsample branch.

    if self.cnt<self.depth_3d:
      if self.cnt==0:
        stride_p=1
      else:
         stride_p=(1,2,2)
      if stride != 1 or self.inplanes != planes * block.expansion:
        if shortcut_type == 'A':
            downsample = partial(downsample_basic_block,
                                 planes=planes * block.expansion,
                                 stride=stride)
        else:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride_p, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

    else:
      if stride != 1 or self.inplanes != planes * block.expansion:
        if shortcut_type == 'A':
          downsample = partial(downsample_basic_block,
                               planes=planes * block.expansion,
                               stride=stride)
        else:
          downsample = nn.Sequential(
              nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=2, bias=False),
              nn.BatchNorm2d(planes * block.expansion)
          )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample,n_s=self.cnt,depth_3d=self.depth_3d,ST_struc=self.ST_struc))
    self.cnt+=1

    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(self.inplanes, planes,n_s=self.cnt,depth_3d=self.depth_3d,ST_struc=self.ST_struc))
        self.cnt+=1

    return nn.Sequential(*layers)

    return