import tensorflow as tf
from tensorflow import layers as tl
from tensorflow import nn
import tensorflow.contrib as tc

def conv_S(in_planes,out_planes,stride=1,padding=1):
  # as is descriped, conv S is 1x3x3
  return tl.conv3d(in_planes,out_planes,kernel_size=(1,3,3),strides=1,
                     padding='valid',bias=False)

def conv_T(in_planes,out_planes,stride=1,padding=1):
  # conv T is 3x1x1
  return nn.conv3d(in_planes,out_planes,kernel_size=(3,1,1),strides=1,
                     padding=padding,bias=False)

def downsample_basic_block(x, planes, stride):
  out = tl.average_pooling3d(x, pool_size=1, strides=stride)
  zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                           out.size(2), out.size(3),
                           out.size(4)).zero_()
  if isinstance(out.data, torch.cuda.FloatTensor):
      zero_pads = zero_pads.cuda()

  out = Variable(torch.cat([out.data, zero_pads], dim=1))

  return out

class bottleneck():
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None,
      n_s=0,depth_3d=47,ST_struc=('A','B','C')):
    self.w_initer=tc.layers.xavier_initializer(tf.float32)

    self.downsample = downsample
    self.depth_3d=depth_3d
    self.ST_struc=ST_struc
    self.len_ST=len(self.ST_struc)
    stride_p=stride
    if not self.downsample ==None:
      stride_p=(1,2,2)
    if n_s<self.depth_3d:
      if n_s==0:
        stride_p=1
      self.conv1 = tl.Conv3D(filters=planes, kernel_size=1, strides=stride_p, 
          padding='same', use_bias=False, kernel_initializer=self.w_initer)
      self.bn1 = nn.BatchNorm3d(planes)
      self.bn1 = tl.BatchNormalization(axis=axis, scale=False,
          training=self.is_training, fused=True)
    else:
      if n_s==self.depth_3d:
          stride_p=2
      else:
          stride_p=1
      self.conv1 = tl.Conv2D(filters=planes, kernel_size=1, strides=stride_p, 
          padding='same', use_bias=False, kernel_initializer=self.w_initer)
      self.bn1 = nn.BatchNorm2d(planes)
    # self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
    #                        padding=1, bias=False)
    self.id=n_s
    self.ST=list(self.ST_struc)[self.id%self.len_ST]
    if self.id<self.depth_3d:
        self.conv2 = conv_S(planes,planes, stride=1,padding=(0,1,1))
        self.bn2 = nn.BatchNorm3d(planes)
        #
        self.conv3 = conv_T(planes,planes, stride=1,padding=(1,0,0))
        self.bn3 = nn.BatchNorm3d(planes)
    else:
        self.conv_normal = nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1,bias=False)
        self.bn_normal = nn.BatchNorm2d(planes)

    if n_s<self.depth_3d:
        self.conv4 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm3d(planes * 4)
    else:
        self.conv4 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)

    self.stride = stride