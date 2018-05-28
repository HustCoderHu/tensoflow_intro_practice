import sys
import os
import os.path as path
from os.path import join as pj
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.slim as slim

model_slim = r'D:\github_repo\models\research\slim'
sys.path.append(model_slim)

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

from datasets import imagenet
from nets import inception
from nets.mobilenet import mobilenet_v2
from preprocessing import inception_preprocessing

from datasets import dataset_utils

image_size = inception.inception_v1.default_image_size
checkpoints_dir = r'D:\Lab408\tfslim\ckpts'

def download_and_uncompress_tarball():
  url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"
  
  
  if not tf.gfile.Exists(checkpoints_dir):
      tf.gfile.MakeDirs(checkpoints_dir)
  
  dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)
  return

def inception_tst():
  with tf.Graph().as_default():
    url = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'
    image_string = urllib.urlopen(url).read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)
    
    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_v1_arg_scope()):
      logits, _ = inception.inception_v1(processed_images, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)

    variables_to_restore = slim.get_variables()
    print(len(variables_to_restore))
    # pprint(variables_to_restore)
    variables_to_restore = slim.get_model_variables()
    print(len(variables_to_restore))
    return
    
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v1.ckpt'),
        slim.get_model_variables('InceptionV1'))
    
    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
        
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.axis('off')
    plt.show()

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))

def mobileNetV2_tst():
  with tf.Graph().as_default():
    is_training = tf.placeholder(tf.bool, [])
    url = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'
    image_string = urllib.urlopen(url).read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)
    # print(processed_images.shape)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_v1_arg_scope()):
      logits, endpoints = mobilenet_v2.mobilenet(processed_images)
    # print(type(endpoints))
    # print(len(endpoints.keys()))
    # print(endpoints['layer_18/output'].shape) # (1, 7, 7, 320)
    # print(endpoints['layer_18'].shape) # (1, 7, 7, 320)
    # print(endpoints['layer_19'].shape) # (1, 7, 7, 320)
    # print(endpoints['global_pool'].shape) # (1, 1, 1, 1280)
    # pprint(endpoints.keys())

    variables_to_restore = slim.get_variables_to_restore(exclude=['MobilenetV2/Logits/Conv2d_1c_1x1'])
    restorer = tf.train.Saver(variables_to_restore)
    print(len(variables_to_restore)) # 260
    # print(variables_to_restore)
    
    dropout_keep_prob = 0.5
    n_classes = 2
    weight_decay = 0.05
    with tf.variable_scope('addition', 'fc'):
      # flatten = tf.flatten(endpoints['global_pool'])
      flatten = slim.flatten(endpoints['global_pool'])
      with slim.arg_scope(
        [slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer = tc.layers.xavier_initializer(tf.float32), 
        # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        activation_fn=tf.nn.relu6) as sc:
        net = slim.fully_connected(flatten, 128, activation_fn=None, scope='fc1')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout')
        logits = slim.fully_connected(net, n_classes, activation_fn=None, scope='fc2')
    
    # with tf.name_scope('loss') :

    probabilities = tf.nn.softmax(logits)

    ckptDir = r'D:\Lab408\tfslim\mobileNetV2'
    ckptPath = pj(ckptDir, 'mobilenet_v2_1.0_224.ckpt')

    # variables_to_restore = slim.get_variables('MobilenetV2/Logits/Conv2d_1c_1x1')
    variables_to_save = slim.get_variables_to_restore(exclude=['MobilenetV2/Logits/Conv2d_1c_1x1'])
    saver = tf.train.Saver(variables_to_save)
    print(len(variables_to_save)) # 264
    # print(variables_to_save)
    # pprint(variables_to_restore)
    # variables_to_restore = slim.get_model_variables()
    # print(len(variables_to_restore))
    
    op_init1 = tf.variables_initializer(tf.global_variables())
    op_init2 = tf.variables_initializer(tf.local_variables())
    op_group = tf.group(op_init1, op_init2)

    init_fn0 = slim.assign_from_checkpoint_fn(ckptPath, variables_to_restore)
    sess_conf = tf.ConfigProto()
    sess_conf.gpu_options.allow_growth = True
    with tf.Session(config= sess_conf) as sess:
      sess.run(op_group)
      init_fn0(sess)
      # Failed to find any matching files for D:\Lab408\tfslim\ckpts\mobilenet_v2_1.0_224.ckpt
      # restorer.restore(sess, ckptPath)
      np_image, probabilities = sess.run([image, probabilities],
          feed_dict={is_training: False})
      probabilities = probabilities[0, 0:]
      sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
      saveto = r'D:\Lab408\tfslim\mobileNetV2-finetue\aa'
      restorer.save(sess, saveto)
      # saver.save(sess, saveto)
    
    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(n_classes):
      index = sorted_inds[i]
      outstr = 'Probability {:.2%} => {}'.format(probabilities[index], names[index])
      print(outstr)
      # print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))
    return

def tst1():
  names = imagenet.create_readable_names_for_imagenet_labels()
  print(type(names))
  print(len(names.keys()))
  n = 0
  for k, v in names.items():
    print('{}: {}'.format(k, v))
    n += 1
    if (n==3):
      break
  # 1001
  # 0: background
  # 1: tench, Tinca tinca
  # 2: goldfish, Carassius auratus

if __name__ == '__main__':
  # tst1()
  # tst_perVideoNpy()
  # download_and_uncompress_tarball()
  # inception_tst()
  mobileNetV2_tst()
  # nParam = 7 * 7 * 320 * 128
  # Mmem = 4*nParam / 1024 / 1024
  # print(Mmem)
  