import time
import sys
import os
import os.path as path
from os.path import join as pj
import tensorflow as tf
# import tensorflow.contrib as tc
import tensorflow.contrib.slim as slim

import var_config as cf
import cnn_dataset
import slim_mbnet_v2

repoRoot = r'E:\github_repo\tensorflow_intro_practice'
sys.path.append(pj(repoRoot, 'proj', 'needle_mushroom'))
import runhooks

def main():
  cwd = cf.cwd
  log_dir = pj(cwd, 'train_eval_log')
  ckpt_dir = path.join(log_dir, 'ckpts')

  videoRoot = cf.videoRoot
  # videoRoot = pj(cwd, 'all_data')
  labeljson = cf.labeljson
  evalSet = cf.evalSet

  if not path.exists(log_dir):
    os.mkdir(log_dir)
  if not path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
  
  eval_interval = 20
  save_summary_steps = 5
  save_ckpts_steps = 100
  train_batchsz = 50
  eval_batchsz = 200

  # ------------------------------ prepare input ------------------------------
  _h = 240
  _w = 320
  dset = cnn_dataset.MyDataset(videoRoot, labeljson, evalSet, resize=(_h, _w))
  dset.setTrainParams(train_batchsz, prefetch=10)
  dset.setEvalParams(eval_batchsz, prefetch=3)
  iter_dict = {
    'train': dset.makeTrainIter(),
    'eval': dset.makeEvalIter()
  }

  holder_handle = tf.placeholder(tf.string, [])
  iter = tf.data.Iterator.from_string_handle(
      holder_handle, dset.output_types)
  # next_elem = iter.get_next()
  inputx, labels, filename = iter.get_next()
  inputx = tf.reshape(inputx, [-1, _h, _w, 3])

  # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ build graph \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  model = slim_mbnet_v2.MyNetV2(n_classes=2)
  logits = model(inputx, castFromUint8=False)
  with tf.name_scope('cross_entropy'):
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    
  with tf.name_scope('accuracy'):
    _pred = tf.argmax(logits, axis=1, output_type=tf.int32)
    acc_vec = tf.equal(labels, _pred)
    acc = tf.reduce_mean(tf.cast(acc_vec, tf.float32))


  with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(1e-4)
  
  variables_to_train = []
  trainable_variables = tf.trainable_variables()
  for var in trainable_variables:
    if var in model.variables_to_restore:
      continue
    variables_to_train.append(var)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step(),
      variables_to_train)
  # ||||||||||||||||||||||||||||||  hooks ||||||||||||||||||||||||||||||
  # >>>  logging
  tf.logging.set_verbosity(tf.logging.INFO)

  # >>>  summary
  summary_conf = {
    'dir': log_dir,
    'saved_steps': save_summary_steps
  }
  summary_protobuf = {
    'loss': tf.summary.scalar('cross_entropy', loss),
    'accuracy': tf.summary.scalar('accuracy', acc)
  }
  # >>> main run hook
  eval_hook = runhooks.RunHook(
    iter_dict= iter_dict,
    eval_steps= eval_interval,
    train_op= train_op,
    training= model.is_training,
    holder_handle= holder_handle,
    summary_conf= summary_conf,
    summary_protobuf= summary_protobuf,
  )
  # >>>  checkpoint saver
  ckpt_saver_hook = runhooks.CkptSaverHook(
    ckpt_dir,
    save_steps= save_ckpts_steps
  )
  all_hooks = [
      eval_hook,
      ckpt_saver_hook,
  ]
  # ////////////////////////////// session config //////////////////////////////
  sess_conf = tf.ConfigProto()
  sess_conf.gpu_options.allow_growth = True
  # sess_conf.gpu_options.per_process_gpu_memory_fraction = 0.9
  
  sess_creator = tf.train.ChiefSessionCreator(
      # scaffold=scaffold,
      # master='',
      config=sess_conf,
      checkpoint_dir=ckpt_dir
  )
  # ------------------------------  start  ------------------------------
  with tf.train.MonitoredSession(
    session_creator= sess_creator,
    hooks= all_hooks,
    stop_grace_period_secs= 3600
    ) as mon_sess:
    while not mon_sess.should_stop():
      step = mon_sess.run(tf.train.get_global_step()) # arg from retval of _EvalHook before_run()
      # training, step = mon_sess.run([model.is_training, tf.train.get_global_step()]) # arg from retval of _EvalHook before_run()
      # if not training:
        # print('step {}: eval xxxxxxxxx'.format(step))
      # print(lr)
  return

if __name__ == '__main__':
  main()
  # print(tf.__version__)