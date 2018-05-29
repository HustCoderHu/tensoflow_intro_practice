import time
import sys
import os
import os.path as path
from os.path import join as pj

import tensorflow as tf

import rnn_dataset_tfrecord
import rnn
cwd = r'E:\github_repo\tensorflow_intro_practice'
# cwd = r'/home/hzx/tensorflow_intro_practice'
sys.path.append(pj(cwd, 'proj', 'needle_mushroom'))
import runhooks

record_prefix= r'D:\Lab408\cnn_rnn\seqlen-32-step-1.tfrecord'
# record_prefix= r'/home/hzx/all_data/seqlen-32-step-1.tfrecord'
train_record = record_prefix + '.train'
eval_record = record_prefix + '.eval'
seqLen = 32
vecSize = 128

log_dir = r'D:\Lab408\cnn_rnn\monsess_log-0502-hidden10'
log_dir = r'D:\Lab408\cnn_rnn\monsess_log-0502-hidden20'
# log_dir = r'/home/hzx/tensorflow_intro_practice/cnn_rnn/monsess_log'
ckpt_dir = pj(log_dir, 'ckpts')

eval_interval = 20

save_summary_steps = 5
save_ckpts_steps = 100
train_batchsz = 50
eval_batchsz = 50
# eval_steps = 40
epoch = 900
img_num = 870 * 2
max_steps = (img_num * epoch) # train_batchsz

def main():
  if not path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
  # ------------------------------ prepare input ------------------------------
  dset = rnn_dataset_tfrecord.MyDataset(train_record, eval_record)
#   dset = rnn_dataset_tfrecord.MyDataset(train_record, eval_record)
  prefetch_batch = None
  iter_dict = {
    'train': dset.train_iter(train_batchsz, prefetch_batch),
    'eval': dset.train_iter(eval_batchsz, prefetch_batch)
  }
  # train_iter = dset.train(train_batchsz, prefetch_batch)
  # eval_iter = dset.eval(eval_batchsz, prefetch_batch)
  # dict_tsr_handle = {
    # 'train': train_iter.string_handle(),
    # 'eval': eval_iter.string_handle()
  # }

  holder_handle = tf.placeholder(tf.string, [])
  iter = tf.data.Iterator.from_string_handle(
      holder_handle, dset.output_types)
  # next_elem = iter.get_next()
  seqs, seq_shape, labels = iter.get_next()
  # print(seqs.shape)
  # inputx = seqs
  inputx = tf.reshape(seqs, [train_batchsz, seqLen, vecSize])
  # print(inputx.shape)
#   return
#   inputx = tf.reshape(inputx, [-1, 200, 250, 3])
  # eval_x.set_shape([eval_batchsz, 200, 250, 3])

  # train_x, train_y, train_fname = dset.train(train_batchsz, prefetch_batch)
  # train_x.set_shape([train_batchsz, 200, 250, 3])
  # eval_x, eval_y, eval_fname = dset.eval(eval_batchsz, prefetch_batch)
  # eval_x.set_shape([eval_batchsz, 200, 250, 3])

  # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ build graph \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  model = rnn.RNN('GRU', 10, 2)
  # model = smodel.Simplest('NHWC')
  logits = model(inputx, train_batchsz)
  with tf.name_scope('cross_entropy'):
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    
  with tf.name_scope('accuracy'):
    _pred = tf.argmax(logits, axis=1, output_type=tf.int32)
    _pred = tf.cast(_pred, tf.int64)
    acc_vec = tf.equal(labels, _pred)
    acc = tf.reduce_mean(tf.cast(acc_vec, tf.float32))
    
  with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
  
  # ||||||||||||||||||||||||||||||  hooks ||||||||||||||||||||||||||||||
  # >>>  logging
  tf.logging.set_verbosity(tf.logging.INFO)
  # global_step = tf.train.get_or_create_global_step()
  # tf.identity(global_step, 'g_step')
  # tf.identity(loss, 'cross_entropy')
  # tf.identity(acc, 'accuracy')
  # tensor_lr = optimizer._lr_t

  tensors = {
    'step': tf.train.get_or_create_global_step().name,
    'loss': loss.name,
    'accuracy': acc.name
  }
  logging_hook = tf.train.LoggingTensorHook(
    tensors= tensors,
    every_n_iter=10
  )

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
  # ckpt_saver_hook = tf.train.CheckpointSaverHook(
  #   checkpoint_dir= ckpt_dir,
  #   save_steps= save_ckpts_steps,
  # )

  all_hooks = [
      # logging_hook,
      # summary_hook,
      eval_hook,
      ckpt_saver_hook,
      # tf.train.StopAtStepHook(max_steps),
      # tf.train.NanTensorHook(loss)
  ]

  # ////////////////////////////// session config //////////////////////////////
  sess_conf = tf.ConfigProto()
  sess_conf.gpu_options.allow_growth = True
  sess_conf.gpu_options.per_process_gpu_memory_fraction = 0.9
  
  sess_creator = tf.train.ChiefSessionCreator(
      # scaffold=scaffold,
      # master='',
      config=sess_conf,
      checkpoint_dir=ckpt_dir
  )
  print('end')
#   return
  # ------------------------------  start  ------------------------------
  with tf.train.MonitoredSession(session_creator= sess_creator,
      hooks= all_hooks, stop_grace_period_secs= 3600) as mon_sess:
    while not mon_sess.should_stop():
      step = mon_sess.run(tf.train.get_global_step()) # arg from retval of _EvalHook before_run()
      # training, step = mon_sess.run([model.is_training, tf.train.get_global_step()]) # arg from retval of _EvalHook before_run()
      # if not training:
        # print('step {}: eval xxxxxxxxx'.format(step))
      # print(lr)
  return

if __name__ == '__main__':
  main()

