RNN 训练集预处理

---
CNN 的前向属于计算密集型任务，尽量只做一次。  
batch 组合成序列没有什么计算量，需要某种长度的序列时处理一下就好

# 1帧1向量
因为每种类别的样本都需要，所以将所有图像经过CNN前向一次，将向量保存成如下结构，这个过程没有用到标签。
```
/video01
  000000.npy
  000001.npy
  000002.npy
  ..
/video02
  000000.npy
  000001.npy
  000002.npy
  ..
```
每帧图像对应1个 npy 文件
执行代码是 `RNNdataPrepare/genAllFeatureVec_noncv2.npy`

计算思路没有问题，但是保存有问题，下面细说

# 连续向量的npy
对于77个视频的所有帧，每个向量.npy大小 640 Bytes，上面1帧1向量将产生 13w+的小文件，处理效率非常低，  
所以考虑将一个视频的所有 npy 组合成一个

`RNNdataPrepare/genAllFeatureVec_noncv_Dataset.py` 相比上面的 genxx，输入使用了`Dataset API`   
每次前向不再是单张jpg，而是可以在启动前设置batch大小，最终保存也是整个视频的所有帧生成的特征向量 `shape=(帧数量, fc_out)`  
所以77个视频就会有77个 npy 文件

tf对gpu mem的占用会一直持续到进程结束，所以代码里每次处理一个视频都单独启动一个进程

每个视频计算量都比较大，gpu 计算已经瓶颈，同时开启多个进程没有意义

# 组合向量
根据需要的序列长度对 npy 切片。以16为例，从起火帧 51 到 66 形成 rnn 的一个训练样本，这个过程要用到标签
```
/raging
  /video01
    000000-000015.npy
    000051-000066.npy
    ..
  /video02
   ..
   ..
/invariant
/weaking
```

分三类
- raging 火势变大
- invariant 不变
- weaking 变小

考虑到读取的性能，将样本做成 tfrecord 格式，原因如下
- 大量小文件读取效率低
  > 如果一个序列一个文件，每个样本大小 `seqLen*fc_out* sizeof(float)` 不到100K，样本数量在 10w 左右，效率低
- 总体积不大
  > 序列步长为1时，估计体积在1000M ~ 2000M ，制作耗时不大, 训练时可以充分利用 tfrecord 格式的优势

文件 `buildRNN_tfrecord.py` 用来制作 record，过程大致分成两步
- 切片
  > 读取 npy，根据序列长度, 滑动步长，切出 n 个序列，作为 n 个样本，用进程池应对视频很多的情况。  
  最后将所有样本list 拼接成一个
- 写入 tfrecord
  > 对应函数 buildRecord(exampleList, recordFile)  
  先打乱上面的 list，然后写入每个样本  
  测试发现，序列长度32, 步长1时，会有82115个序列，生成record时间要20分钟左右，期间cpu占用35%上下，磁盘占用6%左右，推断瓶颈在单线程。  
  遂分离writer.write这个IO调用，前面生成TFexample的过程交给进程池
  最终得到 tfrecord 文件约 1.3G，用时不到5 min
  