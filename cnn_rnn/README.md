CNN 提取特征形成序列输入到 RNN 中

---
# 1 CNN

## .1 TRAIN
TODO

## .2 DEPLOY
TODO

# 2 RNN
## 输入
以 3conv 2fc 结构的 CNN 为例，分类网络训练好之后，取出倒数第2层FC的输出向量，`shape=(batch, fc_out)`  
以 `batch` 为序列长度, 给这个序列标上 label，形成 RNN 的一个训练样本 
`shape=(max_steps, fc_out)` ,   
因为一个样本体积不大(batch*fc_out), 所以将每个样本保存为 `.npy` 文件，文件名可以用对应的 `label`

输入时将多个 npy 文件组合成 batch 对RNN进行训练

**注意**
- 连续帧
  >RNN 需要的是前后相关的向量序列，也就是`CNN`的输入必须是一段视频的连续帧，而不能是训练时打乱图像组合出的batch
- 错误样本
  > 分类网络准确率不可能100%，错误样本是否要去除。另外错误样本还可能是连续帧中的某几帧
- mem 不足
  > 如果按 `batch = max_steps` 每次生成一个序列，在 max_steps 很大，同时网络很深时，大batch会导致CNN需要很大的mem，可以按 `n * batch = max_steps` 组合多个 batch

## 模型
TODO

# 3 整合
如果不要求同时训练两个网络部分，直接拼接 CNN 和 RNN 要考虑的问题就不多。
流程如下
- CNN 的输入
  > cv2 读取视频，按 RNN 需要的序列长度切割成若干块，每块 `shape = (max_steps, 3, h, w)`
- CNN 提取特征
  > 每块通过分类网络输出特征向量 `shape = (max_steps, fc_out)`, 即为一个样本
- RNN
  > 将上面的得到的样本输入 RNN ，输出预测的趋势
