# TO DO

## 零、工程
1. 指定device *

## 一、数据预处理
1. 八方向梯度特征 sobel算子
2. 数据增广 random crop *
3. train, valid, test切分，划出一个新的字体

## 二、网络结构
1. network in network
2. 调整 dropout
3. ResNet, Inception
4. ***Batch Normalization***

## 三、训练
1. 损失函数增加正则项
2. 负例 使用 *
3. 统计负样本的数据，对负样本的数据在训练时进行数据增广


## Done
1. 将所有 Op 抽象出类，使得 train、inference、test 等过程与main分离
2. 整合 train.py 与 load_image.py 参数
3. 文件直接加到内存
4. gpu 使用设置，tensorflow显存使用机制，显存最大占用率
5. 减少pooling数量，卷积3 * 3
6. 每个epoch做一次accuracy，平时只打印 loss
7. 每epoch后shuffle
8. 准确率 正例、负例都进行统计
9. accuracy in tf
10. learning rate decay
11. build graph
12. 完善模型的save、restore等逻辑
13. delete global_step int
14. delete func save get_accuracy
15. inference