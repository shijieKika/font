# TO DO

## 零、工程
1. 完善模型的save、restore等逻辑
3. train accuracy,loss in all data *
5. inference *
6. train, valid, test *
7. sec/batch *
8. ***build graph***

## 一、数据预处理
1. 八方向梯度特征 sobel算子
2. 尝试不同分辨率：(64, 64), (96, 96)等
3. 数据增广 random crop *

## 二、网络结构
1. network in network
3. 调整 dropout
4. ResNet, Inception
6. ***Batch Normalization***

## 三、训练
1. 利用mnist进行预训练获得初识参数
2. 损失函数增加正则项
5. 负例 -1 loss ? *


## Done
1. 将所有 Op 抽象出类，使得 train、inference、test 等过程与main分离
2. 整合 train.py 与 load_image.py 参数
3. 文件直接加到内存
4. gpu 使用设置，tensorflow显存使用机制，显存最大占用率
5. 减少pooling数量，卷积3 * 3
6. 每个epoch做一次accuracy，平时只打印 loss
7. 每epoch后shuffle
6. 准确率 正例、负例都进行统计
7. accuracy in tf
6. learning rate decay
