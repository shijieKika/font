# TO DO

## 零、工程
1. 完善模型的save、restore等逻辑
2. tensorboard 的使用
3. gpu 使用设置，tensorflow显存使用机制，显存最大占用率
4. 测试最大batch
5. 预生成剪裁后的文件放到tmp里，后续直接读取即可

## 一、数据预处理
1. 八方向梯度特征 sobel算子
2. 尝试不同分辨率：(64, 64), (96, 96)等
3. 扩展数据

## 二、网络结构
1. 增大cnn深度与宽度
2. network in network
3. 调整 dropout
4. 没有全连接层时如何dropout

## 三、训练
1. 利用mnist进行预训练获得初识参数
2. 损失函数增加正则项
3. 调整 batch 加速 gpu 上的训练
4. 尝试不同优化器

## Done
1. 将所有 Op 抽象出类，使得 train、inference、test 等过程与main分离
2. 整合 train.py 与 load_image.py 参数