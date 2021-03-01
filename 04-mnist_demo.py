#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 3/1/2021-15:48
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : 04-mnist_demo.py
# @Project  : PaddleTutorial
import paddle

# 以sequential形式组网
from paddle.fluid.reader import DataLoader
from paddle.vision import ToTensor

mnist_seq = paddle.nn.Sequential(
    paddle.nn.Flatten(),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)


# print(mnist_seq.parameters())


# 以subclass形式组网
class Mnist(paddle.nn.Layer):
    def __init__(self):
        super(Mnist, self).__init__()
        self.flatten = paddle.nn.Flatten()
        self.linear_1 = paddle.nn.Linear(784, 512)
        self.linear_2 = paddle.nn.Linear(512, 10)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.2)

    def forward(self, inputs):
        y = self.flatten(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        return y


mnist_sub = Mnist()
# print(mnist_sub.parameters())

# 模型训练高阶API
# 增加了paddle.Model高层API，大部分任务可以使用此API用于简化训练、评估、预测类代码开发。注意区别Model和Net概念，Net是指继承paddle.nn.Layer
# 的网络结构；而Model是指持有一个Net对象，同时指定损失函数、优化算法、评估指标的可训练、评估、预测的实例。
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())
lenet = paddle.vision.models.LeNet()

# Mnist继承paddle.nn.Layer属于Net,model包含训练功能
model = paddle.Model(lenet)

# 设置训练模型所需的optimizer,loss,metric

model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()), paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy(topk=(1, 2)))

# 启动训练
# model.fit(train_dataset, epochs=2, batch_size=64, log_freq=200)
# 启动评估
# model.evaluate(test_dataset, log_freq=20, batch_size=64)

# step 200/938 - loss: 0.4143 - acc_top1: 0.8586 - acc_top2: 0.9277 - 27ms/step
# step 400/938 - loss: 0.0350 - acc_top1: 0.9048 - acc_top2: 0.9562 - 28ms/step
# step 600/938 - loss: 0.1024 - acc_top1: 0.9248 - acc_top2: 0.9670 - 30ms/step
# step 800/938 - loss: 0.0893 - acc_top1: 0.9363 - acc_top2: 0.9730 - 30ms/step
# step 938/938 - loss: 0.0733 - acc_top1: 0.9418 - acc_top2: 0.9758 - 30ms/step
# Epoch 2/2
# step 200/938 - loss: 0.1322 - acc_top1: 0.9744 - acc_top2: 0.9945 - 31ms/step
# step 400/938 - loss: 0.0566 - acc_top1: 0.9753 - acc_top2: 0.9938 - 30ms/step
# step 600/938 - loss: 0.0258 - acc_top1: 0.9760 - acc_top2: 0.9938 - 30ms/step
# step 800/938 - loss: 0.0168 - acc_top1: 0.9769 - acc_top2: 0.9942 - 30ms/step
# step 938/938 - loss: 0.0547 - acc_top1: 0.9774 - acc_top2: 0.9943 - 30ms/step
# Eval begin...
# The loss value printed in the log is the current batch, and the metric is the average value of previous step.
# step  20/157 - loss: 0.1644 - acc_top1: 0.9719 - acc_top2: 0.9977 - 27ms/step
# step  40/157 - loss: 0.0250 - acc_top1: 0.9738 - acc_top2: 0.9953 - 26ms/step
# step  60/157 - loss: 0.0754 - acc_top1: 0.9742 - acc_top2: 0.9940 - 25ms/step
# step  80/157 - loss: 0.0054 - acc_top1: 0.9730 - acc_top2: 0.9939 - 25ms/step
# step 100/157 - loss: 0.0056 - acc_top1: 0.9764 - acc_top2: 0.9952 - 25ms/step
# step 120/157 - loss: 0.0044 - acc_top1: 0.9785 - acc_top2: 0.9953 - 25ms/step
# step 140/157 - loss: 0.0011 - acc_top1: 0.9811 - acc_top2: 0.9960 - 25ms/step
# step 157/157 - loss: 6.8514e-04 - acc_top1: 0.9816 - acc_top2: 0.9960 - 25ms/step
# Eval samples: 10000
#
# Process finished with exit code 0

# 基础API
loss_fn = paddle.nn.CrossEntropyLoss()

# 加载训练集 batch_size 设置为64
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


def train():
    epochs = 2
    adam = paddle.optimizer.Adam(learning_rate=0.001, parameters=lenet.parameters())
    # 用Adam作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = lenet(x_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            loss = loss_fn(predicts, y_data)
            if batch_id % 100 == 0:
                print("epoch:{}, batch_id:{}, loss is:{}, acc is:{}".format(epoch, batch_id, loss, acc))

            adam.step()
            adam.clear_grad()


train()
