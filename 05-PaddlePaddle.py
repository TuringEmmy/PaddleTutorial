#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 3/1/2021-17:20
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : 05-PaddlePaddle.py
# @Project  : PaddleTutorial
import paddle
from paddle.vision import ToTensor

print(paddle.__version__)

# 手写数字识别
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
val_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())

# 组网
mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)

# 训练
# 预计模型结构生成模型对象, 便于进行后续的配置\训练和验证
model = paddle.Model(mnist)

# 模型训练先关配置,准备损失计算方法,优化器和精度计算方法
model.prepare(paddle.optimizer.Adam(parameters=model.parameters()), paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

# 准备模型训练
model.fit(train_dataset, epochs=54, batch_size=64, verbose=1)

# 模型评估
model.evaluate(val_dataset, verbose=0)
# Epoch 50/54
# step 938/938 [==============================] - loss: 0.0057 - acc: 0.9980 - 33ms/step
# Epoch 51/54
# step 938/938 [==============================] - loss: 4.1997e-04 - acc: 0.9979 - 32ms/step
# Epoch 52/54
# step 938/938 [==============================] - loss: 0.0011 - acc: 0.9984 - 31ms/step
# Epoch 53/54
# step 938/938 [==============================] - loss: 4.8328e-05 - acc: 0.9984 - 31ms/step
# Epoch 54/54
# step 938/938 [==============================] - loss: 3.7216e-06 - acc: 0.9983 - 32ms/step
