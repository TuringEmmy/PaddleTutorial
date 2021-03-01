#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 3/1/2021-18:12
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : 08-network.py
# @Project  : PaddleTutorial
import paddle

print("飞浆内置模型:", paddle.vision.models.__all__)
# 飞浆内置模型: ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'VGG', 'vgg11', 'vgg13', 'vgg16',
#          'vgg19', 'MobileNetV1', 'mobilenet_v1', 'MobileNetV2', 'mobilenet_v2', 'LeNet']

lenet = paddle.vision.models.LeNet()
paddle.summary(lenet, (64, 1, 28, 28))

# ---------------------------------------------------------------------------
#  Layer (type)       Input Shape          Output Shape         Param #
# ===========================================================================
#    Conv2D-1      [[64, 1, 28, 28]]     [64, 6, 28, 28]          60
#     ReLU-1       [[64, 6, 28, 28]]     [64, 6, 28, 28]           0
#   MaxPool2D-1    [[64, 6, 28, 28]]     [64, 6, 14, 14]           0
#    Conv2D-2      [[64, 6, 14, 14]]     [64, 16, 10, 10]        2,416
#     ReLU-2       [[64, 16, 10, 10]]    [64, 16, 10, 10]          0
#   MaxPool2D-2    [[64, 16, 10, 10]]     [64, 16, 5, 5]           0
#    Linear-1         [[64, 400]]           [64, 120]           48,120
#    Linear-2         [[64, 120]]            [64, 84]           10,164
#    Linear-3          [[64, 84]]            [64, 10]             850
# ===========================================================================
# Total params: 61,610
# Trainable params: 61,610
# Non-trainable params: 0
# ---------------------------------------------------------------------------
# Input size (MB): 0.19
# Forward/backward pass size (MB): 7.03
# Params size (MB): 0.24
# Estimated Total Size (MB): 7.46
# ---------------------------------------------------------------------------
#
#
# Process finished with exit code 0


# Sequential形式组网

mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)
paddle.summary(mnist, (64, 1, 28, 28))
# ---------------------------------------------------------------------------
#  Layer (type)       Input Shape          Output Shape         Param #
# ===========================================================================
#    Flatten-2     [[64, 1, 28, 28]]        [64, 784]              0
#    Linear-4         [[64, 784]]           [64, 512]           401,920
#     ReLU-3          [[64, 512]]           [64, 512]              0
#    Dropout-1        [[64, 512]]           [64, 512]              0
#    Linear-5         [[64, 512]]            [64, 10]            5,130
# ===========================================================================
# Total params: 407,050
# Trainable params: 407,050
# Non-trainable params: 0
# ---------------------------------------------------------------------------
# Input size (MB): 0.19
# Forward/backward pass size (MB): 1.14
# Params size (MB): 1.55
# Estimated Total Size (MB): 2.88
# ---------------------------------------------------------------------------

# Layer类继承方式组网
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


mnist_2 = Mnist()
paddle.summary(mnist_2, (64, 1, 28, 28))
# ---------------------------------------------------------------------------
#  Layer (type)       Input Shape          Output Shape         Param #
# ===========================================================================
#    Flatten-2     [[64, 1, 28, 28]]        [64, 784]              0
#    Linear-4         [[64, 784]]           [64, 512]           401,920
#     ReLU-3          [[64, 512]]           [64, 512]              0
#    Dropout-1        [[64, 512]]           [64, 512]              0
#    Linear-5         [[64, 512]]            [64, 10]            5,130
# ===========================================================================
# Total params: 407,050
# Trainable params: 407,050
# Non-trainable params: 0
# ---------------------------------------------------------------------------
# Input size (MB): 0.19
# Forward/backward pass size (MB): 1.14
# Params size (MB): 1.55
# Estimated Total Size (MB): 2.88
# ---------------------------------------------------------------------------
