#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 3/1/2021-11:26
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : 01-broadcasting.py
# @Project  : PaddleTutorial

import paddle

# 飞桨的广播机制主要遵循如下规则（参考 Numpy 广播机制 ）：
#
# 每个张量至少为一维张量
# 从后往前比较张量的形状，当前维度的大小要么相等，要么其中一个等于一，要么其中一个不存在
x = paddle.ones((2, 3, 4))
y = paddle.ones((2, 3, 4))
# 两个张量 形状一致 可以广播

z = x + y
print(z)
# Tensor(shape=[2, 3, 4], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[[2., 2., 2., 2.],
#          [2., 2., 2., 2.],
#          [2., 2., 2., 2.]],
#
#         [[2., 2., 2., 2.],
#          [2., 2., 2., 2.],
#          [2., 2., 2., 2.]]])

print(z.shape)
# [2, 3, 4]

x = paddle.ones((2, 3, 1, 5))
y = paddle.ones((3, 4, 1))
z = x + y
print(z)

print(z.shape)
# [2, 3, 4, 5]
