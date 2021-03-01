#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 3/1/2021-15:28
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : 02-tensor.py
# @Project  : PaddleTutorial


import paddle

# 创建类似于vector的1-D tensor
ndim_1_tensor = paddle.to_tensor([2, 3, 4], dtype='float32')
print(ndim_1_tensor)
# Tensor(shape=[3], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [2., 3., 4.])

# 创建类似于2-D tensor, 其ndim为2
ndim_2_tensor = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
print(ndim_2_tensor)
