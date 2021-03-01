#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 3/1/2021-18:05
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : 07-data_process.py
# @Project  : PaddleTutorial
import paddle
from paddle.vision import Compose, ColorJitter, Resize

print("数据处理方法:", paddle.vision.transforms.__all__)
# 数据处理方法: ['BaseTransform', 'Compose', 'Resize', 'RandomResizedCrop', 'CenterCrop', 'RandomHorizontalFlip',
# 'RandomVerticalFlip', 'Transpose', 'Normalize', 'BrightnessTransform', 'SaturationTransform', 'ContrastTransform',
# 'HueTransform', 'ColorJitter', 'RandomCrop', 'Pad', 'RandomRotation', 'Grayscale', 'ToTensor', 'to_tensor', 'hflip',
# 'vflip', 'resize', 'pad', 'rotate', 'to_grayscale', 'crop', 'center_crop', 'adjust_brightness', 'adjust_contrast',
# 'adjust_hue', 'normalize']

# 以下方式随机调整图像的亮度、对比度、饱和度，并调整图像的大小，对图像的其他调整
# 定义要使用的数据增强方式,包括随机调整亮度,对比度和饱和度,改变图片大小
transform = Compose([ColorJitter(), Resize(size=32)])
# 通过transformer的参数传递定义好的数据增强方法即可完成对自带数据集的增强
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)

# 自定义数据集
