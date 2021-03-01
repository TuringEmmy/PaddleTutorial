#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 3/1/2021-15:33
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : 03-dataset_demo.py
# @Project  : PaddleTutorial
# 数据预处理
from paddle.io import Dataset


# 测试定义的数据集
class MyDataset(Dataset):
    """
    步骤一：继承paddle.io。Dataset类
    """

    def __init__(self, mode='train'):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        :param mode:
        """
        super(MyDataset, self).__init__()
        if mode == "train":
            self.data = [['traindata1', 'label1'], ['traindata2', 'label2'], ['traindata3', 'label3'],
                         ['traindata4', 'label4']]
        else:
            self.data = [['testdata1', 'label1'], ['testdata2', 'label2'], ['testdata3', 'label3'],
                         ['testdata4', 'label4']]

    def __getitem__(self, index):
        """
        步骤三:实现__getitem方法,定义指定index时如何获取数据,并返回单条数据(训练数据,对应标签)
        :param inde:
        :return:
        """
        data = self.data[index][0]
        label = self.data[index][1]
        return data, label

    def __len__(self):
        """
        步骤四:实现__len__方法,返回数据集总数目
        :return:
        """
        return len(self.data)


train_dataset = MyDataset(mode='train')
val_dataset = MyDataset(mode='test')
print("训练数据集")
for data, label in train_dataset:
    print(data, label)

print("验证数据集")
for data, label in val_dataset:
    print(data, label)