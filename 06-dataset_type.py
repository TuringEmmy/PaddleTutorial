#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 3/1/2021-17:47
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : 06-dataset_type.py
# @Project  : PaddleTutorial
import paddle
# 数据集接口会自动从远端下载数据集到本机缓存目录~/.cache/paddle/dataset
from paddle.fluid.dataloader import Dataset
from paddle.fluid.reader import DataLoader

print("视觉数据集:", paddle.vision.datasets.__all__)
print("语言数据集:", paddle.text.datasets.__all__)
# 视觉数据集: ['DatasetFolder', 'ImageFolder', 'MNIST', 'FashionMNIST', 'Flowers', 'Cifar10', 'Cifar100', 'VOC2012']
# 语言数据集: ['Conll05st', 'Imdb', 'Imikolov', 'Movielens', 'UCIHousing', 'WMT14', 'WMT16']
# 自定义数据集
BATCH_SIZE = 64
BATCH_NUM = 20

IMAGE_SIZE = (28, 28)
CLASS_NUM = 10


# 测试定义的数据集
class MyDataset(Dataset):
    """
    步骤一:继承paddld.io.Dataset类
    """

    def __init__(self, num_samples):
        """
        步骤二:实现构造函数,定义数据集大小
        """
        super(MyDataset, self).__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        """
        步骤三:实现__getitem__方法,定义指定index时如何获取数据,并返回单条数据(训练数据,对应的标签)
        :param index:
        :return:
        """
        data = paddle.uniform(IMAGE_SIZE, dtype='float32')
        label = paddle.randint(0, CLASS_NUM - 1, dtype='int64')
        return data, label

    def __len__(self):
        """
        步骤四:实现__len__方法,返回数据集总数目
        :return:
        """
        return self.num_samples


custom_dataset = MyDataset(BATCH_SIZE * BATCH_NUM)
print("自定义构造数据集:")
for data, label in custom_dataset:
    print(data.shape, label.shape)
    # [28, 28][1]
    break

# 数据加载
train_loader = DataLoader(custom_dataset, batch_size=BATCH_NUM, shuffle=True)
# 如果要加载内置数据集,将custom_dataset换为train_dataset即可
for batch_id, data in enumerate(train_loader()):
    x_data = data[0]
    y_data = data[1]
    # [28, 28][1]
    # [20, 28, 28][20, 1]

    print(x_data.shape, y_data.shape)
    break
