# 数据集
# 数据预处理


import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import cv2
import utils as us

class DataSet(object):
    def __init__(self, _path):
        self.train_data = torchvision.datasets.MNIST(
            root= _path, # 保存位置
            train=True, # 
            transform=torchvision.transforms.ToTensor(), # 
            
            download=us.DOWNLOAD_MNIST,
        )

        self.test_data = torchvision.datasets.MNIST(
            root=_path,
            train=False # 表明是测试集
        )

        self.train_loader = self.dataLoader()
        self.get_test()
    def dataLoader(self):
        return Data.DataLoader(
            dataset=self.train_data,
            batch_size=us.BATCH_SIZE,
            shuffle=True
        )
    
    def get_test(self):
        self.test_x = torch.unsqueeze(self.test_data.train_data, dim=1).type(torch.FloatTensor)[:2000]
        self.test_y = test_y = self.test_data.test_labels[:2000]
        return (self.test_x, self.test_y)
    # def 