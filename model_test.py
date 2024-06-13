import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import cv2
import data_set
# import test2
import CNN as cn
# train_data = torchvision.datasets.MNIST(
#     root='./data/',  # 保存或提取的位置  会放在当前文件夹中
#     train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
#     transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray

#     download=False,  # 已经下载了就不需要下载了
# )

# test_data = torchvision.datasets.MNIST(
#     root='./data/',
#     train=False  # 表明是测试集
# )

# datas = data_set.DataSet('./data')
# test_x = torch.unsqueeze(datas.test_data.train_data, dim=1).type(torch.FloatTensor)[:2000]
class Model_Test:

    def __init__(self, test_x, model_path)->None:
        self.inputs = test_x
        self.model = torch.load(model_path)
    def predict(self):
        cnn = cn.CNN()
        # cnn.load_state_dict(torch.load('cnn2.pkl'))
        cnn.load_state_dict(self.model)
        cnn.eval()
        # inputs = test_x
        test_output = cnn(self.inputs)
        pred_y = torch.max(test_output, 1)[1].data.numpy()

        print(pred_y, 'prediction number')

        self.pred_y = pred_y
        img = torchvision.utils.make_grid(self.inputs)
        img = img.numpy().transpose(1, 2, 0)

        cv2.imshow('win', img)
        key_predded = cv2.waitKey(0)


    def judge(self, test_y):
        pred_y = self.pred_y

        count = 0

        sum = float(len(test_y) + 0.0)
        print(sum)

        for i in range(len(test_y)):
            if(test_y[i] == pred_y[i]):
                count += 1
        
        print(count / sum)
# model_test = Model_Test(test_x=test_x[:60], model_path='cnn2.pkl')

# model_test.predict()
