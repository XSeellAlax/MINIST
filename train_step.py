import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import cv2
import utils as us
import CNN as cn



class Model_Train:
    def __init__(self, test_x, test_y, train_loader) -> None:
        
        cnn = cn.CNN()
        print(cnn)
        # 优化器选择Adam
        optimizer = torch.optim.Adam(cnn.parameters(), lr= us.LR)
        # 损失函数
        loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted
        
        # 开始训练
        for epoch in range(us.EPOCH):
            for step, (b_x, b_y) in enumerate(train_loader):  # 分配batch data
                output = cnn(b_x)  # 先将数据放到cnn中计算output
                loss = loss_func(output, b_y)  # 输出和真实标签的loss，二者位置不可颠倒
                optimizer.zero_grad()  # 清除之前学到的梯度的参数
                loss.backward()  # 反向传播，计算梯度
                optimizer.step()  # 应用梯度

                if step % 50 == 0:
                    test_output = cnn(test_x)
                    pred_y = torch.max(test_output, 1)[1].data.numpy()
                    accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

        torch.save(cnn.state_dict(), 'cnn2.pkl')#保存模型

