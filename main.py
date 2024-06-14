# author: Wangpeng
# date: 2024/06/13
# function: main function

import model_test as mt
import data_set as ds
import train_step as ts
import utils as us

if __name__=="__main__":

    # 读取数据集 用于训练个模型
    data_set1 = ds.DataSet('./data', samples=2000)
    data_set2 = ds.DataSet("./data", samples=3000)
    
    
    test_x1, test_y1, train_loader1 = data_set1.get_train_samples()# 获取训练样本

    test_x2, test_y2, train_loader2 = data_set2.get_train_samples()

    # 训练模型
    # train_model1 = ts.Model_Train(test_x=test_x1, test_y=test_y1, train_loader=train_loader1, model_name='./model/cnn1.pkl')
    # train_model2 = ts.Model_Train(test_x=test_x2, test_y=test_y2, train_loader=train_loader2, model_name='./model/cnn2.pkl')
    # train_model3 = ts.Model_Train(test_x=test_x2, test_y=test_y2, train_loader=train_loader2, model_name='./model/cnn3.pkl', epoch=15)
    
    # 测试模型


    new_x,new_y = data_set2.get_test_samples() # 获取测试样本

    numbers = eval(input("请输入测试数据数(<1000): "))
    model_test1 = mt.Model_Test(test_x=new_x[-numbers:-1], model_path='./model/cnn1.pkl')
    model_test2 = mt.Model_Test(test_x=new_x[-numbers:-1], model_path='./model/cnn2.pkl')
    model_test3 = mt.Model_Test(test_x=new_x[-numbers:-1], model_path='./model/cnn3.pkl')

    # 开始测试
    print("==========开始测试=================")

    model_test1.predict()
    model_test2.predict()
    model_test3.predict()
    print("=========模型准确度对比=============")
    print("模型1:")
    model_test1.judge(new_y[-numbers:-1])
    print("模型2:")
    model_test2.judge(new_y[-numbers:-1])
    print("模型3:")
    model_test3.judge(new_y[-numbers:-1])


    # new_x = data_set1.new_x
    # new_y = data_set1.new_y
    # model_test = mt.Model_Test(test_x=new_x[-100:-1], model_path='cnn2.pkl')
    # model_test.predict()
    # print()
    # model_test.judge(test_y=new_y[-100:-1])

    # model_test = mt.Model_Test(test_x=new_x[-100:-1], model_path='cnn1.pkl')

    # model_test.predict()
    # model_test.judge(test_y=new_y[-100:-1])
    # 开始预测
    


