import model_test as mt
import data_set as ds
import train_step as ts

if __name__=="__main__":

    # 读取数据集
    data_set = ds.DataSet('./data')

    test_x = data_set.test_x

    test_y = data_set.test_y
    train_loader = data_set.dataLoader()

    # 训练模型
    # train_model = ts.Model_Train(test_x=test_x, test_y=test_y, train_loader=train_loader)
    
    # 测试模型
    model_test = mt.Model_Test(test_x=test_x[-100:-1], model_path='cnn2.pkl')
    # 开始预测
    model_test.predict()
    model_test.judge(test_y=test_y[-100:-1])


