import numpy as np
import pandas as pd
from sklearn import preprocessing, decomposition, neighbors, svm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Titanic(object):
    """
    用于处理Titanic数据集的类
    """
    def __init__(self,
            method='mean',
            nondimen='scale',
            pca=True,
            n_components='mle',
            svd_solver='full',
            weights=[0.5,0.25,0.25],
            model='knn',
            times=10,
            echos=2):
        # 私有属性
        # 累计正确率
        self.accuracy_validation=[]
        self.accuracy_test=[]

        # 操作函数    
        # 数据采集
        self.data_collection()
        # 数据清洗
        self.data_cleaning()
        # 特征提取
        self.feature_extraction()
        # 缺失值计算
        self.missing_value_calculation(method=method)
        # 数据预处理
        self.data_preprocessing(nondimen=nondimen)
        # 特征选择
        self.feature_selection(pca=pca, n_components=n_components, svd_solver=svd_solver)
        # 采样
        self.sampling(weights=weights)
        # 建立模型
        if pca:
            if n_components == 'mle':
                nn_n = self.train.shape[1]
            else:    
                nn_n = n_components
            self.modeling(model=model, times=times, nn_n=nn_n, echos=echos)
        else:
            nn_n = False
            self.modeling(model=model, times=times, nn_n=nn_n, echos=echos)
    
    """数据采集"""
    def data_collection(self):
        # 获取训练数据
        self.train = pd.read_csv('train.csv')
        # 获取检验数据
        self.test = pd.read_csv('test.csv')
        # 获取真实数据
        self.real = pd.read_csv('gender_submission.csv')

    """数据清洗"""
    def data_cleaning(self):
        pass
        
    """采样"""
    def sampling(self, weights=[0.5,0.25,0.25]):
        # 抽取训练集
        # 统一随机因子
        random_state=np.random.RandomState(np.random.randint(2**32))
        self.train_set = self.train.sample(frac=weights[0], random_state=random_state)
        # 映射训练集与实际结果
        dataset = pd.concat([self.train_set,self.train_survived],axis=1).dropna()
        self.train_set = dataset.iloc[:,0:-1]
        self.train_set_survived = dataset.loc[:,['Survived']]
        # 抽取验证集
        random_state=np.random.RandomState(np.random.randint(2**32))
        self.validation_set = self.train.sample(frac=weights[1], random_state=random_state)
        # 映射验证集与实际结果
        dataset = pd.concat([self.validation_set,self.train_survived],axis=1).dropna()
        self.validation_set = dataset.iloc[:,0:-1]
        self.validation_set_survived = dataset.loc[:,['Survived']]
        # 抽取测试集
        random_state=np.random.RandomState(np.random.randint(2**32))
        self.test_set = self.train.sample(frac=weights[2], random_state=random_state)
        # 映射测试集与实际结果
        dataset = pd.concat([self.test_set,self.train_survived],axis=1).dropna()
        self.test_set = dataset.iloc[:,0:-1]
        self.test_set_survived = dataset.loc[:,['Survived']]

    """特征提取"""
    def feature_extraction(self):
        pass

    """缺失值计算"""
    def missing_value_calculation(self, method='mean'):
        # 定义缺失值计算函数
        # 剔除数据
        def drop(data):
            return data.dropna()
        # 均值填充
        def mean(data):
            return data.fillna(data.mean())
        # 中位数填充
        def median(data):
            return data.fillna(data.median())
        # 插值法补充    
        def inter(data):
            return data.interpolate()
        # 缺失值计算函数的字典映射
        switcher = {
            'drop':drop,  # 对于检验数据处理会有bug
            'mean':mean,
            'median':median,
            'inter':inter
        }
        # 获取缺失值计算函数
        missing_value = switcher.get(method)
        # 处理训练数据的缺失值
        self.train = missing_value(self.train)
        # 处理检验数据的缺失值
        self.test = missing_value(self.test)

    """数据预处理"""
    def data_preprocessing(self, nondimen='scale'):
        # 对定量特征二值化
        # 对定性特征亚编码
        # 定义处理函数
        # 性别
        def sex(data):
            male = data.Sex.map(lambda x:int(x == 'male')).rename('Male')
            female = data.Sex.map(lambda x:int(x == 'female')).rename('Female')
            return pd.concat([male,female],axis=1)
        # 登船点
        def embarked(data):
            s = data.Embarked.map(lambda x:int(x == 'S')).rename('S')
            c = data.Embarked.map(lambda x:int(x == 'C')).rename('C')
            q = data.Embarked.map(lambda x:int(x == 'Q')).rename('Q')
            return pd.concat([s,c,q],axis=1)
        # 无量纲化
        # 标准化数据
        def scale(data):
            age = pd.DataFrame(preprocessing.scale(data.Age),columns=['Age'])
            fare = pd.DataFrame(preprocessing.scale(data.Fare),columns=['Fare'])
            return pd.concat([age,fare],axis=1)
        # 归一化数据
        def normalize(data):
            age = (data.Age - np.min(data.Age)) / (np.max(data.Age) - np.min(data.Age))
            fare = (data.Fare - np.min(data.Fare)) / (np.max(data.Fare) - np.min(data.Fare))
            return pd.concat([age,fare],axis=1)
        switcher = {
            'scale':scale,
            'normalize':normalize
        }
        nondimen = switcher.get(nondimen)
        # 数据变换
        # 定义数据变换函数
        def data_reshape(data):
            return pd.concat([sex(data),
                            embarked(data),
                            nondimen(data),
                            data.Pclass,
                            data.SibSp,
                            data.Parch],axis=1)
        # 数据变换
        self.train_survived = self.train.loc[:,['Survived']]
        self.train = data_reshape(self.train)
        self.test_survived = self.real.loc[:,['Survived']]
        self.test = data_reshape(self.test)

    """特征选择"""
    def feature_selection(self, pca=True, n_components='mle', svd_solver='full'):  # n_components='mle'有问题
        # 降维
        # PCA
        if pca:
            pca = decomposition.PCA(n_components=n_components, svd_solver=svd_solver)
            pca.fit(self.train)
            self.train = pd.DataFrame(pca.transform(self.train))
            self.test = pd.DataFrame(pca.transform(self.test))

    """建立模型"""
    def modeling(self, model='knn', times=10, echos=2, nn_n=False):
        # 定义子流程函数
        # 准备工作
        def data_preparing(data):
            return np.array(data).squeeze()
        # 建立模型
        def model_building(kernel='knn'):
            # 核函数构建
            # knn
            def knn_kernel():  # 添加参数控制
                return neighbors.KNeighborsClassifier(
                            weights='distance',
                            # algorithm='kd_tree',
                            # p=1,
                            )
            # svm
            def svm_kernel():
                return svm.SVC()
            # 选择器
            switcher = {
                'knn':knn_kernel,
                'svm':svm_kernel
            }
            kernel = switcher.get(kernel)
            return kernel()
        # 训练模型
        def model_training(clf, times):
            # alert
            print('#'*12+'训练模型'+'#'*12+'\n')
            # 重复训练
            for i in range(times):
                print('#'*12+'第 %03d 次'%(i+1)+'#'*12+'\n')
                # 采样
                self.sampling()
                # 训练模型
                clf.fit(
                    self.train_set,
                    data_preparing(self.train_set_survived)
                    )
                # 预测模型
                model_prediction(
                    clf,
                    self.validation_set,
                    data_preparing(self.validation_set_survived),
                    label=False
                    )
                # 检验模型
                model_testing(
                    clf,
                    self.test_set,
                    data_preparing(self.test_set_survived),
                    label=False
                )
            return clf
        # 预测模型
        def model_prediction(clf, validation, validation_survived, label=True):
            if label:
                print('#'*12+'预测模型'+'#'*12+'\n')
            validation = clf.predict(validation)
            accuracy_validation = (validation == validation_survived).astype(int).mean()
            print('验证集准确率:%f\n'%(accuracy_validation))
            # 保存历史正确率
            self.accuracy_validation.append(accuracy_validation)
        # 检验模型
        def model_testing(clf, test, test_survived, label=True):
            if label:    
                print('#'*12+'检验模型'+'#'*12+'\n')
            test = data_preparing(test)
            test = clf.predict(test)
            accuracy_test = (test == test_survived).astype(int).mean()
            print('测试集准确率:%f\n'%(accuracy_test))
            # 保存历史正确率
            self.accuracy_test.append(accuracy_test)
        # 训练结果
        def train_result():
            print('#'*12+'训练结果'+'#'*12+'\n')
            self.accuracy_validation = np.mean(self.accuracy_validation)
            self.accuracy_test = np.mean(self.accuracy_test)
            print('验证集平均正确率:%f\n'%(self.accuracy_validation))
            print('测试集平均正确率:%f\n'%(self.accuracy_test))
        # 实际检验结果
        def real_result(clf):
            print('#'*12+'实际结果'+'#'*12+'\n')
            # 数据变换
            self.test_survived = np.array(self.test_survived).squeeze()
            # 预测
            test_predict = clf.predict(self.test)
            # 计算正确率
            self.model_accuracy = (test_predict == self.test_survived).astype(int).mean()
            # 打印正确率
            print('实际正确率:%f\n'%(self.model_accuracy))
        # 模型函数实现
        # KNN
        def KNN(nn_n, times=10, echos=2):
            # 说明模型
            print('#'*12+'  K-NN  '+'#'*12+'\n')
            # 建立模型
            self.kn_clf = model_building(kernel='knn')
            # 训练模型
            self.kn_clf = model_training(self.kn_clf, times)
            # 训练结果
            train_result()
            # 实际检验结果
            real_result(self.kn_clf)

        # SVM
        def SVM(nn_n, times=10, echos=2):
            # 说明模型
            print('#'*12+'  SVM  '+'#'*12+'\n')
            # 建立模型
            self.svm_clf = model_building(kernel='svm')
            # 训练模型
            self.svm_clf = model_training(self.svm_clf,times)
            # 训练结果
            train_result()
            # 实际检验结果
            real_result(self.svm_clf)
        # NN
        def NN(nn_n, times=10, echos=2):
            # 说明模型
            print('#'*12+'  NN  '+'#'*12+'\n')
            # 建立模型
            # 建立神经网络
            # 定义类
            class Net(nn.Module):
                def __init__(self, n_feature):
                    super(Net, self).__init__()
                    self.f1 = nn.Linear(n_feature, 16)
                    self.d1 = nn.Dropout(p=0.5)
                    self.f2 = nn.Linear(16, 16)
                    self.d2 = nn.Dropout(p=0.5)
                    self.f3 = nn.Linear(16, 1)

                def forward(self, data):
                    data = self.f1(data)
                    data = self.d1(data)
                    # data = F.sigmoid(data)
                    data = F.relu(data)
                    data = self.f2(data)
                    data = self.d2(data)
                    # data = F.sigmoid(data)
                    data = F.relu(data)
                    data = self.f3(data)
                    return data
            # 输入层数
            if nn_n:
                n = nn_n
            else:
                n = 10                
            # 实例化神经网络
            self.net = Net(n)
            # 定义优化算法,损失函数
            # SGD
            # optimizer = torch.optim.SGD(self.net.parameters(), lr=0.5)
            # Adam
            optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01, weight_decay=0.001)
            loss_function = torch.nn.MSELoss()
            # 训练
            # alert
            print('#'*12+'训练模型'+'#'*12+'\n')
            for echo in range(echos):
                # 采样
                self.sampling()
                # 处理数据
                # Tensor化
                train_set = torch.from_numpy(np.array(self.train_set, dtype='float32'))
                train_set_survived = torch.from_numpy(np.array(self.train_set_survived, dtype='float32'))
                # variable化
                train_set, train_set_survived = Variable(train_set, requires_grad=False), Variable(train_set_survived, requires_grad=False)
                # 训练
                for time in range(times):
                    prediction = self.net(train_set)
                    loss = loss_function(prediction, train_set_survived)
                    if (time+1)%1000==0:
                        # alert
                        print('#'*8+'第 %02d 轮 第 %03d 次'%(echo+1, time+1)+'#'*8+'\n')
                        # 计算损失值
                        print('损失值:%f\n'%(loss.data.numpy()))
                        # 计算准确率
                        predict = torch.round(prediction)
                        self.model_accuracy = np.mean((predict == train_set_survived).data.numpy())
                        print('准确率:%d'%(self.model_accuracy*100)+'%\n')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # 计算预测准确率
                print('#'*12+'预测准确率'+'#'*12+'\n')
                # 处理数据
                # Tensor化
                test = torch.from_numpy(np.array(self.test, dtype='float32'))
                test_survived = torch.from_numpy(np.array(self.test_survived, dtype='float32'))
                # variable化
                test, test_survived = Variable(test, requires_grad=False), Variable(test_survived, requires_grad=False)
                # 预测
                prediction = self.net(test)
                predict = torch.round(prediction)
                self.model_accuracy = np.mean((predict == test_survived).data.numpy())
                print('准确率:%d'%(self.model_accuracy*100)+'%\n')
        # else

        # 选择器    
        switcher = {
            'knn':KNN,
            'svm':SVM,
            'nn':NN
        }
        training = switcher.get(model)
        print('\n'+'-'*12+' 分割线 '+'-'*12+'\n')
        training(nn_n=nn_n , times=times, echos=echos)
