# basic
import sys
import getopt
# numpy&pandas
import numpy as np
import pandas as pd
# pyplot
import matplotlib.pyplot as plt
# sklearn
# Z-scores
from sklearn.preprocessing import scale
# 缺失值
from sklearn.preprocessing import Imputer
# 文本编号
# from sklearn.preprocessing import LabelEncoder
# 分类数据扩维
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
# 二值化
from sklearn.preprocessing import Binarizer
# 无量纲化
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# 特征选择
## 随机Lasso
from sklearn.linear_model import RandomizedLasso
## SelectFromModel
from sklearn.feature_selection import SelectFromModel
## 随机森林
from sklearn.ensemble import RandomForestRegressor 
# 降维
## PCA
# from sklearn.decomposition import PCA
## LDA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# 建模
## KNN
from sklearn.neighbors import KNeighborsClassifier
## SVC
from sklearn.svm import SVC 
## MLP
from sklearn.neural_network import MLPClassifier
# 交叉验证
from sklearn.cross_validation import train_test_split
# from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
# 度量
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
# 画图
# import matplotlib.pyplot as plt
# xgboost
import xgboost as xgb
# GridSearchCV
from sklearn.model_selection import GridSearchCV


# 定义Titanic类
class Titanic(object):
    """
    News: 
    ===========================================
    降维暂时失效,用随机Lasso代替,选择特征并降维
    ===========================================
    ```
    说明: 
    本类用于处理Titanic数据集
    参数:

    用法:
    01_开始:
        11_数据采集:
            类函数实现:data_collection
            参数:无
            说明:获得Titanic数据集的数据用于处理
            是否完成:是
        12_数据分析:
            类函数实现:data_analysing
            参数:无
            说明:分析数据
            是否完成:否
        13_数据清洗:
            类函数实现:data_cleaning
            参数:method
            说明:清洗数据
            是否完成:否
        14_采样:
            类函数实现:sampling
            参数:无
            说明:采样
            是否完成:否
        15_特征提取:
            类函数实现:feature_extraction
            参数:无
            说明:提取特征
            是否完成:是
    02_预处理:
        21_缺失值计算:
            类函数实现:missing_value_calculation
            参数:method
            说明:计算缺失值,method指定方法,默认'most_frequent'
            是否完成:是
        22_对定量特征二值化:
            类函数实现:binarization_processing
            参数:无
            说明:对'SibSp','Parch'进行二值化
            是否完成:是
        23_对定性特征哑编码:
            类函数实现:feature_coding
            参数:无
            说明:对'Pclass','Sex','Embarked'进行哑编码,对分类数据进行扩维
            是否完成:是
        24_数据变换:
            类函数实现:data_variation
            参数:无
            说明:变换数据
            是否完成:否
        25_无量纲化:
            类函数实现:nondimensionalize
            参数:无
            说明:对'Age','Fare'进行无量纲化
            是否完成:是
        26_特征选择:
            类函数实现:feature_selection
            参数:method
            说明:method选择算法,rl使用随机Lasso算法,rf使用随机森林算法
            是否完成:是
        27_降维:
            类函数实现:deminsionality_reduction
            参数:method,n_components,svd_solver
            说明:method决定降维方法,n_components决定聚类数目,svd_solver配合'mle'
            是否完成:否
    03_建立模型
        31_训练:
            类函数实现:training
            参数:model
            说明:model决定训练方法
            是否完成:否        
        32_预测
            类函数实现:predicting
            参数:无
            说明:预测
            是否完成:否
    流程:    
    --01_开始---------------
    | 11_数据采集          |
    | 12_数据分析          |
    | 13_数据清洗          |
    | 14_采样              |
    | 15_特征提取          |
    |-02_预处理------------|
    | 21_缺失值计算        |
    | 22_对定量特征二值化  |
    | 23_对定性特征哑编码  |
    | 24_数据变换          |
    | 25_无量纲化          |
    | 26_特征选择          |
    | 27_降维              |
    |-03_建立模型----------|
    | 31_训练              |
    | 32_预测              |
    | 33_评估              |
    --04_结束---------------
    ```
    """
    # 初始化
    def __init__(self,
                missing_method='most_frequent',
                select_method='xgbr',
                demin_method='lda',
                n_components=10,
                svd_solver='full',
                training_model='xgb'):
        self.data_collection()
        # self.data_cleaning()
        self.feature_extraction()
        self.missing_value_calculation(method=missing_method)
        self.binarization_processing()
        self.feature_coding()
        self.nondimensionalize()
        self.feature_selection(method=select_method)
        # self.deminsionality_reduction(method=demin_method, n_components=n_components, svd_solver=svd_solver)
        self.training(model=training_model)
        if training_model in ['knn', 'svm', 'mlp']:
            self.predicting()
    
    """数据采集"""
    def data_collection(self):
        """
        ```
        说明:采集数据
        参数:无
        用法:空
        是否完成:是
        ```
        """
        # 获取训练数据
        self.train = pd.read_csv('train.csv')
        # 获取检验数据
        self.test = pd.read_csv('test.csv')
        # 获取真实数据
        self.real = pd.read_csv('gender_submission.csv')
    """数据分析"""
    def data_analysing(self, method=None):
        """
        ```
        说明:分析数据
        参数:method
        用法:
        01_箱线图:
            子函数实现:box
            参数:无
            说明:箱线图分析异常值
            是否实现:是
        是否完成:否
        ```
        """
        # 其它
        def other():
            pass
        # 01_箱线图
        def box():
            plt.figure("箱线图")
            plt.subplot(221)
            self.train.loc[:,['Age']].boxplot()
            plt.subplot(222)
            self.train.loc[:,['SibSp']].boxplot()
            plt.subplot(223)
            self.train.loc[:,['Parch']].boxplot()
            plt.subplot(224)
            self.train.loc[:,['Fare']].boxplot()
            plt.show()
        # 选择器
        switcher = {
                'box':box
            }
        analyse = switcher.get(method, other)
        return analyse()
    """数据清洗"""
    def data_cleaning(self, method=None):
        """
        ```
        说明:数据清洗
        参数:method
        用法:空
        是否完成:否
        ```
        """
        # Z-scores 清除异常值
        # Age
        # 标准化
        Age_scaled = pd.DataFrame(scale(self.train.Age.dropna()), index=self.train.Age.dropna().index, columns=['Age_scaled'])
        # 标识异常值
        Age_scaled_drop = Age_scaled[Age_scaled>-2]
        Age_scaled_drop = Age_scaled_drop[Age_scaled_drop<2]
        Age_scaled_drop = Age_scaled_drop.isnull()
        # 剔除异常值
        self.train = pd.concat([self.train, Age_scaled_drop], axis=1)
        self.train = self.train[self.train.Age_scaled!=True]
        self.train = self.train.drop(['Age_scaled'], axis=1)
        # Fare
        # 标准化
        Fare_scaled = pd.DataFrame(scale(self.train.Fare.dropna()), index=self.train.Fare.dropna().index, columns=['Fare_scaled'])
        # 标识异常值
        Fare_scaled_drop = Fare_scaled[Fare_scaled>-2]
        Fare_scaled_drop = Fare_scaled_drop[Fare_scaled_drop<2]
        Fare_scaled_drop = Fare_scaled_drop.isnull()
        # 剔除异常值
        self.train = pd.concat([self.train, Fare_scaled_drop], axis=1)
        self.train = self.train[self.train.Fare_scaled!=True]
        self.train = self.train.drop(['Fare_scaled'], axis=1)
    """采样"""
    def sampling(self):
        pass
    """特征提取"""
    def feature_extraction(self):
        """
        ```
        说明:提取特征
        参数:无
        用法:空
        是否完成:是
        ```
        """
        # Fare
        self.test.Fare = self.test.Fare.fillna(self.test.Fare.mean())

        # Cabin
        self.train.Cabin = self.train.Cabin.fillna('Unknown')
        self.train.Cabin = self.train.Cabin.apply(lambda x:x[0])
        self.test.Cabin = self.test.Cabin.fillna('Unknown')
        self.test.Cabin = self.test.Cabin.apply(lambda x:x[0])

        # FamilySize
        self.train['FamilySize'] = self.train.SibSp + self.train.Parch + 1
        self.test['FamilySize'] = self.test.SibSp + self.test.Parch + 1
        # PersonalFare
        self.train['PersonalFare'] = self.train.Fare / self.train.FamilySize
        self.test['PersonalFare'] = self.test.Fare / self.test.FamilySize

        self.train_survived = self.train.loc[:,['Survived']]
        self.train = self.train.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1)
        self.test_survived = self.real.loc[:,['Survived']]
        self.test = self.test.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    """缺失值计算"""
    def missing_value_calculation(self, method='most_frequent'):
        """
        ```
        说明:缺失值计算
        参数:method
        用法:空
        是否完成:是
        ```
        """
        # age
        imputer = Imputer(strategy=method)
        self.train.loc[:,['Age']] = pd.DataFrame(imputer.fit_transform(self.train.loc[:,['Age']]), columns=['Age'])
        self.test.loc[:,['Age']] = pd.DataFrame(imputer.fit_transform(self.test.loc[:,['Age']]), columns=['Age'])
        self.test.loc[:,['Fare']] = pd.DataFrame(imputer.fit_transform(self.test.loc[:,['Fare']]), columns=['Fare'])
        # embarked
        self.train.dropna(inplace=True)
        self.train_survived = self.train_survived.loc[self.train.index,self.train_survived.columns]
    """对定量特征二值化"""
    def binarization_processing(self):
        """
        ```
        说明:对定量特征二值化
        参数:无
        用法:空
        是否完成:是
        ```
        """
        binarizer = Binarizer()
        self.train.loc[:,['SibSp']] = pd.DataFrame(binarizer.fit_transform(self.train.loc[:,['SibSp']]), index=self.train.index, columns=['SibSp'])
        self.train.loc[:,['Parch']] = pd.DataFrame(binarizer.fit_transform(self.train.loc[:,['Parch']]), index=self.train.index, columns=['Parch'])
        self.test.loc[:,['SibSp']] = pd.DataFrame(binarizer.fit_transform(self.test.loc[:,['SibSp']]), index=self.test.index, columns=['SibSp'])
        self.test.loc[:,['Parch']] = pd.DataFrame(binarizer.fit_transform(self.test.loc[:,['Parch']]), index=self.test.index, columns=['Parch'])
    """对定性特征哑编码"""
    def feature_coding(self):
        """
        ```
        说明:对定性特征哑编码
        参数:无
        用法:空
        是否完成:是
        ```
        """
        # 定义处理函数
        def processing(data):
            # pclass
            onehotencoder = OneHotEncoder(sparse=False)
            pclass = pd.DataFrame(onehotencoder.fit_transform(data.loc[:,['Pclass']]), index=data.index, columns=['Pclass_1', 'Pclass_2', 'Pclass_3'])
            data = data.drop(['Pclass'], axis=1)
            data = pd.concat([data, pclass], axis=1)
            # sex
            dictvectorizer = DictVectorizer(sparse=False)
            sex = pd.DataFrame(dictvectorizer.fit_transform(data.loc[:,['Sex']].to_dict(orient='records')), index=data.index, columns=['Female', 'Male'])
            data = data.drop(['Sex'], axis=1)
            data = pd.concat([data, sex], axis=1)
            # embarked
            embarked = pd.DataFrame(dictvectorizer.fit_transform(data.loc[:,['Embarked']].to_dict(orient='records')), index=data.index, columns=['Embarked_C', 'Embarked_Q', 'Embarked_S'])
            data = data.drop(['Embarked'], axis=1)
            data = pd.concat([data, embarked], axis=1)
            return data
        # 处理train
        self.train = processing(self.train)
        # Cabin
        dictvectorizer = DictVectorizer(sparse=False)
        cabin = pd.DataFrame(dictvectorizer.fit_transform(self.train.loc[:,['Cabin']].to_dict(orient='records')), index=self.train.index, columns=['Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_U'])
        self.train = self.train.drop(['Cabin'], axis=1)
        if True:
            self.train = pd.concat([self.train, cabin], axis=1)
            # Cabin_U
            self.train = self.train.drop(['Cabin_U'], axis=1)
        # 处理test
        self.test = processing(self.test)
        # Cabin
        dictvectorizer = DictVectorizer(sparse=False)
        cabin = pd.DataFrame(dictvectorizer.fit_transform(self.test.loc[:,['Cabin']].to_dict(orient='records')), index=self.test.index, columns=['Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_U'])
        self.test = self.test.drop(['Cabin'], axis=1)
        if True:
            self.test = pd.concat([self.test, cabin], axis=1)
            self.test.insert(self.test.shape[1] - 1, 'Cabin_T', pd.DataFrame(0,index=self.test.index, columns=['Cabin_T']))
            # Cabin_U
            self.test = self.test.drop(['Cabin_U'], axis=1)
    """数据变换"""
    def data_variation(self):
        pass
    """无量纲化"""
    def nondimensionalize(self):
        """
        ```
        说明:无量纲化
        参数:无
        用法:空
        是否完成:是
        ```
        """
        scaler = MinMaxScaler()
        self.train.loc[:,['Age']] = pd.DataFrame(scaler.fit_transform(self.train.loc[:,['Age']]), index=self.train.index, columns=['Age'])
        self.train.loc[:,['Fare']] = pd.DataFrame(scaler.fit_transform(self.train.loc[:,['Fare']]), index=self.train.index, columns=['Fare'])
        self.train.loc[:,['FamilySize']] = pd.DataFrame(scaler.fit_transform(self.train.loc[:,['FamilySize']]), index=self.train.index, columns=['FamilySize'])
        self.train.loc[:,['PersonalFare']] = pd.DataFrame(scaler.fit_transform(self.train.loc[:,['PersonalFare']]), index=self.train.index, columns=['PersonalFare'])
        self.test.loc[:,['Age']] = pd.DataFrame(scaler.fit_transform(self.test.loc[:,['Age']]), index=self.test.index, columns=['Age'])
        self.test.loc[:,['Fare']] = pd.DataFrame(scaler.fit_transform(self.test.loc[:,['Fare']]), index=self.test.index, columns=['Fare'])
        self.test.loc[:,['FamilySize']] = pd.DataFrame(scaler.fit_transform(self.test.loc[:,['FamilySize']]), index=self.test.index, columns=['FamilySize'])
        self.test.loc[:,['PersonalFare']] = pd.DataFrame(scaler.fit_transform(self.test.loc[:,['PersonalFare']]), index=self.test.index, columns=['PersonalFare'])
    """特征选择"""
    def feature_selection(self, method='xgbr'):
        """
        ```
        说明:特征选择
        参数:method
        用法:method选择算法,rl使用随机Lasso算法,rf使用随机森林算法
        是否完成:是
        ```
        """
        # 其它
        def other():
            pass
        # RandomLasso
        def RLasso():
            rlasso = RandomizedLasso(max_iter=1000)
            rlasso.fit(self.train,np.array(self.train_survived).ravel())
            self.rlasso_scores = pd.DataFrame(np.append(self.train.columns,rlasso.scores_).reshape((2,len(rlasso.scores_))).transpose(), columns=['Feature', 'score'])
            # 作图
            if False:
                plt.plot(np.arange(self.rlasso_scores.score.shape[0]),self.rlasso_scores.score.sort_values(ascending=False))
                plt.scatter(np.arange(self.rlasso_scores.score.shape[0]),self.rlasso_scores.score.sort_values(ascending=False))
                plt.show()
            # 丢弃部分特征
            # self.train_reduced = self.train
            self.train_reduced = self.train.drop(self.rlasso_scores.Feature[self.rlasso_scores.score<0.5].values, axis=1)
            # self.train_reduced = self.train.drop(['Pclass_2', 'Male', 'Embarked_Q', 'Embarked_C'], axis=1)
            # self.test_reduced = self.test
            self.test_reduced = self.test.drop(self.rlasso_scores.Feature[self.rlasso_scores.score<0.5].values, axis=1)
            # self.test_reduced = self.test.drop(['Pclass_2', 'Male', 'Embarked_Q', 'Embarked_C'], axis=1)
        # RandomForest
        def RForest():
            rf = RandomForestRegressor(warm_start=True)
            rf.fit(self.train, np.array(self.train_survived).ravel())
            print(rf.feature_importances_)
            model = SelectFromModel(rf, prefit=True)
            self.train_reduced = pd.DataFrame(model.transform(self.train), index=self.train.index)
            self.test_reduced = pd.DataFrame(model.transform(self.test), index=self.test.index)
        # XGBRegressor
        def XGBR():
            xgbr = xgb.XGBRegressor()
            xgbr.fit(self.train, np.array(self.train_survived).ravel())
            print(xgbr.feature_importances_)
            model = SelectFromModel(xgbr, prefit=True)
            self.train_reduced = pd.DataFrame(model.transform(self.train), index=self.train.index)
            self.test_reduced = pd.DataFrame(model.transform(self.test), index=self.test.index)
        # 选择器
        switcher = {
            'rl':RLasso,
            'rf':RForest,
            'xgbr':XGBR
        }
        select = switcher.get(method, other)
        return select()
    """降维"""
    def deminsionality_reduction(self, method='lda' ,n_components=10, svd_solver='full'):
        """
        ```
        说明:降维
        参数:method,n_components,svd_solver
        用法:method决定降维方法,n_components决定聚类数目,svd_solver配合'mle'
        是否完成:否
        ```
        """
        # TODO
        """有问题,暂时弃用"""
        # 其它
        def other(n_components=n_components, svd_solver=svd_solver):
            pass
        # PCA
        def pca_kernel(n_components=n_components, svd_solver=svd_solver):
            pca = PCA(n_components=n_components, svd_solver=svd_solver)
            pca.fit(self.train)
            self.train_reduced = pca.transform(self.train)
            self.test_reduced = pca.transform(self.test)
        # LDA
        def lda_kernel(n_components=n_components, svd_solver=svd_solver):
            lda = LDA(n_components=n_components)
            lda.fit(self.train, self.train_survived.squeeze())
            self.train_reduced = lda.transform(self.train)
            self.test_reduced = lda.transform(self.test)
        # 选择器
        switcher = {
            'pca':pca_kernel,
            'lda':lda_kernel
        }
        demin_reduce = switcher.get(method, other)
        return demin_reduce()
    """训练"""
    def training(self, model='xgb'):
        """
        ```
        说明:训练模型
        参数:model
        用法:model决定训练方法
        是否完成:否
        ```
        """
        # alert
        print('-'*12+' 训练模型 '+'-'*12+'\n')
        # 其它
        def other():
            pass
        # KNN
        def knn_kernel():
            print('-'*14+' KNN '+'-'*14+'\n')
            multi_scores = []
            print('-'*14+' 开始 '+'-'*14+'\n')
            for time in range(10): 
                print('-'*12+' 第 %02d 次 '%(time+1)+'-'*12+'\n')
                #
                knn = KNeighborsClassifier(weights='distance')
                scores = cross_val_score(knn, self.train_reduced, self.train_survived.squeeze(), cv=10, scoring='accuracy')
                print('10折交叉验证准确率:%f\n'%(scores.mean()))
                multi_scores.append(scores.mean())
                #
                predicted = cross_val_predict(knn, self.train_reduced, self.train_survived.squeeze(), cv=10)
                accuracy = metrics.accuracy_score(self.train_survived.squeeze(),predicted)
                print('训练集预测准确率:%f\n'%(accuracy))
                knn.fit(self.train_reduced, self.train_survived.squeeze())
            print('-'*14+' 结束 '+'-'*14+'\n')
            self.kernel = knn
            print('10次10折交叉验证准确率:%f\n'%(np.mean(multi_scores)))
        # SVM
        def svm_kernel():
            print('-'*14+' SVM '+'-'*14+'\n')
            multi_scores = []
            print('-'*14+' 开始 '+'-'*14+'\n')
            for time in range(10):
                print('-'*12+' 第 %02d 次 '%(time+1)+'-'*12+'\n')
                #
                svc = SVC()
                scores = cross_val_score(svc, self.train_reduced, self.train_survived.squeeze(), cv=10, scoring='accuracy')
                print('10折交叉验证准确率:%f\n'%(scores.mean()))
                multi_scores.append(scores.mean())
                #
                predicted = cross_val_predict(svc, self.train_reduced, self.train_survived.squeeze(), cv=10)
                accuracy = metrics.accuracy_score(self.train_survived.squeeze(),predicted)
                print('训练集预测准确率:%f\n'%(accuracy))
                svc.fit(self.train_reduced, self.train_survived.squeeze())
            print('-'*14+' 结束 '+'-'*14+'\n')
            self.kernel = svc
            print('10次10折交叉验证准确率:%f\n'%(np.mean(multi_scores)))
        # MLP
        def mlp_kernel():
            print('-'*14+' MLP '+'-'*14+'\n')
            multi_scores = []
            print('-'*14+' 开始 '+'-'*14+'\n')
            for time in range(10):
                print('-'*12+' 第 %02d 次 '%(time+1)+'-'*12+'\n')
                #
                mlp = MLPClassifier(hidden_layer_sizes=(25),max_iter=1000)
                scores = cross_val_score(mlp, self.train_reduced, self.train_survived.squeeze(), cv=10, scoring='accuracy')
                print('10折交叉验证准确率:%f\n'%(scores.mean()))
                multi_scores.append(scores.mean())
                #
                predicted = cross_val_predict(mlp, self.train_reduced, self.train_survived.squeeze(), cv=10)
                accuracy = metrics.accuracy_score(self.train_survived.squeeze(),predicted)
                print('训练集预测准确率:%f\n'%(accuracy))
                mlp.fit(self.train_reduced, self.train_survived.squeeze())
            print('-'*14+' 结束 '+'-'*14+'\n')
            self.kernel = mlp
            print('10次10折交叉验证准确率:%f\n'%(np.mean(multi_scores)))
        # XGB
        def xgb_kernel():
            print('-'*14+' XGB '+'-'*14+'\n')
            params = {
                # General Parameters
                'booster':'gbtree',
                'objective': 'binary:logistic', # 二分类的问题,损失函数Logistic
                # Tree Booster Parameters
                'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2
                'max_depth':8, # 构建树的深度，越大越容易过拟合
                'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合
                'subsample':0.7, # 随机采样训练样本
                'colsample_bytree':0.7, # 生成树时进行的列采样
                'min_child_weight':4, 
                'silent':0,# 设置成1则没有运行信息输出，最好是设置为0
                'eta': 0.1, # 如同学习率
                'seed':1,
                'nthread':5,# cpu 线程数
                'eval_metric': 'auc'
            }
            params_list = list(params.items())
            # Task Parameters
            num_rounds = 1000 # 迭代次数
            train_set,validation_set = train_test_split(pd.concat([self.train_reduced,self.train_survived], axis=1), test_size=0.3, random_state=np.random.RandomState(np.random.randint(2**32)))

            xgb_train = xgb.DMatrix(train_set.drop(['Survived'], axis=1), label=train_set.Survived)
            xgb_validation = xgb.DMatrix(validation_set.drop(['Survived'], axis=1), label=validation_set.Survived)
            xgb_test = xgb.DMatrix(self.test_reduced)

            watchlist = [(xgb_train, 'train'),(xgb_validation, 'val')]

            print('-'*14+' 开始 '+'-'*14+'\n')

            # 记录程序运行时间
            import time 
            start_time = time.time()
            # 训练模型
            model = xgb.train(params_list, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
            # 获取运行时长
            cost_time = time.time()-start_time
            # 储存训练出的模型
            model.save_model('xgb.model')
            print("Best Ntree Limit:%d\n"%(model.best_ntree_limit))
            
            self.kernel = model
            print(model.get_fscore())
            
            # 预测
            predict = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
            # 计算正确率
            accuracy = metrics.accuracy_score(np.array(self.test_survived).squeeze(), np.array(np.round(predict).astype(int)))
            print('测试集正确率:%f\n'%(accuracy))
            # 分类报告及混淆矩阵
            print("分类报告:")
            print(classification_report(np.array(self.test_survived).squeeze(), np.array(np.round(predict).astype(int))))
            print("混淆矩阵:")
            print(confusion_matrix(np.array(self.test_survived).squeeze(), np.array(np.round(predict).astype(int))))
            print('-'*14+' 结束 '+'-'*14+'\n')
            print("运行时长:%d"%(cost_time)+"(s)\n")
        # GridSearchCV
        def gscv_kernel():
            params = {
                # General Parameters
                'booster':['gbtree'],
                'objective': ['binary:logistic'], # 二分类的问题,损失函数Logistic
                # Tree Booster Parameters
                'gamma':[0.1, 0.2],  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2
                'max_depth':[6, 8, 10, 12], # 构建树的深度，越大越容易过拟合
                'reg_lambda':[2],  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合
                'subsample':[0.7], # 随机采样训练样本
                'colsample_bytree':[0.7], # 生成树时进行的列采样
                'min_child_weight':[1, 2, 3, 4, 5], 
                'silent':[0],# 设置成1则没有运行信息输出，最好是设置为0
                'learning_rate': [0.1, 0.01, 0.001, 0.0001], # 如同学习率
                # 'seed':np.random.randint(2**32),
                'random_state':[1],
                # 'nthread':5,# cpu 线程数
                'n_jobs':[5],
            }
            xgbc = xgb.XGBClassifier()
            # params_list = list(params.items())
            self.kernel = GridSearchCV(xgbc, params, cv=10)
            self.kernel.fit(self.train, np.array(self.train_survived).ravel())
            self.cv_result = pd.DataFrame.from_dict(self.kernel.cv_results_)
            print("模型最佳参数组合:")
            print(self.kernel.best_params_)
        # 选择器
        switcher = {
            'knn':knn_kernel,
            'svm':svm_kernel,
            'mlp':mlp_kernel,
            'xgb':xgb_kernel,
            'gscv':gscv_kernel
        }
        model_train = switcher.get(model, other)
        return model_train()
    """预测"""
    def predicting(self):
        """
        ```
        说明:进行预测
        参数:无
        用法:空
        是否完成:否
        ```
        """
        # alert
        print('-'*12+' 检验模型 '+'-'*12+'\n')
        # 预测
        predict = self.kernel.predict(self.test_reduced)
        # 计算正确率
        accuracy = metrics.accuracy_score(np.array(self.test_survived).squeeze(), predict)
        print('测试集正确率:%f\n'%(accuracy))
        # 分类报告及混淆矩阵
        print("分类报告:")
        print(classification_report(np.array(self.test_survived).squeeze(),predict))
        print("混淆矩阵:")
        print(confusion_matrix(np.array(self.test_survived).squeeze(),predict))

# basic
# 定义Usage类
class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg
        
# 定义入口函数
def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except (getopt.error) as msg:
            raise Usage(msg)
        # begin

    except (Usage) as err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2     

# 程序入口
if __name__ == '__main__':
    sys.exit(main())
            