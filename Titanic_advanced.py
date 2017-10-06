# basic
import sys
import getopt
# numpy&pandas
# import numpy as np
import pandas as pd
# pyplot
import matplotlib.pyplot as plt
# sklearn
# 缺失值
from sklearn.preprocessing import Imputer
# 分类数据扩维
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
# 二值化
from sklearn.preprocessing import Binarizer
# 无量纲化
from sklearn.preprocessing import MinMaxScaler
# 降维
## PCA
from sklearn.decomposition import PCA
## LDA
from sklearn.decomposition import LatentDirichletAllocation as LDA
# from sklearn.cross_validation import train_test_split

# 定义Titanic类
class Titanic(object):
    """
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
            参数:无
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
            参数:无
            说明:选择特征
            是否完成:否
        27_降维:
            类函数实现:deminsionality_reduction
            参数:n_components,svd_solver
            说明:n_components决定聚类数目,svd_solver配合'mle'
            是否完成:是

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
    def __init__(self):
        self.data_collection()
        self.feature_extraction()
        self.missing_value_calculation()
        self.binarization_processing()
        self.feature_coding()
        self.nondimensionalize()
        self.deminsionality_reduction()
    
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
    def data_cleaning(self):
        pass
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
        self.train_survived = self.train.loc[:,['Survived']]
        self.train = self.train.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
        self.test_survived = self.real.loc[:,['Survived']]
        self.test = self.test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
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
        # 处理train和test
        self.train = processing(self.train)
        self.test = processing(self.test)
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
        minmaxscaler = MinMaxScaler()
        self.train.loc[:,['Age']] = pd.DataFrame(minmaxscaler.fit_transform(self.train.loc[:,['Age']]), index=self.train.index, columns=['Age'])
        self.train.loc[:,['Fare']] = pd.DataFrame(minmaxscaler.fit_transform(self.train.loc[:,['Fare']]), index=self.train.index, columns=['Fare'])
        self.test.loc[:,['Age']] = pd.DataFrame(minmaxscaler.fit_transform(self.test.loc[:,['Age']]), index=self.test.index, columns=['Age'])
        self.test.loc[:,['Fare']] = pd.DataFrame(minmaxscaler.fit_transform(self.test.loc[:,['Fare']]), index=self.test.index, columns=['Fare'])
    """特征选择"""
    def feature_selection(self):
        pass
    """降维"""
    def deminsionality_reduction(self, n_components='mle', svd_solver='full'):
        """
        ```
        说明:降维
        参数:n_components,svd_solver
        用法:n_components决定聚类数目,svd_solver配合'mle'
        是否完成:是
        ```
        """
        pca = PCA(n_components=n_components, svd_solver=svd_solver)
        pca.fit(self.train)
        self.train_reduced = pd.DataFrame(pca.transform(self.train))
        self.test_reduced = pd.DataFrame(pca.transform(self.train))
    """训练"""
    def training(self):
        pass

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
        t = Titanic()

    except (Usage) as err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2     

# 程序入口
if __name__ == '__main__':
    sys.exit(main())
            