from pandas import read_csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from matplotlib import pyplot
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

# 导入数据
filename = '/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/d_train_20180102.csv'
data = read_csv(filename,header=0,encoding='GB2312')
data['year'] = data["体检日期"].apply(lambda x: x.split('/')[2])
data['year'] = data['year'].astype(float)
data['month'] = data["体检日期"].apply(lambda x: x.split('/')[1])
data['month'] = data['month'].astype(float)
data['day'] = data["体检日期"].apply(lambda x: x.split('/')[0])
data['day'] = data['day'].astype(float)
data.loc[data["性别"] == '女', "性别"] = 0
data.loc[data["性别"] == '男', "性别"] = 1
data.loc[data["性别"] == '??', "性别"] = 2
# print(data["性别"].unique())
data['性别'] = data['性别'].astype(int)


# data = data.drop(["乙肝e抗体","乙肝表面抗原","乙肝表面抗体","乙肝e抗原","乙肝核心抗体"], axis=1)
# data = data.drop(["尿素","肌酐","尿酸"], axis=1)
# data = data.drop(["*天门冬氨酸氨基转换酶","*碱性磷酸酶","*r-谷氨酰基转换酶","*总蛋白","白蛋白","*球蛋白","白球比例","*丙氨酸氨基转换酶"], axis=1)
# data = data.drop(["低密度脂蛋白胆固醇","甘油三酯","总胆固醇","高密度脂蛋白胆固醇"], axis=1)
# print(data.isnull().sum().sort_values(ascending=False).head(100))
data.isnull().sum()
print("-------------")
mean_cols = data.mean()
mean_cols.head(10)
data = data.fillna(mean_cols)
print(data.isnull().sum().sort_values(ascending=False).head(1))

# 将数据分为输入数据和输出结果
X = data.drop(["id","血糖","体检日期","year"], axis=1)
Y = data["血糖"]
Y= np.log1p(Y)


# print(len(X.columns))
quantity = [attr for attr in X.columns if X.dtypes[attr] != 'object']  # 数值变量集合
# print(len(quantity))
# for column in quantity:
#     if X.dtypes[column] == 'float64':
#         dt=X[column].values.reshape(-1, 1)
#         data_scaler = MinMaxScaler(feature_range=(0, 1))
#         temp=data_scaler.fit_transform(dt).ravel()
#         X[column]=temp

# for column in quantity:
#     if X.dtypes[column] == 'float64':
#         dt=X[column].values.reshape(-1, 1)
#         scaler = StandardScaler().fit(dt)
#         temp = scaler.transform(dt).ravel()
#         X[column]=temp

scaler = Normalizer().fit(X)
X = scaler.transform(X)

clf=XGBRegressor()
kfold = KFold(n_splits=5, random_state=7)
test_score = np.sqrt(-cross_val_score(clf, X, Y, cv=kfold, scoring='neg_mean_squared_error'))
print("------test_score--------")
print(test_score)
print(np.mean(test_score))