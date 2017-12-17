import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from hyperopt import tpe
from sklearn.preprocessing import Imputer
from hpsklearn import HyperoptEstimator, any_regressor
from hpsklearn import svc
#随机划分训练集和测试集：
from sklearn.model_selection import train_test_split

train_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练.xlsx',header=0,encoding='utf-8')
y_all=train_df["Y"]
train_df=train_df.drop(["ID","Y"], axis=1)
quantity = [attr for attr in train_df.columns if train_df.dtypes[attr] != 'object']  # 数值变量集合
print(len(quantity))
train_df=train_df[quantity]
# X_all = Imputer().fit_transform(train_df)
for key in quantity:
    train_df[key] = train_df[key].fillna(0)
num_test = 0.33 # 测试集占据比例，，如果是整数的话就是样本的数量
X_train, X_test, y_train, y_test = train_test_split(train_df, y_all, test_size=num_test, random_state=23)
print(X_train.head())
print("------------")
print(X_test.head())
print("------------")
print(y_train.head())
print("------------")
print(y_test.head())
print("------------")
estim = HyperoptEstimator(classifier=any_regressor('clf'),algo=tpe.suggest, seed=0)
estim.fit(X_train,y_train)
print(estim.score(X_test,y_test))
print(estim.best_model())