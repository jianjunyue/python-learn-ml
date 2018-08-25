import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import  SelectKBest,f_classif
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble  import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

path="/Users/jianjun.yue/PycharmGItHub/data/kaggle/House-Prices/train_预处理.csv"
data=pd.read_csv(path)
path_test="/Users/jianjun.yue/PycharmGItHub/data/kaggle/House-Prices/test_预处理.csv"
data_test=pd.read_csv(path_test)

# data_test["TotalBsmtSF"] = data_test["TotalBsmtSF"].fillna(int(data_test["TotalBsmtSF"].mean()))
# data_test["BsmtUnfSF"] = data_test["BsmtUnfSF"].fillna(int(data_test["BsmtUnfSF"].mean()))
# data_test["BsmtFinSF2"] = data_test["BsmtFinSF2"].fillna(int(data_test["BsmtFinSF2"].mean()))
# data_test["BsmtFinSF1"] = data_test["BsmtFinSF1"].fillna(int(data_test["BsmtFinSF1"].mean()))

keyName="BsmtFinSF1"
data_test[keyName] = data_test[keyName].fillna(data_test[keyName].mean())
data_test[keyName] = data_test[keyName].apply(lambda x:int(x))

keyName="BsmtFinSF2"
data_test[keyName] = data_test[keyName].fillna(data_test[keyName].mean())
data_test[keyName] = data_test[keyName].apply(lambda x:int(x))

keyName="BsmtUnfSF"
data_test[keyName] = data_test[keyName].fillna(data_test[keyName].mean())
data_test[keyName] = data_test[keyName].apply(lambda x:int(x))
keyName="TotalBsmtSF"
data_test[keyName] = data_test[keyName].fillna(data_test[keyName].mean())
data_test[keyName] = data_test[keyName].apply(lambda x:int(x))

# print(data.isnull().sum())
print(data.isnull().sum().sort_values(ascending=False).head(10))
# print(data_test.isnull().sum())
print(data_test.isnull().sum().sort_values(ascending=False).head(10))

keyName="BsmtFinSF1"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())
print(data_test[keyName].mean())
data_test[keyName] = data_test[keyName].fillna(int(data_test[keyName].mean()))



y=data["SalePrice"]
# print(data.columns.values.tolist())
columns=[]
# for column in data.columns.values.tolist():
#     if column!="Id":
#         if column!="SalePrice":
#             columns.append(column)
#             print("----------"+column+"------------")
#             train = data[columns]
#             alg = RandomForestRegressor()
#             test = data_test[columns]
#             alg.fit(train, y)
#             pre = alg.predict(test)
#             print(pre)
