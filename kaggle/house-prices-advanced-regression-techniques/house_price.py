
# 1. kaggle房价预测/Ridge/RandomForest/cross_validation
#http://blog.csdn.net/youyuyixiu/article/details/72840893

# 2. Kaggle房价预测进阶版/bagging/boosting/AdaBoost/XGBoost
#http://blog.csdn.net/youyuyixiu/article/details/72841703

#Kaggle房价预测：数据探索——练习
#http://blog.csdn.net/qilixuening/article/details/75151026

#Kaggle房价预测：数据预处理——练习
#http://blog.csdn.net/qilixuening/article/details/75153131

# coding:utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

train_df = pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/house_price/house_price_train.csv",index_col = 0)
test_df = pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/house_price/house_price_test.csv",index_col = 0)
# print(train_df.shape)
# print(test_df.shape)
# print(train_df.head())  # 默认展示前五行 这里是5行,80列
# print(test_df.head())   # 这里是5行,79列
sns.distplot(train_df['SalePrice'])
plt.show()
prices = pd.DataFrame({'price':train_df['SalePrice'],'log(price+1)':np.log1p(train_df['SalePrice'])})
# print(prices.head())
train_df["SalePrice"]=np.log1p(train_df['SalePrice'])
train_df['MSSubClass'] = train_df['MSSubClass'].astype(str)
test_df['MSSubClass'] = test_df['MSSubClass'].astype(str)
train_df = pd.get_dummies(train_df, columns=["MSSubClass"])
test_df=  pd.get_dummies(test_df, columns=["MSSubClass"]) 
# print(test_df.describe())
# print(train_df.isnull().sum().sort_values(ascending = False).head(5) )
print(train_df.isnull().sum().sort_values(ascending = False).head(5) )
train_df=train_df[train_df.columns[train_df.dtypes != 'object']] 
test_df=test_df[test_df.columns[test_df.dtypes != 'object']]
# 我们这里用mean填充
mean_cols = train_df.mean()
# print(mean_cols.head(10))
all_dummy_df = train_df.fillna(mean_cols)
# print(all_dummy_df)
# print(all_dummy_df.isnull().sum().sum())
numeric_cols = train_df.columns[train_df.dtypes != 'object']
train_df=train_df[numeric_cols]
test_numeric_cols = test_df.columns[test_df.dtypes != 'object']
test_df=test_df[test_numeric_cols]
print(all_dummy_df["LotFrontage"])


train_df=train_df.drop(['LotFrontage','GarageYrBlt','MasVnrArea','BsmtFullBath','BsmtHalfBath','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF'],axis=1)
test_df=test_df.drop(['LotFrontage','GarageYrBlt','MasVnrArea','BsmtFullBath','BsmtHalfBath','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF'],axis=1)

print(train_df.isnull().sum().sort_values(ascending = False).head(5) )
print(test_df.isnull().sum().sort_values(ascending = False).head(5) )
# print(train_df["LotFrontage"] )

train_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/house_price/model/house_price_train.csv' ,index = False)
test_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/house_price/model/house_price_test.csv' ,index = False)