#https://www.kaggle.com/c/house-prices-advanced-regression-techniques
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
#房价预测案例

#file:///Users/jianjun.yue/Desktop/%E7%99%BE%E5%BA%A6%E4%BA%91/06.Kaggle%E5%AE%9E%E6%88%98/Kaggle%E5%AE%9E%E6%88%98/Kaggle%E7%AC%AC%E4%BA%8C%E8%AF%BE%E4%BB%A3%E7%A0%81/%E7%AC%AC%E4%BA%8C%E8%AF%BE/house%20price/notebook/house_price.html

#Step 1: 检视源数据集
import matplotlib.pyplot as plt
from sklearn import preprocessing
train_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/house_price/train.csv', index_col=0)
test_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/house_price/test.csv', index_col=0)
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)
# print(train_df.columns)
# print(train_df.head())
# quantity = [attr for attr in train_df.columns if train_df.dtypes[attr] == 'object']  # 数值变量集合
for column in train_df.columns:
    print(train_df[column].value_counts())
    print("---------------------------------")

# train_df=pd.get_dummies(train_df['MSSubClass'], prefix='MSSubClass')

print(train_df.columns)
# for column in train_df.columns:
#     if column.find('PoolQC') :
#         print(column)
# print(train_df.head())
# print(len(train_df["PoolQC"]))
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)
print(train_df.isnull().sum().sort_values(ascending=False).head(20))
print(test_df.isnull().sum().sort_values(ascending=False).head(20))
# print(train_df[["LotFrontage","GarageYrBlt","MasVnrArea"]].head())

train_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/house_price/train_20171217.csv', index=False)
test_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/house_price/test_20171217.csv', index=False)