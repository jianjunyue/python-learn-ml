from pandas import read_csv
import numpy as np
import pandas as pd


# 导入数据
filename = '/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/d_train_20180102.csv'
data = read_csv(filename,header=0,encoding='GB2312')

predictors=["id","性别"]
print("---------------------------")
print(data[predictors].info())
print("--------------value_counts-------------")
print(data["性别"].value_counts())
print("-------------unique--------------")
print(data["性别"].unique())
print("--------------describe-------------")
print(data[predictors].describe())
print("---------------------------")

data['year'] = data["体检日期"].apply(lambda x: x.split('/')[2])
data['year'] = data['year'].astype(float)

data.loc[data["性别"] == '女', "性别"] = 0
data.loc[data["性别"] == '男', "性别"] = 1
data.loc[data["性别"] == '??', "性别"] = 2
#对单个特征缺失值，用中位数填充
data["性别"] = data["性别"].fillna(data["性别"].median())
data["性别"] = data["性别"].fillna(0)

data = data.drop(["id","体检日期","year"], axis=1)

#保留至少3个非空值的行：一行中有3个值是非空的就保留
data=data.dropna(thresh=3)

#缺失值统计
data.isnull().sum()
data.isnull().sum().sort_values(ascending=False).head(10)
#平均值填充缺失值
mean_cols = data.mean()
data = data.fillna(mean_cols)
print(type(mean_cols))
print(mean_cols)

#特征处理 ok
# http://blog.csdn.net/oxuzhenyi/article/details/57438667