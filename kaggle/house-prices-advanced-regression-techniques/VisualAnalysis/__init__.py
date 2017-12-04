
#Kaggle房价预测：数据探索——练习
#http://blog.csdn.net/qilixuening/article/details/75151026

#Kaggle房价预测：数据预处理——练习
#http://blog.csdn.net/qilixuening/article/details/75153131

# coding:utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.stats import norm
from scipy import stats

train_df = pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/house_price/house_price_train.csv",index_col = 0)
# test_df = pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/house_price/house_price_test.csv",index_col = 0)
# print(train_df.shape)
# print(test_df.shape)
# print(train_df.head())  # 默认展示前五行 这里是5行,80列
# print(test_df.head())   # 这里是5行,79列
# sns.distplot(train_df['SalePrice'])
# plt.show()
# print(train_df.columns)
colums={ 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',
       'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice' }
print(type(train_df["MSZoning"].value_counts()))
# print(type(colums))columns
def group_values(value_counts):
    dict = value_counts.to_dict()
    df = pd.DataFrame(columns=['keyid', 'count'])
    listkey = []
    listcount = []
    for key in dict:
        listkey.append(key)
        listcount.append(dict[key])
        df.loc[df.shape[0] + 1] = {'keyid': key, 'count': dict[key]}
    df['keyid']=listkey
    df['count']=listcount
    return df
value_counts =train_df["MSSubClass"].value_counts()
group_df=group_values(value_counts)
group_df=group_df.sort_values(by="keyid" , ascending=True)
# print(group_df)

# plt.bar(left= range(len(group_df["keyid"].values)), height=group_df["count"].values, width=0.35, align="center", yerr=0.0001)
# plt.xticks(range(len(group_df["keyid"].values)),group_df["keyid"].values)
# plt.title("test title")
# plt.xlabel('test X');
# plt.ylabel('test Y');
# plt.show()

# print(train_df["SalePrice"])
# plt.hist(train_df['SalePrice'], bins=100)
# plt.show()

# plt.bar(left= range(len(train_df["GrLivArea"].values)), height=train_df["SalePrice"].values, width=0.35, align="center", yerr=0.0001)
# plt.xticks(range(len(train_df["GrLivArea"].values)),train_df["GrLivArea"].values)
# plt.show()

output,var,var1,var2 = 'SalePrice', 'GrLivArea', 'TotalBsmtSF', 'OverallQual'
# fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(16,5))
# train_df.plot.scatter(x=var,y=output,ylim=(0,800000),ax=axes[0])
# train_df.plot.scatter(x=var1,y=output,ylim=(0,800000),ax=axes[1])
# train_df.plot.scatter(x=var2,y=output,ylim=(0,800000),ax=axes[2])

# fig=plt.figure(figsize=(45,5))
# ax1=fig.add_subplot(1,3,1)
# ax2=fig.add_subplot(1,3,2)
# ax3=fig.add_subplot(1,3,3)
# ax1.scatter(train_df["GrLivArea"].values,train_df["SalePrice"].values)
# ax2.scatter(train_df["TotalBsmtSF"].values,train_df["SalePrice"].values)
# ax3.scatter(train_df["OverallQual"].values,train_df["SalePrice"].values)
# plt.show()

var3 = 'YearBuilt'
fig, ax = plt.subplots(figsize=(36,5))
plt.boxplot(train_df["YearBuilt"].values,train_df["SalePrice"].values)
ax.set_ylim(0,800000)
plt.xticks(rotation=90)
plt.show()
