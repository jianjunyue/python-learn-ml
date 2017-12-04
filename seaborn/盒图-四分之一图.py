
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

fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(train_df["OverallQual"].values,train_df["SalePrice"].values)
ax.set_ylim(0,800000)
plt.xticks(rotation=90)
plt.show()
