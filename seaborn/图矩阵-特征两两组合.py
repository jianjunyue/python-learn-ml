
#Kaggle房价预测：数据探索——练习
#http://blog.csdn.net/qilixuening/article/details/75151026

#Kaggle房价预测：数据预处理——练习
#http://blog.csdn.net/qilixuening/article/details/75153131

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/house_price/house_price_train.csv",index_col = 0)
var_set = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

# plt.figure(figsize=(10,10))
sns.set(font_scale=0.8)  # 设置横纵坐标轴的字体大小
sns.pairplot(train_df[var_set],size=1.2, palette="husl")  # 7*7图矩阵
# plt.xticks(rotation=90)
plt.show()
