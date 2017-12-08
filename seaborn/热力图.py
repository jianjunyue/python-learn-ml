
#Kaggle房价预测：数据探索——练习
#http://blog.csdn.net/qilixuening/article/details/75151026

#Kaggle房价预测：数据预处理——练习
#http://blog.csdn.net/qilixuening/article/details/75153131

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/house_price/house_price_train.csv",index_col = 0)
var_set = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt']

corrmat = train_df[var_set].corr()
plt.subplots(figsize=(8, 6))
# sns.heatmap(corrmat, vmax=.8, square=True);
# 设置annot使其在小格内显示数字，annot_kws调整数字格式
sns.heatmap(corrmat, annot=True, annot_kws={'size':9}, square=True);
plt.xticks(rotation=90)
plt.yticks(rotation=1)
plt.show()
