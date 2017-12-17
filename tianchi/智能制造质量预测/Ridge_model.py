import pandas as pd
import numpy as np
#Python sklearn数据分析中常用方法
#http://blog.csdn.net/qq_16234613/article/details/76534673

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# train_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练.xlsx',header=0,encoding='utf-8')
# y_train=train_df["Y"]
# y_train.to_csv("/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练_y_train_20171216.csv",index=False)
X_train=pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练_quantity_2017121622.csv',header=0)
# X_train.shape
y_train=pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练_y_train_20171216.csv',header=0)
# train_df=train_df.drop(["ID","Y"], axis=1)
# quantity = [attr for attr in train_df.columns if train_df.dtypes[attr] != 'object']  # 数值变量集合
quantity = [attr for attr in X_train.columns if X_train.dtypes[attr] == 'float64']  # 数值变量集合
print(len(quantity))
print(quantity)
# print(X_train[quantity].head())
# X_train=X_train.drop(["750X1452"], axis=1)
print(X_train.shape)
print(y_train.shape)
# # print(len(X_train.columns))
# X_train=train_df[quantity]
# X_train.to_csv("/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练_quantity_20171216.csv",index=False)
# X_train = Imputer().fit_transform(X_train)
# data1=np.isnan(X_train).any()
# print( anynull(data1).head())
# 检查数据中是否有缺失值
# print(type(np.isnan(X_train).any()))
# 2、删除有缺失值的行
# train.dropna(inplace=True)
num=0
# count=0
# for column in X_train.columns:
#     try:
#         count=count+1
#         if X_train.dtypes[column] == 'float64':
#             print(column + "::"+str(count)+"/"+str(num))
#             X_train[column] = X_train[column].astype(float)
#             print(X_train.dtypes[column])
#             print(X_train[column][0])
#             # X_train[column] = X_train[column].apply(lambda x:)
#             # data['year'] = data.Date.apply(lambda x: x.split('-')[0])
#     except Exception as err:
#         print(column+":::"+err)
#         # X_train[column] = X_train[column].fillna(0)
#         # X_train = Imputer().fit_transform(train_df)

# for column in X_train.columns:
#     try:
#         count=count+1
#         print(column + "::"+str(count)+"/"+str(num))
#         X_train[column] = X_train[column].fillna(X_train[column].median())
#     except Exception as err:
#         print(column+":::"+err)
#         X_train[column] = X_train[column].fillna(0)
#         # X_train = Imputer().fit_transform(train_df)

# X_train.to_csv("/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练_quantity_2017121622.csv",index=False)
#交叉验证，获取最优参数( 最终使用GridSearch获取最优参数 )
alphas=np.logspace(-3,2,50)
test_scores=[]
for alpha in alphas:
    clf=Ridge(alpha)
    test_score=np.sqrt(-cross_val_score(clf,X_train,y_train,cv=5,scoring="neg_mean_squared_error"))
    test_scores.append(np.mean(test_score))

print(test_scores)
plt.plot(alphas,test_scores)
plt.title("Ridge Model")
plt.show()
