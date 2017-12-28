import numpy as np
import pandas as pd
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# model
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = [2, 5, 8, 11, 0]
intersection = list(set(a).intersection(set(b)))
print(intersection)
# train_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练.xlsx',header=0,encoding='utf-8')
train_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练处理get_dummies.csv',header=0,encoding='utf-8')
# train_df = train_df.fillna(0)

predict_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A.xlsx',header=0,encoding='utf-8')
# predict_df = predict_df.fillna(0)
predict_df = pd.get_dummies(predict_df)
quantity_pre_1 = [attr for attr in predict_df.columns if predict_df.dtypes[attr] != 'object']
quantity_1 = [attr for attr in train_df.columns if train_df.dtypes[attr] != 'object']  # 数值变量集合
quantity = list(set(quantity_pre_1).intersection(set(quantity_1)))
y_train=np.log1p(train_df["Y"])
print("---1----")
train_df=train_df.drop(["Y"], axis=1)
print("---11----")
train_df=train_df[quantity]
X_train=train_df
print("---111----")
# xgb1 = XGBRegressor()
# etr=ExtraTreesRegressor(bootstrap=True, max_features=0.25, min_samples_leaf=4, min_samples_split=12, n_estimators=100)
# # print(X_train.isnull().sum().sort_values(ascending=False).head(10))
# # print(y_train.isnull().sum().sort_values(ascending=False).head(10))
# # X_train = transform.transform(X_train)
# etr.fit(X_train, y_train)


clf_type = []
test_scores = []
# clf = RandomForestRegressor()
# test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv = 5,scoring = 'neg_mean_squared_error'))
# test_scores.append(np.mean(test_score))
# clf_type.append("RandomForestRegressor")
# print("RandomForestRegressor")
# print(np.mean(test_score))
#
# clf = ExtraTreesRegressor(bootstrap=True, max_features=0.25, min_samples_leaf=4, min_samples_split=12, n_estimators=100)
# test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv = 5,scoring = 'neg_mean_squared_error'))
# test_scores.append(np.mean(test_score))
# clf_type.append("ExtraTreesRegressor1")
# print("ExtraTreesRegressor1")
# print(np.mean(test_score))
#
# clf = ExtraTreesRegressor(bootstrap=False, max_features=1.0, min_samples_leaf=2, min_samples_split=13, n_estimators=100)
# test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv = 5,scoring = 'neg_mean_squared_error'))
# test_scores.append(np.mean(test_score))
# clf_type.append("ExtraTreesRegressor2")
# print("ExtraTreesRegressor2")
# print(np.mean(test_score))

clf = XGBRegressor()
test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv = 5,scoring = 'neg_mean_squared_error'))
test_scores.append(np.mean(test_score))
clf_type.append("XGBRegressor")
print("XGBRegressor")
print(np.mean(test_score))

plt.plot(range(len(clf_type)),test_scores)
plt.xticks(range(len(clf_type)),clf_type)
plt.title('Max Features vs CV Error')
plt.show()