import numpy as np
import pandas as pd
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
# model
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split


train_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练.xlsx',header=0,encoding='utf-8')
# print(train_df.head(3))
# print(train_df.columns)
# quality = [attr for attr in train_df.columns if train_df.dtypes[attr] == 'object']  # 类型变量集合
# print(train_df[quantity].head(5))
# y_train=train_df["Y"]
y_train=np.log1p(train_df["Y"])
print("---1----")
train_df=train_df.drop(["ID"], axis=1)
quantity = [attr for attr in train_df.columns if train_df.dtypes[attr] == 'object']  # 数值变量集合
quantity111 = [attr for attr in train_df.columns if train_df.dtypes[attr] != 'object']
print(len(quantity111))
# print(quantity)
# print(train_df[quantity].head())
train_df = pd.get_dummies(train_df)
train_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练处理get_dummies.csv', index=False)

# for key in quantity:
#     print(train_df[key].head())
#     keyTemp=key.replace("_","1111").replace(" ","2222")
#     print("------------"+key+"--------------")
#     train_df[key] = train_df[key].astype(str)
#     print(train_df[key].value_counts())
#
#     train_df=pd.get_dummies(train_df[key], prefix=keyTemp)
#     print("--------------------------")
# train_df=train_df[quantity]
# # X_train = Imputer().fit_transform(train_df)
X_train=train_df

quantity222 = [attr for attr in train_df.columns if train_df.dtypes[attr] != 'object']
print(len(quantity222))
print("---111----")
print(train_df.head())

quantityqqq = [attr for attr in train_df.columns if train_df.dtypes[attr] == 'object']  # 数值变量集合
print(len(quantityqqq))
print("---quantityqqq----")
params = range(50,500,50)
print(params)
test_scores = []
for param in params:
    clf = XGBRegressor(n_estimators=param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
    print(param)
    print(np.mean(test_score))
#
# plt.plot(params, test_scores)
# plt.title("Alpha vs CV Error");
# plt.show()
# xgb1 = XGBRegressor()
# xgb1.fit(X_train, y_train)
# print("---2----")
# predict_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A.xlsx',header=0,encoding='utf-8')
#
# # predict_df=predict_df.drop(["ID"],axis=1)
# predict_df=predict_df[quantity]
# # predict_df = Imputer().fit_transform(predict_df)
# print("---3----")
# # pred=xgb1.predict(predict_df)
# pred=np.expm1(xgb1.predict(predict_df))
# pred_df=pd.DataFrame()
# pred_df["pred"]=pred
#
# pred_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_pred.csv', index=False, float_format='%.9f')
# print("---4----")
# submission_df=pd.DataFrame()
#
# submission_iddf = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板.csv')
# pred_df_TEMP = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_pred.csv')
# submission_df["id"]=submission_iddf["id"]
# submission_df["pred"]=pred_df_TEMP["pred"]
# print(submission_df.head(3))
# submission_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_sub.csv',header=False, index=False, float_format='%.9f')