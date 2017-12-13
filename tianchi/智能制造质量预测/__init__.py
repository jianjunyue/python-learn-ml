import numpy as np
import pandas as pd
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# model
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

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
y_train=train_df["Y"]
print("---1----")
train_df=train_df.drop(["ID","Y"], axis=1)
print("---11----")
quantity = [attr for attr in train_df.columns if train_df.dtypes[attr] != 'object']  # 数值变量集合
train_df=train_df[quantity]
X_train = Imputer().fit_transform(train_df)
print("---111----")
xgb1 = RandomForestRegressor(n_estimators=200)
xgb1.fit(X_train, y_train)
print("---2----")
predict_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A.xlsx',header=0,encoding='utf-8')

# predict_df=predict_df.drop(["ID"],axis=1)
predict_df=predict_df[quantity]
predict_df = Imputer().fit_transform(predict_df)
print("---3----")
pred=xgb1.predict(predict_df)
pred_df=pd.DataFrame()
pred["pred"]=pred
pred_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_pred.csv', index=False, float_format='%.4f')
print("---4----")
# submission_df=pd.DataFrame()
#
# submission_iddf = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板.csv')
# pred_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_pred.csv')
# submission_df["id"]=submission_iddf["id"]
# submission_df["pred"]=pred_df["pred"]
# print(submission_df.head(3))
# submission_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_sub.csv',header=False, index=False, float_format='%.4f')