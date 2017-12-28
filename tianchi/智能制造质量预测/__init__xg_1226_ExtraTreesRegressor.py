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

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = [2, 5, 8, 11, 0]
intersection = list(set(a).intersection(set(b)))
print(intersection)
# train_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练.xlsx',header=0,encoding='utf-8')
train_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练处理get_dummies.csv',header=0,encoding='utf-8')
train_df = train_df.fillna(0)
print("---head----")
print(train_df["340X82"].head(10))
print("---describe----")
print(train_df["340X82"].describe())
print("---info----")
# print(train_df["340X82"].info())
print("---unique----")
print(train_df["340X82"].unique())
print("---1----")

predict_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A.xlsx',header=0,encoding='utf-8')
predict_df = predict_df.fillna(0)
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
etr=ExtraTreesRegressor(bootstrap=False, max_features=1.0, min_samples_leaf=2, min_samples_split=13, n_estimators=100)
# print(X_train.isnull().sum().sort_values(ascending=False).head(10))
# print(y_train.isnull().sum().sort_values(ascending=False).head(10))
# X_train = transform.transform(X_train)
etr.fit(X_train, y_train)
# xgb1.fit(X_train, y_train)
print("---2----")
predict_df=predict_df[quantity]
# print(predict_df.isnull().sum().sort_values(ascending=False).head(10))
print("---3----")
pred=np.expm1(etr.predict(predict_df))
pred_df=pd.DataFrame()
pred_df["pred"]=pred
pred_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_pred.csv', index=False, float_format='%.9f')
print("---4----")
submission_df=pd.DataFrame()

submission_iddf = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板.csv')
pred_df_TEMP = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_pred.csv')
submission_df["id"]=submission_iddf["id"]
submission_df["pred"]=pred_df_TEMP["pred"]
print(submission_df.head(3))
submission_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_sub_1226_ExtraTreesRegressor.csv',header=False, index=False, float_format='%.9f')