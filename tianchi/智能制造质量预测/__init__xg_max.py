import numpy as np
import pandas as pd
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# model
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

from sklearn.model_selection import GridSearchCV
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
# X_train = Imputer().fit_transform(train_df)
X_train=train_df
print("---111----")
# xgb1 = XGBRegressor(n_estimators=200)
# xgb1.fit(X_train, y_train)
# print("---2----")
# predict_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A.xlsx',header=0,encoding='utf-8')

# predict_df=predict_df.drop(["ID"],axis=1)
# predict_df = Imputer().fit_transform(predict_df)
print("---3----")
def getRange(start,stop,step):
    listTemp = [start]
    for i in range(start+step, stop, step):
        listTemp.append(i)
    return listTemp
param_test1 = {
    'n_estimators':getRange(100,1500,100),
    'learning_rate':[0.01, 0.05, 0.08, 0.1, 0.15],
    'max_depth':getRange(3,10,2),
    'min_child_weight':getRange(1,6,2),
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)],
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
}
# xgb1 = XGBRegressor(subsample=0.8)
# xgb1.fit(X_train, y_train)
gsearch1 = GridSearchCV(estimator = XGBRegressor(),param_grid = param_test1,scoring='neg_mean_squared_error',n_jobs=4,iid=False,cv=5)
gsearch1.fit(X_train, y_train)
print("------------------------")
print(gsearch1.grid_scores_)
print("------------------------")
print(gsearch1.best_params_)
print("------------------------")
print(gsearch1.best_score_)

# predict_df=predict_df[quantity]
# pred=xgb1.predict(predict_df)
# pred_df=pd.DataFrame()
# pred_df["pred"]=pred
# pred_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_pred.csv', index=False, float_format='%.4f')
# print("---4----")
# submission_df=pd.DataFrame()
#
# submission_iddf = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板.csv')
# pred_df_TEMP = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_pred.csv')
# submission_df["id"]=submission_iddf["id"]
# submission_df["pred"]=pred_df_TEMP["pred"]
# print(submission_df.head(3))
# submission_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_sub.csv',header=False, index=False, float_format='%.4f')