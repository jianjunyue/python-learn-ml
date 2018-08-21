from pandas import read_csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from matplotlib import pyplot
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
# from pinyin import PinYin
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor


filename_test = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/第三届阿里云安全算法挑战赛/test.csv'
filename_test = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/第三届阿里云安全算法挑战赛/test_temp.csv'


filename = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/第三届阿里云安全算法挑战赛/train.csv'
filename = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/第三届阿里云安全算法挑战赛/train_temp.csv'
data = read_csv(filename,header=0)
test = read_csv(filename_test,header=0)
test=test[["tid","return_value"]]
print(data.head())
X=data[["tid","return_value"]]
Y=data["label"]
print(list(np.unique(Y)))

def rename_columns(pre_name,columns_name):
    name_dict={}
    for name in columns_name:
        name_dict[name]=pre_name+name
    return name_dict
pclass_dummies   = pd.get_dummies(data["label"])
Y=pclass_dummies
print(Y.head())
clf=XGBRegressor(objective='multi:softprob')
# clf.fit()
# test_score = np.sqrt(-cross_val_score(clf, X, Y, cv=5, scoring='neg_mean_squared_error'))
print("------test_score--------")
# print(test_score)
# print(np.mean(test_score))
print("---2----")

clf.fit(X, Y)
print("---3----")
yprob = clf.predict( test )
print(yprob)
# pred_df=pd.DataFrame()
# pred_df["pred"]=pred
#
# pred_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/sub_0105.csv',header=False, index=False, float_format='%.3f')


# # 将数据分为输入数据和输出结果

# for c_id in dict_data.keys():
#     df = dict_data[c_id]
#     X = df.drop(["sale_quantity"], axis=1)
#     Y = df["sale_quantity"]
#     Y = np.log1p(Y)
#     clf=XGBRegressor()
#     kfold = KFold(n_splits=5, random_state=1)
#     test_score = np.sqrt(-cross_val_score(clf, X, Y, cv=kfold, scoring='neg_mean_squared_error'))
#     print("-------------------")
#     print(c_id)
#     print(test_score)
#     print(np.mean(test_score))

#查看重要程度
# feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
# kfold = KFold(n_splits=10, random_state=7)
# test_score = np.sqrt(-cross_val_score(clf, X, Y, cv=kfold, scoring='neg_mean_squared_error'))
# print("------test_score--------")
# print(test_score)
# print(np.mean(test_score))
print("---2----")
# clf=XGBRegressor()
# clf.fit(X, Y)
print("---3----")
# pred=np.expm1(clf.predict(test))
# pred_df=pd.DataFrame()
# pred_df["pred"]=pred

# pred_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/sub_0114_性别_get_dummies.csv',header=False, index=False, float_format='%.3f')

