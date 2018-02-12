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

filename = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/乘用车零售量预测/yancheng_train_20171226_0211_处理.csv'
data = read_csv(filename,header=0)
data=data.sort_values("sale_date")
class_id=data.get("class_id")
dict_class_id={}
for c_id in class_id:
    dict_class_id[c_id]=1

dict_data={}
for c_id in dict_class_id.keys():
    data_temp=data[data.class_id==c_id]
    dict_data[c_id]=data_temp

# # 将数据分为输入数据和输出结果

for c_id in dict_data.keys():
    df = dict_data[c_id]
    X = df.drop(["sale_quantity"], axis=1)
    Y = df["sale_quantity"]
    Y = np.log1p(Y)
    clf=XGBRegressor()
    kfold = KFold(n_splits=5, random_state=1)
    test_score = np.sqrt(-cross_val_score(clf, X, Y, cv=kfold, scoring='neg_mean_squared_error'))
    print("-------------------")
    print(c_id)
    print(test_score)
    print(np.mean(test_score))

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

