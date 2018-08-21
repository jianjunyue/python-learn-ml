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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


filename_test = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/第三届阿里云安全算法挑战赛/test.csv'
filename_test = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/第三届阿里云安全算法挑战赛/test_temp.csv'


filename = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/第三届阿里云安全算法挑战赛/train.csv'
filename = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/第三届阿里云安全算法挑战赛/train_temp.csv'
data = read_csv(filename,header=0)
test = read_csv(filename_test,header=0)
test=test[["tid","return_value"]]
# print(data.head())
X=data[["tid","return_value"]]
Y=data["label"]
def rename_columns(pre_name,columns_name):
    name_dict={}
    for name in columns_name:
        name_dict[name]=pre_name+name
    return name_dict
# pclass_dummies   = pd.get_dummies(data["label"])
# print(pclass_dummies.head())
# occ_cols = ['label_' +  columns_name for columns_name in pclass_dummies_titanic.columns]
# pclass_dummies_titanic.rename(columns=rename_columns('label_',pclass_dummies_titanic.columns), inplace = True)
print(list(np.unique(Y)))
print(Y.value_counts())
# print(pclass_dummies.head())
clf = RandomForestClassifier(n_estimators=200, random_state=1)
print("---2----")
clf.fit(X, Y)
# yprob = clf.predict(test)
yprob = clf.predict_proba(test)
print(yprob)
pred_df=pd.DataFrame()
pred_df["prob0"]=yprob[:,0]
pred_df["prob1"]=yprob[:,1]
pred_df["prob2"]=yprob[:,2]
pred_df["prob3"]=yprob[:,3]
pred_df["prob4"]=yprob[:,4]
pred_df["prob5"]=yprob[:,5]
print("------------------------------")
print(pred_df.head(10))
