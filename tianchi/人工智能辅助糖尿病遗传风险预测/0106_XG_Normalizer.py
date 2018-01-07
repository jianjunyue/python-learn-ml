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

from sklearn.preprocessing import Normalizer
# 导入数据
filename = '/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/d_train_20180102.csv'
data = read_csv(filename,header=0,encoding='GB2312')
data['year'] = data["体检日期"].apply(lambda x: x.split('/')[2])
data['year'] = data['year'].astype(float)
data['month'] = data["体检日期"].apply(lambda x: x.split('/')[1])
data['month'] = data['month'].astype(float)
data['day'] = data["体检日期"].apply(lambda x: x.split('/')[0])
data['day'] = data['day'].astype(float)
data.loc[data["性别"] == '女', "性别"] = 0
data.loc[data["性别"] == '男', "性别"] = 1
data.loc[data["性别"] == '??', "性别"] = 2
# print(data["性别"].unique())
data['性别'] = data['性别'].astype(int)

test = read_csv("/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/d_test_A_20180102.csv",header=0,encoding='GB2312')
test['year'] = test["体检日期"].apply(lambda x: x.split('/')[2])
test['year'] = test['year'].astype(int)
test['month'] = test["体检日期"].apply(lambda x: x.split('/')[1])
test['month'] = test['month'].astype(int)
test['day'] = test["体检日期"].apply(lambda x: x.split('/')[0])
test['day'] = test['day'].astype(int)
test.loc[test["性别"] == '女', "性别"] = 0
test.loc[test["性别"] == '男', "性别"] = 1
test.loc[test["性别"] == '??', "性别"] = 2
# print(data["性别"].unique())
test['性别'] = test['性别'].astype(int)
test = test.drop(["id","体检日期","year"], axis=1)
# print(test.head())


mean_cols = data.mean()
data = data.fillna(mean_cols)

test_mean_cols = test.mean()
test = test.fillna(test_mean_cols)

# 将数据分为输入数据和输出结果
X = data.drop(["id","血糖","体检日期","year"], axis=1)

# X = X.drop(["单核细胞%"], axis=1)
# X = X.drop(["淋巴细胞%"], axis=1)
# X = X.drop(["乙肝e抗原"], axis=1)
# X = X.drop(["乙肝表面抗体"], axis=1)
Y = data["血糖"]
Y=np.log1p(Y)

scaler = Normalizer().fit(X)
X = scaler.transform(X)

clf=XGBRegressor()
kfold = KFold(n_splits=5, random_state=7)
test_score = np.sqrt(-cross_val_score(clf, X, Y, cv=kfold, scoring='neg_mean_squared_error'))
print("------test_score--------")
print(test_score)
print(np.mean(test_score))

print("---2----")
clf=XGBRegressor()
clf.fit(X, Y)
# feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
# print(clf.booster().get_fscore())

scaler = Normalizer().fit(test)
test = scaler.transform(test)
pred=np.expm1(clf.predict(test))
pred_df=pd.DataFrame()
pred_df["pred"]=pred
pred_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/sub_0106_XG_Normalizer.csv',header=False, index=False, float_format='%.3f')
