# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor

train_df = pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/house_price/model/house_price_train.csv",index_col = 0)
test_df = pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/house_price/model/house_price_test.csv",index_col = 0)
numeric_cols = train_df.columns[train_df.dtypes != 'object']
y_train=train_df["SalePrice"]
X_train=train_df.drop(['SalePrice'],axis=1)
test_df=test_df.drop(['MSSubClass_90'],axis=1)

ridge = Ridge(alpha = 15)
# bagging 把很多小的分类器放在一起，每个train随机的一部分数据，然后把它们的最终结果综合起来（多数投票）
# bagging 算是一种算法框架
params = [1,10,15,20,25,30,40]
test_scores = []
for param in params:
    clf = BaggingRegressor(base_estimator = ridge,n_estimators = param)
    test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv = 10,scoring = 'neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

plt.plot(params,test_scores)
plt.title('n_estimators vs CV Error')
plt.show()

br = BaggingRegressor(base_estimator = ridge,n_estimators = 25)
br.fit(X_train,y_train)
y_final = np.expm1(br.predict(test_df))