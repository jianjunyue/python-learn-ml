# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/house_price/model/house_price_train.csv",index_col = 0)
test_df = pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/house_price/model/house_price_test.csv",index_col = 0)
numeric_cols = train_df.columns[train_df.dtypes != 'object']
y_train=train_df["SalePrice"]
X_train=train_df.drop(['SalePrice'],axis=1)
test_df=test_df.drop(['MSSubClass_90'],axis=1)

print(train_df.columns[X_train.dtypes != 'object'])
print("------------------------------")
print(train_df.columns[test_df.dtypes != 'object'])

ridge = Ridge(alpha = 15)
rf = RandomForestRegressor(n_estimators = 500,max_features = .3)
ridge.fit(X_train,y_train)
rf.fit(X_train,y_train)

y_ridge = np.expm1(ridge.predict(test_df))
y_rf = np.expm1(rf.predict(test_df))

y_final = (y_ridge + y_rf) / 2

print(y_final)