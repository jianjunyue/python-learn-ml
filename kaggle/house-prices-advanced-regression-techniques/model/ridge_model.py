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
# print(X_train.isnull().sum().sort_values(ascending = False))
alphas = np.logspace(-3,2,50)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv = 10,scoring = 'neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(alphas,test_scores)
plt.title('Alpha vs CV Error')
plt.show()