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

max_features = [.1,.3,.5,.7,.9,.99]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators = 200,max_features = max_feat)
    test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv = 5,scoring = 'neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(max_features,test_scores)
plt.title('Max Features vs CV Error')
plt.show()