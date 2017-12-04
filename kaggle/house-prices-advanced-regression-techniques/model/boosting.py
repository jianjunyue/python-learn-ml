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

# boosting 比bagging更高级，它是弄来一把分类器，把它们线性排列，下一个分类器把上一个分类器分类不好的地方加上更高的权重，这样，下一个分类器在这部分就能学习得更深刻

ridge = Ridge(alpha = 15)
params = [10,15,20,25,30,35,40,45,50]
test_scores = []
for param in params:
    clf = AdaBoostRegressor(base_estimator = ridge,n_estimators = param)
    test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv = 10,scoring = 'neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(params,test_scores)
plt.title('n_estimators vs CV Error')
plt.show()