import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.preprocessing import Imputer
# from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

train_df = pd.read_csv('../../../data/house_price_train.csv')
test_df = pd.read_csv('../../../data/house_price_test.csv')

submission_filename = "../../../data/house_price/sample_Submission"

numeric_cols = train_df.columns[train_df.dtypes != 'object']
numeric_cols=numeric_cols.drop("SalePrice")
x_train =train_df[numeric_cols]

# target=train_df["SalePrice"]
prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price + 1)":np.log1p(train_df["SalePrice"])})
y_train=np.log1p(train_df.pop('SalePrice'))
# mean_cols = x_train.mean()
x_train=x_train.fillna(0)
# numeric_col_means = x_train.loc[:, numeric_cols].mean()
# numeric_col_std = x_train.loc[:, numeric_cols].std()
# x_train.loc[:, numeric_cols] = (x_train.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std

test_df=test_df.fillna(0)
x_test=test_df[numeric_cols]

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=9)

xgb_params = {}

def loss_score(y_pre, y_test):
    loss_error = 0;
    z = zip(y_pre, y_test)
    for py, ty in z:
        ls=(py - ty)/ty
        if ls<0:
            ls=-1*ls
        loss_error+=ls
    return loss_error

def min_score_index(paras,scores):
    index=0
    min_score=9999
    for i in range(len(scores)):
        if scores[i]<min_score:
            min_score=scores[i]
            index=i
    return  min_score,index



params = [1,2,3,4,5,6,8,10,12,15,18,20]
test_scores = []
for param in params:
    xgb_params['max_depth'] = param
    clf = XGBRegressor(**xgb_params)
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    ls = loss_score(y_pre, y_test)
    test_scores.append(ls)

print(test_scores)
minscore,index=min_score_index(params,test_scores)
print("----------minscore-------")
print(minscore)
print("--------max_features---------")
print(params[index])

import matplotlib.pyplot as plt
plt.plot(params, test_scores)
plt.title("max_depth vs loss_score Error");
# plt.show()


xgb_params['max_depth'] = 4
clf = XGBRegressor(**xgb_params)
clf.fit(X_train,y_train)
y_pre=clf.predict(X_test)


ls=loss_score(y_pre,y_test)
print("----------------")
print(ls)

