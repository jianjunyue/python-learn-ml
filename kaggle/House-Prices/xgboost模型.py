import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import  SelectKBest,f_classif
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble  import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor

path="/Users/jianjun.yue/PycharmGItHub/data/kaggle/House-Prices/train_预处理.csv"
data=pd.read_csv(path)
path_test="/Users/jianjun.yue/PycharmGItHub/data/kaggle/House-Prices/test_预处理.csv"
data_test=pd.read_csv(path_test)

keyName="BsmtFinSF1"
data_test[keyName] = data_test[keyName].fillna(data_test[keyName].mean())
data_test[keyName] = data_test[keyName].apply(lambda x:int(x))

keyName="BsmtFinSF2"
data_test[keyName] = data_test[keyName].fillna(data_test[keyName].mean())
data_test[keyName] = data_test[keyName].apply(lambda x:int(x))

keyName="BsmtUnfSF"
data_test[keyName] = data_test[keyName].fillna(data_test[keyName].mean())
data_test[keyName] = data_test[keyName].apply(lambda x:int(x))
keyName="TotalBsmtSF"
data_test[keyName] = data_test[keyName].fillna(data_test[keyName].mean())
data_test[keyName] = data_test[keyName].apply(lambda x:int(x))

# print(data.columns.values.tolist())
columns=[]
for column in data.columns.values.tolist():
    if column!="Id":
        if column!="SalePrice":
            columns.append(column)

# print(columns)
predictors=columns
train=data[predictors]

y=data["SalePrice"]

alg=XGBRegressor(n_estimators=1000, learning_rate=0.05)
# alg=XGBRegressor(base_score=0.5, colsample_bylevel=1,
#        colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#        max_depth=3, min_child_weight=1, missing=None, n_estimators=20, objective='reg:linear',
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#        silent=True, subsample=1)
test=data_test[predictors]

kfold=StratifiedKFold(n_splits=5,random_state=1, shuffle=True)#分层划分
scores=cross_val_score(alg,train,y,cv=kfold)
print("------test_score--------")
print(scores)
print(np.mean(scores))

# kfold=StratifiedKFold(n_splits=3,random_state=1, shuffle=True)#分层划分
# scores=cross_val_score(alg,train,y,cv=3)
# print(scores)
# scores = np.sqrt(-cross_val_score(alg,train,y,cv=3, scoring='neg_mean_squared_error'))
# print(scores)
# print(test.isnull)
alg.fit(train,y)
pre=alg.predict(test)
print(pre)
y_pred=pd.DataFrame()
y_pred["Id"]=data_test["Id"]
y_pred["SalePrice"]=pre
y_pred.to_csv('/Users/jianjun.yue/PycharmGItHub/data/kaggle/House-Prices/submission.csv',index=None)

feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
print(type(feat_imp))
print(feat_imp)
#新版需要转换成dict or list
# feat_imp = pd.Series(dict(alg.get_booster().get_fscore())).sort_values(ascending=False)
# plt.bar(feat_imp.index, feat_imp)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()