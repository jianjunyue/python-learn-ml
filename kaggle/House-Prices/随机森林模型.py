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

alg=RandomForestRegressor()
alg=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=3,
           min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
test=data_test[predictors]

kfold=StratifiedKFold(n_splits=5,random_state=1, shuffle=True)#分层划分
scores=cross_val_score(alg,train,y,cv=kfold)
print("------test_score--------")
print(scores)
print(np.mean(scores))
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

print("------------特征重要性可视化显示-----------")
selector=SelectKBest(f_classif,k=5)
selector.fit(data[predictors],data["SalePrice"])
scores= -np.log10(selector.pvalues_)

plt.bar(range(len(predictors)),scores)
plt.xticks(range(len(predictors)),predictors,rotation="vertical")
plt.show()
