import pandas as pd
import numpy as np
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# model
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from sklearn.preprocessing import Imputer

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

train_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练.xlsx',header=0,encoding='utf-8')
y_train=train_df["Y"]
train_df=train_df.drop(["ID","Y"], axis=1)
quantity = [attr for attr in train_df.columns if train_df.dtypes[attr] != 'object']  # 数值变量集合
print(len(quantity))
train_df=train_df[quantity]
X_train = Imputer().fit_transform(train_df)

max_features = [.1,.5,.9]
test_scores = []
print("test")
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators = 200,max_features = max_feat)
    test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv = 5,scoring = 'neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
    print(test_score)
plt.plot(max_features,test_scores)
plt.title('Max Features vs CV Error')
plt.show()

# xgb1.fit(X_train, y_train)
# predict_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A.xlsx',header=0,encoding='utf-8')
#
# predict_df=predict_df[quantity]
# predict_df = Imputer().fit_transform(predict_df)
# pred=xgb1.predict(predict_df)
# pred_df=pd.DataFrame()
# pred["pred"]=pred
# pred_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/tmp/测试A-RandomForest_pred.csv', index=False, float_format='%.4f')
print("---4----")
