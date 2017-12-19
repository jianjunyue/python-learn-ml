import numpy as np
import pandas as pd
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# model
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split


train_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练.xlsx',header=0,encoding='utf-8')
y_train=train_df["Y"]
y_train = np.log1p(train_df.pop('Y'))
y_train.hist()
# plt.bar(left= range(len(y_train.values)), height=y_train.values, width=0.35, align="center", yerr=0.0001)
# plt.xticks(df["key"].values)
#
# plt.title("test title")
# plt.xlabel('test X');
# plt.ylabel('test Y');
plt.show()