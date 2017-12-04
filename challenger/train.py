import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt

stock_data=pd.read_csv("/Users/jianjun.yue/PycharmProjects/data/stock_train_data_20170901.csv")

print("-------------weight---------------------------------------------------------------------------------------------")
print(stock_data["weight"].value_counts())
print("-------------label---------------------------------------------------------------------------------------------")
print(stock_data["label"].value_counts())
print("-------------group---------------------------------------------------------------------------------------------")
print(stock_data["group"].value_counts())
print("-------------era---------------------------------------------------------------------------------------------")
print(stock_data["era"].value_counts())
# stock=stock_data.head(10000)
# plt.scatter(stock["id"], stock["feature2"])
# plt.show()

