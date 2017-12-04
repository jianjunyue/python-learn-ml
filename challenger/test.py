import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt

stock_test_data=pd.read_csv("/Users/jianjun.yue/PycharmProjects/data/stock_test_data_20170901.csv")

print(stock_test_data.info())
print("----------------------------------------------------------------------------------------------------------")
print(stock_test_data.describe())
# stock=stock_data.head(10000)
# plt.scatter(stock["id"], stock["feature2"])
# plt.show()