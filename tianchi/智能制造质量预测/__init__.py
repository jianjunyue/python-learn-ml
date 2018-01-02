import numpy as np
import pandas as pd
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# model
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from pandas import Series, DataFrame
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split


series_3=pd.Series([1,2,'3',4,'a'])
print(list(series_3))
print(len(list(series_3)))
print(type(series_3.shape))