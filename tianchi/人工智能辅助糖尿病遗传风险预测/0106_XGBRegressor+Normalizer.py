import numpy as np
import pandas as pd
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# model
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

sub_0105_XGBRegressor_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/sub_0105.csv',names = ['sub_0105_pre'] )
print(sub_0105_XGBRegressor_df.head())

sub_0106_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/sub_0106_XG_Normalizer.csv',names = [ 'sub_0106_pre'] )
print(sub_0106_df.head())

join_df=sub_0105_XGBRegressor_df.join(sub_0106_df)
print(join_df.head())

join_df["sub_0106_join_pre"]=(join_df["sub_0105_pre"]+join_df["sub_0106_pre"])/2

print(join_df.head())
join_df=join_df[["sub_0106_join_pre"]]
print(join_df.head())
join_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/sub_0106_XGBRegressor+Normalizer.csv',header=False, index=False, float_format='%.3f')