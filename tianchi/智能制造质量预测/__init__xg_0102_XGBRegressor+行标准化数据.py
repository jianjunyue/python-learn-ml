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

sub_1225_XGBRegressor_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_sub_1225_XGBRegressor.csv',names = ['sub_1225_id', 'sub_1225_pre'] )
print(sub_1225_XGBRegressor_df.head())

sub_1231_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_sub_1231_行标准化数据.csv',names = ['sub_1231_id', 'sub_1231_pre'] )
print(sub_1231_df.head())

join_df=sub_1225_XGBRegressor_df.join(sub_1231_df)
print(join_df.head())

join_df["sub_0102_pre"]=(join_df["sub_1225_pre"]+join_df["sub_1231_pre"])/2

print(join_df.head())
join_df=join_df[["sub_1225_id","sub_0102_pre"]]
print(join_df.head())
join_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_sub_0102_XGBRegressor+行标准化数据.csv',header=False, index=False, float_format='%.9f')