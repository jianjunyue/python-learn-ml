import xgboost as xgb
import numpy as np
import sys
import datetime

from xgboost.sklearn import XGBClassifier

import pandas as pd
# 导入数据
filename = 'out.txt'
data =pd.read_table(filename,header=0)
# print(type(data))
# print(data.head())
for row in data.values:
    # print(type(row))
    print(row[0].split('{')[0])
    print("{"+row[0].split('{')[1])
    print("----------")