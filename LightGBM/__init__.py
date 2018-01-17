# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

#LightGBM： https://github.com/Microsoft/LightGBM

# 安装步骤：
#
# 1、安装Anaconda3
#
# 2、pip install lightgbm