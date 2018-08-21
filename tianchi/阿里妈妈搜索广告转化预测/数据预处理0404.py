from pandas import read_table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from matplotlib import pyplot
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import MinMaxScaler

# 导入数据
# filename = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/阿里妈妈搜索广告转化预测/round1_ijcai_18_train_20180301.txt'
# train_data =  read_table(filename,header=0,delim_whitespace=True)

filename = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/阿里妈妈搜索广告转化预测/round1_ijcai_18_test_a_20180301.txt'
test_data =  read_table(filename,header=0,delim_whitespace=True)
# print(data.head())

# print(test_data.columns)
# print(test_data["item_category_list"])

dict_item_category={}
# for item_category in train_data["item_category_list"].values :
#     # print(type(item_category))
#     # print(item_category)
#     categorys= item_category.split(";")
#     for category in categorys:
#         if len(category)>0:
#             dict_item_category[category]="item_category_list_"+category

for item_category in test_data["item_category_list"].values :
    # print(type(item_category))
    # print(item_category)
    categorys= item_category.split(";")
    for category in categorys:
        if len(category)>0:
            dict_item_category["item_category_list_"+category]=0

print(dict_item_category)

# train_data["sale_date"]=train_data["sale_date"].apply(lambda x:getMonth(x))
#
# def getItem_category_list(item_category_list):
#     dict_category = {}
#     categorys = item_category.split(";")
#     for category in categorys:
#         if dict_item_category.get("item_category_list_"+category)==None:
#             dict_category["item_category_list_"+category]=0
#         else:
#             dict_category["item_category_list_"+category]=1
#     return dict_category

category_df = pd.DataFrame()
# category_Series= test_data["item_category_list"].apply(lambda x:getItem_category_list(x))
for data in test_data.values :
    dict_category = dict_item_category
    print(data)
    print(type(data))
    categorys = item_category["item_category_list"].split(";")
    # instance_id=item_category["instance_id"]
    # dict_category["instance_id"]=instance_id
    # for category in categorys:
    #     if len(category)>0:
    #         dict_category["item_category_list_"+category]=1
    # category_df=category_df.append(pd.DataFrame.from_dict(dict_item_category, orient='index').T)
    # category_df.append(temp)

# print(category_df.head())
# result = test_data.join(category_df, on='key')
# # category_df.merge(category_Series)
# print("--------------------------------")
# # category_df = pd.DataFrame(category_Series)
# print(test_data.head())
