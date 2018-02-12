from pandas import read_csv
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
filename = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/乘用车零售量预测/yancheng_train_20171226.csv'
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename,header=0)
data.loc[data["gearbox_type"] == 'AT', "gearbox_type"] = 0
data.loc[data["gearbox_type"] == 'MT', "gearbox_type"] = 1
data.loc[data["gearbox_type"] == 'DCT', "gearbox_type"] = 2
data.loc[data["gearbox_type"] == 'CVT', "gearbox_type"] = 3
data.loc[data["gearbox_type"] == 'AMT', "gearbox_type"] = 4
data.loc[data["gearbox_type"] == 'AT;DCT', "gearbox_type"] = 5
data.loc[data["gearbox_type"] == 'MT;AT', "gearbox_type"] = 6
data['gearbox_type'] = data['gearbox_type'].astype(int)

data.loc[data["if_charging"] == 'L', "if_charging"] =0
data.loc[data["if_charging"] == 'T', "if_charging"] =1
data['if_charging'] = data['if_charging'].astype(int)

data.loc[data["price_level"] == '10-15W', "min_price_level"] =10
data.loc[data["price_level"] == '10-15W', "max_price_level"] =15
data.loc[data["price_level"] == '15-20W', "min_price_level"] =15
data.loc[data["price_level"] == '15-20W', "max_price_level"] =20
data.loc[data["price_level"] == '35-50W', "min_price_level"] =30
data.loc[data["price_level"] == '35-50W', "max_price_level"] =50
data.loc[data["price_level"] == '20-25W', "min_price_level"] =20
data.loc[data["price_level"] == '20-25W', "max_price_level"] =25
data.loc[data["price_level"] == '8-10W', "min_price_level"] =8
data.loc[data["price_level"] == '8-10W', "max_price_level"] =10
data.loc[data["price_level"] == '5-8W', "min_price_level"] =5
data.loc[data["price_level"] == '5-8W', "max_price_level"] =8
data.loc[data["price_level"] == '25-35W', "min_price_level"] =25
data.loc[data["price_level"] == '25-35W', "max_price_level"] =35
data.loc[data["price_level"] == '5WL', "min_price_level"] =0
data.loc[data["price_level"] == '5WL', "max_price_level"] =5
data.loc[data["price_level"] == '50-75W', "min_price_level"] =50
data.loc[data["price_level"] == '50-75W', "max_price_level"] =75
data['max_price_level'] = data['max_price_level'].astype(int)
data['min_price_level'] = data['min_price_level'].astype(int)
data = data.drop(["price_level"], axis=1)

data.loc[data["level_id"] == '-', "level_id"] =0
data['level_id'] = data['level_id'].astype(int)
data.loc[data["TR"] == '5;4', "TR"] =2
data.loc[data["TR"] == '8;7', "TR"] =3
data['TR'] = data['TR'].astype(int)

data.loc[data["price"] == '-', "price"] =(data["min_price_level"]+data["max_price_level"])/2
data['price'] = data['price'].astype(float)

data.loc[data["power"] == '81/70', "power"] =75.5
data['power'] = data['power'].astype(float)

data.loc[data["engine_torque"] == '-', "engine_torque"] =202.5
data.loc[data["engine_torque"] == '155/140', "engine_torque"] =147.5
data['engine_torque'] = data['engine_torque'].astype(float)

data.loc[data["rated_passenger"] == '7-8', "rated_passenger"] =7.5
data.loc[data["rated_passenger"] == '6-7', "rated_passenger"] =6.5
data.loc[data["rated_passenger"] == '6-8', "rated_passenger"] =7
data.loc[data["rated_passenger"] == '4-5', "rated_passenger"] =4.5
data.loc[data["rated_passenger"] == '5-8', "rated_passenger"] =6.5
data.loc[data["rated_passenger"] == '5-7', "rated_passenger"] =6
data['rated_passenger'] = data['rated_passenger'].astype(float)

data.loc[data["fuel_type_id"] == '-', "fuel_type_id"] =4
data['fuel_type_id'] = data['fuel_type_id'].astype(int)

#201201开始，计算月差 201304
def getMonth(sale_date):
    sale_date=sale_date-201201
    year=sale_date/100
    count=int(year)*12
    month=year-int(year)
    count =count+month*100
    return int(count)

data["sale_date"]=data["sale_date"].apply(lambda x:getMonth(x))
data['sale_date'] = data['sale_date'].astype(int)

# for key in data.columns :
#     print("-----------"+key+"----------")
#     print(data[key].value_counts())

# data=data.drop(["price_level"])
quantity = [attr for attr in data.columns if data.dtypes[attr] == 'object']
print(quantity)
for key in quantity:
    print("-----------"+key+"----------")
    print(data.dtypes[key])
    print(data[key].value_counts())
# print(data["fuel_type_id"].value_counts())
data.to_csv('/Users/jianjun.yue/PycharmGItHub/data/tianchi/乘用车零售量预测/yancheng_train_20171226_0211_处理.csv',index=False)