from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss
from ModelEnsemble import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import math
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import math

# train_shop=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/训练数据-ccf_first_round_shop_info.csv")
train_user_shop=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/训练数据-ccf_first_round_user_shop_behavior.csv")
# test_user_shop=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/AB榜测试集-evaluation_public.csv")

# train_shop['shop_id'] = train_shop["shop_id"].apply(lambda x: x.replace("s_",""))
# train_shop['category_id'] = train_shop["category_id"].apply(lambda x: x.replace("c_",""))
# train_shop['mall_id'] = train_shop["mall_id"].apply(lambda x: x.replace("m_",""))

train_user_shop['user_id'] = train_user_shop["user_id"].apply(lambda x: x.replace("u_",""))
train_user_shop['shop_id'] = train_user_shop["shop_id"].apply(lambda x: x.replace("s_",""))
# train_user_shop=train_user_shop.drop(["wifi_infos"],axis=1)
weekday= train_user_shop["time_stamp"].apply(lambda x: datetime.datetime.strptime(x.split(" ")[0], "%Y-%m-%d").date().weekday())
hour= train_user_shop["time_stamp"].apply(lambda x: x.split(" ")[1].split(":")[0])
minute= train_user_shop["time_stamp"].apply(lambda x: x.split(" ")[1].split(":")[1])
train_user_shop["weekday"]=weekday
train_user_shop["hour"]=hour
train_user_shop["minute"]=minute

wifi_count=train_user_shop["wifi_infos"].apply(lambda x: x.split(";").__len__())
train_user_shop["wifi_count"]=wifi_count

train_shop=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/训练数据-ccf_first_round_shop_info.csv")
train_shop['shop_id'] = train_shop["shop_id"].apply(lambda x: x.replace("s_","")).astype(int)
train_shop['category_id'] = train_shop["category_id"].apply(lambda x: x.replace("c_",""))
train_shop['mall_id'] = train_shop["mall_id"].apply(lambda x: x.replace("m_",""))
train_shop['shop_longitude'] =train_shop['longitude']
train_shop['shop_latitude'] =train_shop['latitude']

train_shop_temp=pd.DataFrame(index=train_shop["shop_id"])

train_shop_temp["category_id"]=train_shop["category_id"].values
train_shop_temp["shop_longitude"]=train_shop["shop_longitude"].values
train_shop_temp["shop_latitude"]=train_shop["shop_latitude"].values
train_shop_temp["price"]=train_shop["price"].values
train_shop_temp["mall_id"]=train_shop["mall_id"].values

train_user_shop['shop_id'] = train_user_shop['shop_id'].astype(int)
train_user_shop_join = train_user_shop.join(train_shop_temp, on='shop_id')


PI = 3.14159265358979323846
EARTH_RADIUS = 6378.137;# 地球半径
def rad(d):
    return d * PI / 180.0

def GetPreciseDistance(lat1,   lng1,   lat2,   lng2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    s = s * EARTH_RADIUS
    s = s*1000
    return s

user_shop_Distance=[]
for train_user_shop_Distance in train_user_shop_join[["latitude","shop_longitude","shop_latitude","shop_longitude"]].values:
    distance=GetPreciseDistance(train_user_shop_Distance[0],train_user_shop_Distance[1],train_user_shop_Distance[2],train_user_shop_Distance[3])
    user_shop_Distance.append(distance)
train_user_shop_join["user_shop_distance"]=user_shop_Distance
user_shop_distance_int=train_user_shop_join["user_shop_distance"].apply(lambda x: round(x))
train_user_shop_join["user_shop_distance_int"]=user_shop_distance_int

train_user_shop_join.to_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/训练数据-ccf_first_round_user_shop_behavior_21.csv",index=False)

print("----------------train_new_user_shop-------------")
print(train_user_shop.head(10))
# print("-----------------------------")
# print(train_user_shop.head(3))
print("----------------values-------------")
# print(train_shop["category_id"].values())
print("---------------sort_values--------------")
# sort_values=train_shop.sort_values(by='category_id', ascending=False)
# print(sort_values)
# print(train_shop["category_id"].value_counts())


colormap = plt.cm.viridis
plt.figure(figsize=(8,6))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
# sns.heatmap(train_shop.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# plt.show()