import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss
from ModelEnsemble import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

#https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/notebook
#https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.522056b3Q4H7rC&raceId=231620

train_user_shop=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/训练数据-ccf_first_round_user_shop_behavior_new_datetime.csv")
# train_user_shop=train_user_shop.drop(["Unnamed: 0","Unnamed: 0.1"])
print(train_user_shop.columns)
print("----------------------")
train_shop=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/训练数据-ccf_first_round_shop_info.csv")
train_shop['shop_id'] = train_shop["shop_id"].apply(lambda x: x.replace("s_","")).astype(int)
train_shop['category_id'] = train_shop["category_id"].apply(lambda x: x.replace("c_",""))
train_shop['mall_id'] = train_shop["mall_id"].apply(lambda x: x.replace("m_",""))
train_shop['shop_longitude'] =train_shop['longitude']
train_shop['shop_latitude'] =train_shop['latitude']
# print(train_shop.head())

train_shop_temp=pd.DataFrame(index=train_shop["shop_id"])

train_shop_temp["category_id"]=train_shop["category_id"].values
train_shop_temp["shop_longitude"]=train_shop["shop_longitude"].values
train_shop_temp["shop_latitude"]=train_shop["shop_latitude"].values
train_shop_temp["price"]=train_shop["price"].values
train_shop_temp["mall_id"]=train_shop["mall_id"].values
print(train_shop_temp.head())
print("----------------------")


train_user_shop['shop_id'] = train_user_shop['shop_id'].astype(int)
train_user_shop_info = train_user_shop.join(train_shop_temp, on='shop_id')
print(train_user_shop_info.head())


train_user_shop_info.to_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/训练数据-ccf_first_round_user_shop_behavior_new_datetime_shopinfo.csv",index=False)

print("----------------------")