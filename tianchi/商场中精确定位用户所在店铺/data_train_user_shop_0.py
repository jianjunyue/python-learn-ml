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


import seaborn as sns
import matplotlib.pyplot as plt

# train_shop=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/训练数据-ccf_first_round_shop_info.csv")
train_user_shop=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/训练数据-ccf_first_round_user_shop_behavior.csv")
# test_user_shop=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/AB榜测试集-evaluation_public.csv")

# train_shop['shop_id'] = train_shop["shop_id"].apply(lambda x: x.replace("s_",""))
# train_shop['category_id'] = train_shop["category_id"].apply(lambda x: x.replace("c_",""))
# train_shop['mall_id'] = train_shop["mall_id"].apply(lambda x: x.replace("m_",""))

train_user_shop['user_id'] = train_user_shop["user_id"].apply(lambda x: x.replace("u_",""))
train_user_shop['shop_id'] = train_user_shop["shop_id"].apply(lambda x: x.replace("s_",""))

print(train_user_shop.head(3))
train_new_user_shop = pd.DataFrame(columns=['user_id','shop_id','time_stamp','longitude','latitude','bssid','signal','used']);
print(train_new_user_shop.head(3))
user_id_array=[]
shop_id_array=[]
time_stamp_array=[]
longitude_array=[]
latitude_array=[]
bssid_array=[]
signal_array=[]
used_array=[]
for train_user_shop_row in train_user_shop.values:
    wifi_infos=train_user_shop_row[5]
    wifi_info_list= wifi_infos.split(";")
    for wifi_info in wifi_info_list:
        bssid,signal,used=wifi_info.split("|")
        # used= 1 if used == "true" else 0
        user_id_array.append(train_user_shop_row[0])
        shop_id_array.append(train_user_shop_row[1])
        time_stamp_array.append(train_user_shop_row[2])
        longitude_array.append(train_user_shop_row[3])
        latitude_array.append(train_user_shop_row[4])
        bssid_array.append(bssid.replace("b_",""))
        signal_array.append(signal)
        used_array.append(1 if used == "true" else 0)
        # new_row= pd.DataFrame({"user_id":train_user_shop_row[0],"shop_id":train_user_shop_row[1],"time_stamp":train_user_shop_row[2],"longitude":train_user_shop_row[3],"latitude":train_user_shop_row[4],"bssid":bssid.replace("b_",""),"signal":signal,"used":used},index=["0"])
        # train_new_user_shop=train_new_user_shop.append(new_row,ignore_index=True)
        # new_row={"user_id":train_user_shop_row[0],"shop_id":train_user_shop_row[1],"time_stamp":train_user_shop_row[2],"longitude":train_user_shop_row[3],"latitude":train_user_shop_row[4],"bssid":bssid.replace("b_",""),"signal":signal,"used":used}
        # print(type(new_row))
        # print("-----------------------------------------")
        # print(train_new_user_shop.head())
train_new_user_shop["user_id"] =user_id_array
train_new_user_shop["shop_id"] =shop_id_array
train_new_user_shop["time_stamp"] =time_stamp_array
train_new_user_shop["longitude"] =longitude_array
train_new_user_shop["latitude"] =latitude_array
train_new_user_shop["bssid"] =bssid_array
train_new_user_shop["signal"] =signal_array
train_new_user_shop["used"] =used_array

train_new_user_shop.to_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/训练数据-ccf_first_round_user_shop_behavior_new.csv",index=False)

print("----------------train_new_user_shop-------------")
print(train_new_user_shop.head(10))
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