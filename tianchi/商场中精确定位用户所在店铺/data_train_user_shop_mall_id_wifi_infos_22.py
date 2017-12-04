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

train_user_shop_all=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/训练数据-ccf_first_round_user_shop_behavior_21_sample.csv")
# print(train_user_shop_all[["mall_id"]])
mall_id_wifi_info={}
bssids={}
mall_id_wifis=train_user_shop_all[["mall_id","wifi_infos"]]
for train_user_shop_row in mall_id_wifis.values:
    mall_id=train_user_shop_row[0]
    wifi_infos=train_user_shop_row[1].split(";")
    # print(mall_id, wifi_infos)
    for wifi_info in wifi_infos:
        bssid, signal, used = wifi_info.split("|")
        bssid=bssid.replace("b_", "")
        bssids=mall_id_wifi_info.get(mall_id)
        if(bssids==None):
            bssids = {}
        bssids[bssid]="1"
        mall_id_wifi_info[mall_id]=bssids
        # print(mall_id_wifi_info)

for mall_id_key in mall_id_wifi_info.keys():
    # print(mall_id_key)
    train_user_shop=train_user_shop_all[train_user_shop_all["mall_id"]==mall_id_key]
    # print(train_user_shop.head(3))
    mall_id_wifis=train_user_shop[["mall_id","wifi_infos"]]
    bssids=mall_id_wifi_info.get(mall_id_key)
    mall_id_columns=['user_id','shop_id','weekday','hour','minute','wifi_count','category_id','price','user_shop_distance','user_shop_distance_int']
    mall_id_wifi_columns_bssid=[]
    for bssid in bssids.keys():
        mall_id_columns.append("bssid_"+str(bssid))
        mall_id_columns.append("signal_"+str(bssid))
        mall_id_columns.append("used_wifi_"+str(bssid))

        mall_id_wifi_columns_bssid.append(bssid)

    dict_columns_wifi_info={}
    data_array = []
    for train_user_shop_row in mall_id_wifis.values:
        # print(train_user_shop_row)
        wifi_info_list = train_user_shop_row[1].split(";")
        for wifi_info in wifi_info_list:
            if wifi_info=="":
                wifi_info=" | | "
            bssid, signal, used = wifi_info.split("|")
            bssid = bssid.replace("b_", "")
            used = 1 if used == "true" else 0
            for c_bssid in mall_id_wifi_columns_bssid:
                c_bssid_value=0
                c_signal_value=-9999
                c_used_value=0
                if(c_bssid==bssid):
                    c_bssid_value=1
                    c_signal_value=signal
                    c_used_value=used

                data_array = dict_columns_wifi_info.get( str("bssid_" + c_bssid))
                if (data_array == None):
                    data_array = []
                data_array.append(c_bssid_value)
                dict_columns_wifi_info[str("bssid_" + c_bssid)] = data_array

                data_array = dict_columns_wifi_info.get(str("signal_"+c_bssid))
                if (data_array == None):
                    data_array = []
                data_array.append(c_signal_value)
                dict_columns_wifi_info[str("signal_"+c_bssid)] = data_array

                data_array = dict_columns_wifi_info.get( str("used_wifi_"+c_bssid))
                if (data_array == None):
                    data_array = []
                data_array.append(c_used_value)
                dict_columns_wifi_info[str("used_wifi_"+c_bssid)] = data_array


    # print(mall_id_wifi_columns)
    train_user_shop_in_mall_id=pd.DataFrame();
    columns_temp= ['user_id', 'shop_id', 'weekday', 'hour', 'minute', 'wifi_count', 'category_id', 'price', 'user_shop_distance','user_shop_distance_int']
    for column in columns_temp :
        # print(type(train_user_shop[column]))
        train_user_shop_in_mall_id[column]=train_user_shop[column]

    for mall_id_wifi_column in mall_id_wifi_columns_bssid:
        # print(type(dict_columns_wifi_info.get(mall_id_wifi_column)))
        train_user_shop_in_mall_id["bssid_" +mall_id_wifi_column]=pd.Series(dict_columns_wifi_info.get("bssid_" +mall_id_wifi_column))
        train_user_shop_in_mall_id["signal_"+mall_id_wifi_column]=pd.Series(dict_columns_wifi_info.get("signal_"+mall_id_wifi_column))
        train_user_shop_in_mall_id["used_wifi_"+mall_id_wifi_column]=pd.Series(dict_columns_wifi_info.get("used_wifi_"+mall_id_wifi_column))
    print(train_user_shop_in_mall_id.head(3))
    print("mall_id",mall_id_key)
    print("------------------------------------------------------------------")
    train_user_shop_in_mall_id.to_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/mall_id/训练数据-mall_"+str(mall_id_key)+".csv",index=False)











