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
import datetime
#https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/notebook

train_user_shop=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/训练数据-ccf_first_round_user_shop_behavior_new.csv")
# time_stamp=train_user_shop['time_stamp']

# date= train_user_shop["time_stamp"].apply(lambda x: x.split(" ")[0])
weekday= train_user_shop["time_stamp"].apply(lambda x: datetime.datetime.strptime(x.split(" ")[0], "%Y-%m-%d").date().weekday())
hour= train_user_shop["time_stamp"].apply(lambda x: x.split(" ")[1].split(":")[0])
minute= train_user_shop["time_stamp"].apply(lambda x: x.split(" ")[1].split(":")[1])
train_user_shop["weekday"]=weekday
train_user_shop["hour"]=hour
train_user_shop["minute"]=minute
# train_user_shop.drop(["time_stamp"])
print(train_user_shop.head())
train_user_shop.to_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/训练数据-ccf_first_round_user_shop_behavior_new_datetime.csv",index=False)

# colormap = plt.cm.viridis
# plt.figure(figsize=(10,9))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# sns.heatmap(train_user_shop.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# plt.show()