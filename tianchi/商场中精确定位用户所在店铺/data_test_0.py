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

import datetime

import seaborn as sns
import matplotlib.pyplot as plt

test_user=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/AB榜测试集-evaluation_public.csv")


test_user['user_id'] = test_user["user_id"].apply(lambda x: x.replace("u_",""))
test_user['mall_id'] = test_user["mall_id"].apply(lambda x: x.replace("m_",""))

weekday= test_user["time_stamp"].apply(lambda x: datetime.datetime.strptime(x.split(" ")[0], "%Y-%m-%d").date().weekday())
hour= test_user["time_stamp"].apply(lambda x: x.split(" ")[1].split(":")[0])
minute= test_user["time_stamp"].apply(lambda x: x.split(" ")[1].split(":")[1])
test_user["weekday"]=weekday
test_user["hour"]=hour
test_user["minute"]=minute

print(test_user.head())

test_user.to_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/AB榜测试集-evaluation_public_new.csv",index=False)
