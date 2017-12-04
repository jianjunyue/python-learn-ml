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

# train_shop=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/训练数据-ccf_first_round_shop_info.csv")
train_user_shop=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/商场中精确定位用户所在店铺/训练数据-ccf_first_round_user_shop_behavior_21.csv")

print("----------------train_new_user_shop-------------")
print(train_user_shop.head(1))
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