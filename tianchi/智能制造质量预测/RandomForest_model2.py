import pandas as pd
import numpy as np
#Python sklearn数据分析中常用方法
#http://blog.csdn.net/qq_16234613/article/details/76534673

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
X_train=pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练_quantity_2017121622.csv',header=0)
# X_train.shape
y_train=pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练_y_train_20171216.csv',header=0)
#交叉验证，获取最优参数( 最终使用GridSearch获取最优参数 )
max_features = [.1,.5,.9]
test_scores = []
print("test")
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators = 200,max_features = max_feat)
    test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv = 5,scoring = 'neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
    print(test_score)
plt.plot(max_features,test_scores)
plt.title('Max Features vs CV Error')
plt.show()