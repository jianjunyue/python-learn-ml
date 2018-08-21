import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

def iris_type(s):
    it = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    return it[s]

path = 'iris.data'  # 数据文件路径
data = pd.read_csv(path,header=None, converters={4: iris_type})
x_prime, y = np.split(data, (4,), axis=1)
# 随机森林 n_estimators表示子模型的数量 criterion表示特征选择标准 max_depth 最大深度
clf = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=3)
rf_clf = clf.fit(x_prime, y)
y_hat = clf.predict(x_prime)
y_hat = clf.predict_proba(x_prime)
print(y_hat)