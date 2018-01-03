# 调整数据尺度（0..）
from pandas import read_csv
from numpy import set_printoptions

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("--------以列为单位处理数据，处理后每一列均值(和)=0，方差=1. "
      "----------------------------")

# listdata=np.array([[1, 2],[2, 3],[3, 5]])

listdata=np.array([[4],[6],[10]])
print(listdata)
scaler = StandardScaler().fit(listdata)
# 数据转换
rescaledX = scaler.transform(listdata)
print(rescaledX)
print("-------------------------------")
listdata=np.array([[2],[3],[5]])
rescaledX = scaler.transform(listdata)
print(rescaledX)
