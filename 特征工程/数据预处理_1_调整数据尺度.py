# 调整数据尺度（0..）
from pandas import read_csv
from numpy import set_printoptions
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# 导入数据

print("--------以列为单位处理数据，列的转化值=列的实际值/(每列的最大值-每列的最少值)----------------------------")

listdata=np.array([[1, 2],[2, 3],[3, 5]])
# listdata=np.array([[2],[3],[5]])
print(type(listdata))
print(listdata)
data_scaler = MinMaxScaler(feature_range=(0, 1))
# 数据转换
data_rescaledX = data_scaler.fit_transform(listdata)
# 设定数据的打印格式
# set_printoptions(precision=3)
print(data_rescaledX)

print("-------------------------------")

listdata=np.array([[1],[3],[4]])
print(listdata)
data_rescaledX = data_scaler.fit_transform(listdata)
# 设定数据的打印格式
# set_printoptions(precision=3)
print(data_rescaledX)
