# 调整数据每一行矢量距离为1
from pandas import read_csv
from numpy import set_printoptions

from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn.preprocessing import MinMaxScaler
print("--------以行为单位处理数据，调整后，每一行数据矢量距离(每一行数据向量长度)为1 ----------------------------")

listdata=np.array([[1, 2],[2, 3],[3, 5]])
# listdata=np.array([[2],[3],[5]])
print(listdata)
scaler = Normalizer().fit(listdata)
# 数据转换
rescaledX = scaler.transform(listdata)
# 设定数据的打印格式
# set_printoptions(precision=3)
print(rescaledX)
