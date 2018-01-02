# 二值数据
from pandas import read_csv
from numpy import set_printoptions
import numpy as np
from sklearn.preprocessing import Binarizer
# 导入数据

print("--------二值数据 以列为单位处理数据。列的转化值=列的实际值>threshold?1:0----------------------------")
listdata=np.array([[1, 2],[2, 3],[3, 5]])
listdata=np.array([[2],[3],[5]])
print(listdata)
transform = Binarizer(threshold=3.0).fit(listdata)   # threshold 0，1二值划分阈值
# 数据转换
newX = transform.transform(listdata)
# 设定数据的打印格式
set_printoptions(precision=3)
print(newX)
