import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

#自定义损失函数评分
clf = LogisticRegression()

X=np.array([
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9]
])
y=np.array([1,1,1,1,0,0,0,0,1,1])

clf.fit(X, y)
print("--------clf.score---------")
print(clf.score(X, y))
print("-----------------")

def loss(pre_y,test_y):
    score = pre_y == test_y
    error=np.average(score, weights=None)
    return error

#自定义损失函数评分
#实际为0，预测为1，loss 10分
#实际为1，预测为0，loss 2分
def loss_score(pre_y,test_y):
    loss_error=0;
    z=zip(pre_y, test_y)
    for py,ty in z:
        if(py!=ty):
            if(ty==0):
                loss_error += 10
            else:
                loss_error += 2
    return loss_error

pre_Y=clf.predict(X)
print("-------loss---------")
print(loss(pre_Y,y))
print("----------------")
print(pre_Y)
print("----------------")
print(y)
print("-------loss_score---------")
print(loss_score(pre_Y,y))

