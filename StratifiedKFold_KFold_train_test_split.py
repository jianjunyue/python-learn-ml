import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split

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
    [9],
    [10],
    [11],
    [12],
    [13],
    [14],
    [15],
    [16],
    [17],
    [18],
    [19],
    [20],
    [21],
    [22],
    [23],
    [24],
    [25],
    [26],
    [27],
    [28],
    [29],
    [30],
    [31],
    [32],
    [33],
    [34]
])

y=np.array([1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0])
#n_folds这个参数没有，引入的包不同，
floder = KFold(n_splits=4,random_state=0,shuffle=False) #分块划分
sfolder = StratifiedKFold(n_splits=4,random_state=0,shuffle=False) #分层划分
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0) # 一次性按test_size比例随机划分

print('train_test_split - X_train:')
print((X_train))
print('train_test_split - X_test:' )
print((X_test))
print('train_test_split - y_train:' )
print((y_train))
print('train_test_split - y_test:' )
print((y_test))
print(" ")
print("-----------------------------")

for train, test in sfolder.split(X,y):
    print('StratifiedKFold - train %s | test: %s' % (train, test))
    print(" ")

print("-----------------------------")
for train, test in floder.split(X,y):
    print('KFold - train %s | test: %s' % (train, test))
    print(" ")

