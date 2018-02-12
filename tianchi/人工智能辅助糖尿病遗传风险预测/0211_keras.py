from pandas import read_csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
# Sequential按顺序构成的模型
from keras.models import Sequential
# Dense 全连接层
from keras.layers import Dense,Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
import seaborn as sns
def rename_columns(pre_name,columns_name):
    name_dict={}
    for name in columns_name:
        name_dict[name]=pre_name+name
    return name_dict
# 导入数据
filename = '/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/d_train_20180102.csv'
data = read_csv(filename,header=0,encoding='GB2312')

data['year'] = data["体检日期"].apply(lambda x: x.split('/')[2])
data['year'] = data['year'].astype(int)
data['month'] = data["体检日期"].apply(lambda x: x.split('/')[1])
data['month'] = data['month'].astype(int)
data['day'] = data["体检日期"].apply(lambda x: x.split('/')[0])
data['day'] = data['day'].astype(int)

pclass_dummies_titanic  = pd.get_dummies(data['性别'])
occ_cols = ['性别_' +  columns_name for columns_name in pclass_dummies_titanic.columns]
pclass_dummies_titanic.rename(columns=rename_columns('性别_',pclass_dummies_titanic.columns), inplace = True)
pclass_dummies_titanic = pclass_dummies_titanic.drop(["性别_??"], axis=1)
data = data.join(pclass_dummies_titanic)
#
test = read_csv("/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/d_test_A_20180102.csv",header=0,encoding='GB2312')
test['year'] = test["体检日期"].apply(lambda x: x.split('/')[2])
test['year'] = test['year'].astype(int)
test['month'] = test["体检日期"].apply(lambda x: x.split('/')[1])
test['month'] = test['month'].astype(int)
test['day'] = test["体检日期"].apply(lambda x: x.split('/')[0])
test['day'] = test['day'].astype(int)
pclass_dummies_titanic  = pd.get_dummies(test['性别'])
occ_cols = ['性别_' +  columns_name for columns_name in pclass_dummies_titanic.columns]
pclass_dummies_titanic.rename(columns=rename_columns('性别_',pclass_dummies_titanic.columns), inplace = True)
test = test.join(pclass_dummies_titanic)

test = test.drop(["id","体检日期","性别","year"], axis=1)
# # 将数据分为输入数据和输出结果
X = data.drop(["id","血糖","体检日期","性别","year"], axis=1)
Y = data["血糖"]
Y=np.log1p(Y)
mean_cols = X.mean()
X = X.fillna(mean_cols)
print(len(X.columns))

X = X.values
Y = Y.values
X = X[:,0:42]
# Y = Y[:,0]

print(X)
print("---------------------------")
print(Y)

model = Sequential()
# 定义第一层, 由于是回归模型, 因此只有一层
model.add(Dense(units = 1, input_dim = 42))

# 选择损失函数和优化方法
model.compile(loss = 'mse', optimizer = 'sgd')

print('----Training----')
# 训练过程
for step in range(501):
    # 进行训练, 返回损失(代价)函数
    cost = model.train_on_batch(X , Y)
    if step % 100 == 0:
        print('loss: ', cost)

print('----Testing----')
# 训练结束进行测试
# cost = model.evaluate(X_test, Y_test, batch_size = 40)
# print 'test loss: ', cost