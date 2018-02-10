import numpy as np
import matplotlib.pyplot as plt
# Sequential按顺序构成的模型
from keras.models import Sequential
# Dense 全连接层
from keras.layers import Dense
from keras.models import load_model,model_from_json

# 使用numpy生产100个随机点
x_data=np.random.rand(100)
noise=np.random.normal(0,0.01,x_data.shape)
y_data=x_data*0.1+0.2+noise

#显示随机点(散点图)
# plt.scatter(x_data,y_data)
# plt.show()

#构建一个顺序模型
model=Sequential()
#在模型中添加一个全连接层
#input_dim输入维度，units输出维度
model.add(Dense(units=1,input_dim=1))
#optimizer 优化方式 sgd 随机梯度下降法
#mse 均方差
model.compile(optimizer="sgd",loss="mse")

for step in range(3001):
    # 每次训练一个批次
    cost=model.train_on_batch(x_data,y_data)
    if step%500==0:
        print("cost:",cost)

#打印权值和配置值
W,b=model.layers[0].get_weights()
print("W:",W,"b:",b)
#x_data输入网络中，得到预测值y_pred
y_pred=model.predict(x_data)
# plt.scatter(x_data,y_data)
# plt.plot(x_data,y_pred)
# plt.show()

#保存模型
model.save("model.h5")  #hdf5文件

#加载模型
model_load=load_model("model.h5")

#只保存模型参数
model.save_weights("model_weights.h5") #保存模型参数
model.load_weights("model_weights.h5") #加载模型参数

#保存网络结构和加载网络结构
model_json=model.to_json()
model=model_from_json(model_json)