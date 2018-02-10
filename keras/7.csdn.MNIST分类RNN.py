import numpy as np
import matplotlib.pyplot as plt
# Sequential按顺序构成的模型
from keras.models import Sequential
# Dense 全连接层
from keras.layers import Dense,Activation
from keras.optimizers import SGD,Adam
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers.recurrent import SimpleRNN,LSTM

#数据长度-- 一行有28个像素
input_size=28
#序列长度-- 每张图片有28行数据
time_steps=28
#隐藏层cell个数
cell_size=50

#导入数据
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#(60000, 28, 28)
x_train=x_train/255.0
x_test=x_test/255.0
# print(x_train)
#label 换one hot 格式
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)


#创建模型，输入784个神经元，输出10个神经元
#bias_initializer 偏置值，activation激活函数
model=Sequential()

model.add(SimpleRNN(
    units=cell_size, #输出
    input_shape=(time_steps,input_size)  #输入（时间段数，每次特征数）
                    ))
model.add(Dense(10,activation="softmax"))

#定义优化器
# opt=SGD(lr=0.2)
opt=Adam(lr=0.001)

model.compile(
    optimizer=opt,
    loss="categorical_crossentropy", # mse: 均方差;categorical_crossentropy:交叉熵
    metrics=["accuracy"]
)

#训练模型
model.fit(x_train,y_train,batch_size=32,epochs=10)
# model.predict()
#评估模型
loss,accuracy=model.evaluate(x_test,y_test)

print("\ntest loss",loss)
print("accuracy",accuracy)