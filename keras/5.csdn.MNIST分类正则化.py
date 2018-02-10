import numpy as np
import matplotlib.pyplot as plt
# Sequential按顺序构成的模型
from keras.models import Sequential
# Dense 全连接层
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils
from keras.regularizers import l2

#导入数据
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print("x_shape:",x_train.shape)
print("y_shape:",y_train.shape)
# print(x_train)
print(y_train)
#(60000, 28, 28) -> (60000, 784)
x_train=x_train.reshape(x_train.shape[0],-1)/255.0
x_test=x_test.reshape(x_test.shape[0],-1)/255.0
# print(x_train)
#label 换one hot 格式
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)
print(y_train)

#创建模型，输入784个神经元，输出10个神经元
#bias_initializer 偏置值，activation激活函数
model=Sequential([
    Dense(units=200,input_dim=784,bias_initializer="one",activation="tanh",kernel_initializer=l2(0.002)),
    Dense(units=50,bias_initializer="one",activation="tanh",kernel_initializer=l2(0.0002)),
    Dense(units=10,bias_initializer="one",activation="softmax") #softmax 一般用于最后输出层
])
#定义优化器
sgd=SGD(lr=0.2)

model.compile(
    optimizer=sgd,
    loss="categorical_crossentropy", # mse: 均方差;categorical_crossentropy:交叉熵
    metrics=["accuracy"]
)

#训练模型
model.fit(x_train,y_train,batch_size=32,epochs=10)
# model.predict()
#评估模型
loss,accuracy=model.evaluate(x_train,y_train)
print("\ntrain loss",loss)
print("train accuracy",accuracy)

loss,accuracy=model.evaluate(x_test,y_test)

print("\ntest loss",loss)
print("accuracy",accuracy)