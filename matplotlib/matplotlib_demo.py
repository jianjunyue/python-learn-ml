import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model

#----------------多个线性图--------------------
t =np.array([0,1,2,3,4,5,6,7,8,9,10])
print(type(t))

# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# plt.title('线性图 Title in a custom color',color='#123456')
# plt.plot(t, t, 'r--')
# plt.plot(t, t**2, 'bs')
# plt.show()

#----------------图例标签说明--------------------
#legend 图例标签说明 函数
x=np.array([1,2,3,4])
line_up, =plt.plot(x,np.array([1,5,3,7]), label='Line 2')
line_down, =plt.plot(x,np.array([3,6,1,9]), label='Line 1')
#loc=1 在右上角 =2左上角 =3 左下角 =4 右下角 一般 =‘best’ 自动调整
plt.legend(handles =[line_up,line_down] , labels=['up','down'], loc = "best")
plt.show()

#----------------直方图--------------------
#直方图 函数：hist
# data = np.random.randn(10)
# print(data)
# plt.hist(data)
# plt.show()

#----------------条形图--------------------
# data = np.array([6,17,28])
# print(data)
# x = np.arange(len(data))
# # x = ["a","b","c"]
# plt.plot(x, data, color = 'r')
# plt.bar(x, data,  color = 'g')
# plt.show()

#----------------盒须图--------------------
#盒须图 函数：boxplot
# data = np.random.randn(10)
# print(data)
# plt.boxplot(data)
# plt.show()

#----------------饼图-------------------
# labels='frogs','hogs','dogs','logs'
# sizes=15,20,45,10
# colors='yellowgreen','gold','lightskyblue','lightcoral'
# explode=0,0.1,0,0
# plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
# plt.axis('equal')
# plt.show()

