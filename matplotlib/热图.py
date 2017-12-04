import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 10、热图
# data = np.random.rand(4, 2)
data1=[1,4]
data2=[3,4]
data3=[2,3]
data4=[1,3]
data=[data1,data2,data3,data4]
print(data)
rows = list('1234')
columns = list('MF')
fig, ax = plt.subplots()
ax.pcolor(data, cmap=plt.cm.Reds, edgecolor='k')
ax.set_xticks(np.arange(0, 2)+0.5)
ax.set_yticks(np.arange(0, 4)+0.5)
# ax.xaxis.tick_bottom()
# ax.yaxis.tick_left()
ax.set_xticklabels(columns, minor=False, fontsize=20)
ax.set_yticklabels(rows, minor=False, fontsize=20)
plt.show()