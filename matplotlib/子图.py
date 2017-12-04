import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

x=["a1","a2","a3","a4","a5","a6","a7","a8"]
x1=[1,3,5,6,8,9,5,22]
y=[21,34,5,6,78,9,5,22]
df = pd.DataFrame(columns=['key', 'key_name', 'count'])
df["key"]=x
df["key_name"]=x
df["count"]=y

fig=plt.figure(figsize=(8,6))
ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2)
#第一个参数表示子图的列数；第二个参数表示子图的行数；第三个参数表示子图的位置
# ax1=fig.add_subplot(2,2,1)
# ax2=fig.add_subplot(2,2,2)
# ax3=fig.add_subplot(2,2,3)
# ax4=fig.add_subplot(2,2,4)

ax1.bar(left= range(len(df["key"].values)), height=df["count"].values, width=0.35, align="center", yerr=0.0001)
ax1.set_xticklabels(df["key_name"].values,rotation='1') #子图 用 set_xticklabels
ax1.set_title("ax1 title")
ax1.set_xlabel('ax1 X');
ax1.set_ylabel('ax1 Y');

ax2.scatter(range(len(df["key"].values)),  df["count"].values,s=30,c='red',marker='o',alpha=0.5,label='C1')
ax2.set_xticklabels(df["key_name"].values,rotation='1')
ax2.set_title("ax2 title")
ax2.set_xlabel('ax2 X');
ax2.set_ylabel('ax2 Y');

plt.show()