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
df["count1"]= y

plt.figure(figsize=(8,6))

plt.scatter(range(len(df["key"].values)),  df["count"].values,s=30,c='red',marker='o',alpha=0.5,label='C1')
plt.scatter(range(len(df["key"].values)),  df["count1"].values,s=30,c='blue',marker='x',alpha=0.5,label='C2')
plt.xticks(range(len(df["key"].values)),df["key_name"].values) #给X轴赋值名称,没有就默认自适应
# plt.xticks(range(len(df["key"].values)),df["key_name"].values)
plt.legend(loc='upper right')
plt.title("test title")
plt.xlabel('test X');
plt.ylabel('test Y');
plt.show()