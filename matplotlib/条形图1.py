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

plt.bar(left= range(len(df["key"].values)), height=df["count"].values, width=0.35, align="center", yerr=0.0001)

plt.xticks(range(len(df["key"].values)),df["key_name"].values)

plt.title("test title")
plt.xlabel('test X');
plt.ylabel('test Y');
plt.show()