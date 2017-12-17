import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
from matplotlib.font_manager import FontProperties
from pylab import *
myfont=matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttf')  #微软雅黑字体
mpl.rcParams['axes.unicode_minus'] = False
def getChineseFont():
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
week1=[8.9,9.3,8]
week2=[8.4,8.8,7.8]
week3=[9.6,9.7,9.1]
week4=[9.5,9.5,8.1]
week5=[9.1,9.1,7.9]
week6=[9.0,9.2,8.7]
week7=[9.33,9.26,9.06]
week8=[8.3,7.8,7.17]

time=["2017.06.21",""]

df = pd.DataFrame(columns=['饿了么', '美团外卖', '百度外卖'])
df['饿了么']=[week1[0],week2[0],week3[0],week4[0],week5[0],week6[0],week7[0],week8[0]]
df['美团外卖']=[week1[1],week2[1],week3[1],week4[1],week5[1],week6[1],week7[1],week8[1]]
df['百度外卖']=[week1[2],week2[2],week3[2],week4[2],week5[2],week6[2],week7[2],week8[2]]
print(df)
plt.figure(figsize=(8,6))
zhfont1 =FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
plt.legend(prop=zhfont1)

# plt.plot(range(len(df["key"].values)),  df["count"].values)
plt.plot(range(8),  df['饿了么'].values,marker='o',label='ele',color = 'b')
plt.plot(range(8),  df['美团外卖'].values,marker='o',label = "meitaun",color = 'y' )
plt.plot(range(8),  df['百度外卖'].values,marker='o',label = "baidu",color = 'r' )
# plt.plot(range(8),  df['百度外卖'].values,s=30,c='blue',marker='x',alpha=0.5,label='百度外卖')
# plt.xticks(range(len(df["key"].values)),df["key_name"].values) #给X轴赋值名称
# plt.xticks(range(len(df["key"].values)),df["key_name"].values)
plt.legend(loc='upper right')
plt.title(u"关键字搜索DCG评测",fontproperties=getChineseFont())
plt.xlabel(u"评测次数",fontproperties=getChineseFont());
plt.ylabel(u"评测分数",fontproperties=getChineseFont());
plt.show()