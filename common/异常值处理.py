import numpy as np
import pandas as pd

def upper(data):
    data_temp=data.copy()
    data_temp.sort()
    count=len(data_temp)
    count_nine_five=int(count*0.95)
    count_five=int(count*0.05)
    value_nine_five =   data_temp[count_nine_five]
    value_five =   data_temp[count_five]
    base_value= (value_nine_five-value_five)*0.9/count  #平均每增加一个排名，增加的值
    uppercount=   count- count_nine_five -1
    dict_temp={}
    for uc in  range(uppercount):
        print(uc)
        uc=uc+1
        dict_temp[data_temp[value_nine_five+uc]] =value_nine_five+ base_value*uc

    for i in len(data):
        print("-----------")
        print(i)
        value=data[i]
        print(value)
        if dict_temp.get(value)!=None:
            data[i]=dict_temp.get(value)

    return   data




a=[1.1,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,2.2,3.2,1000.2,1021.1]
# a.sort()
# print(a)
# print(a[3])

data=upper(a)
print(data)

