import zipfile
import numpy as np
import pandas as pd

def getPartFile(filePath,count):
    file = open(filePath)
    str=""
    while 1:
        if count<=0:
            return str
        line = file.readline()
        if line!=None:
            count=count-1
            str += line
        else:
            return str

data_root = "/Users/jianjun.yue/PycharmGItHub/data/国内比赛/腾讯移动App广告转化率预估/test.csv"
part_data=getPartFile(data_root,100000)
print(len(part_data))
# print(part_data.count())
# print(part_data)
file_object = open('/Users/jianjun.yue/PycharmGItHub/data/国内比赛/腾讯移动App广告转化率预估/test_part.csv', 'w')
file_object.write(part_data)
file_object.close( )