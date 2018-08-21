

#读取部分文件
def getPartFile(filePath,count):
    file = open(filePath)
    list=""
    while 1:
        if count<=0:
            return list
        line = file.readline()
        if line!=None:
            count=count-1
            list+=line
        else:
            return list

#字符串写入文件
def writeFile(filePath,data):
    file_object = open(filePath, 'w')
    file_object.write(data)
    file_object.close()



filename_test = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/第三届阿里云安全算法挑战赛/test.csv'
# filename_test = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/第三届阿里云安全算法挑战赛/test_temp.csv'

data=getPartFile(filename_test,3000)
filename_test = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/第三届阿里云安全算法挑战赛/test_temp.csv'
writeFile(filename_test,data)

filename = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/第三届阿里云安全算法挑战赛/train.csv'
data=getPartFile(filename,300000)
filename = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/第三届阿里云安全算法挑战赛/train_temp.csv'
writeFile(filename,data)