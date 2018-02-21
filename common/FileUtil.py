

#读取部分文件
def getPartFile(filePath,count):
    file = open(filePath)
    list=[]
    while 1:
        if count<=0:
            return list
        line = file.readline()
        if line!=None:
            count=count-1
            list.append(line)
        else:
            return list

#字符串写入文件
def writeFile(filePath,data):
    file_object = open(filePath, 'w')
    file_object.write(data)
    file_object.close()