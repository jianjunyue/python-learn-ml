import numpy as np
import time
import math
# print( np.log1p(0))

distances=range(0,30000,100)
for distance in distances:
   # print(math.log(distance + 1)/10)
   print(np.log1p(distance + 1)/10)

# count=0
# w = open("/Users/jianjun.yue/PycharmGItHub/data/avazu-ctr-prediction/train_small.csv","w")
# with open("/Users/jianjun.yue/PycharmGItHub/data/avazu-ctr-prediction/train") as f:
#     for line in f:
#         count = count + 1
#         if(count<100000):
#             print(count)
#             w.write(line)
#         else:
#             print("time.sleep(1000)")
#             time.sleep(1000)
