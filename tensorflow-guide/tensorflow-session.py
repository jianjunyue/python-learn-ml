import tensorflow as tf
import numpy as np

data1=tf.constant([[3,3]])
data2=tf.constant([[2],[2]])
product=tf.matmul(data1,data2)
print(product)
sess=tf.Session()
result=sess.run(product)
print(result)
sess.close()