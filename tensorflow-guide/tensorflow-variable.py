import tensorflow as tf
import numpy as np

data1=tf.Variable(0,name="counter")
print(data1)
one=tf.constant(1)
new_value=tf.add(data1,one)
update=tf.assign(data1,new_value)

init=tf.initialize_all_variables() # 定义有变化量，必须要执行该行代码

sess=tf.Session()
sess.run(init)
for i in range(3):
    sess.run(update)
    print(sess.run(data1))
sess.close()