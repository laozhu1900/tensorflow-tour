# coding: utf-8
import numpy as np
from main import FNN
import tensorflow as tf
import matplotlib.pyplot as plt

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 1, 1, 0]

X = np.array(inputs).reshape((4, 1, 2)).astype('int16')
Y = np.array(outputs).reshape((4, 1, 1)).astype('int16')

ff = FNN(learning_rate=1e-3, drop_keep=1.0, Layers=1, N_hidden=[2], D_input=2, D_label=1, Task_type='regression',
         L2_lambda=1e-2)
# 下面是实际输出内容


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 将所有的summary合成一个ops
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('log' + '/train', sess.graph)

W0 = sess.run(ff.W[0])
W1 = sess.run(ff.W[1])

print('W_0:\n%s' % sess.run(ff.W[0]))
print('W_1:\n%s' % sess.run(ff.W[1]))

# 训练前的输出
pY = sess.run(ff.output, feed_dict={ff.inputs: X.reshape((4, 2)), ff.drop_keep_rate: 1.0})
print(pY)
# 画图（4个数据循序用红、绿、蓝、黑表示）
plt.scatter([0, 1, 2, 3], pY, color=['red', 'green', 'blue', 'black'], s=25, alpha=0.4, marker='o')
