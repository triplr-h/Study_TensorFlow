import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100))  
# 随机输入，np.random.rand()返回[0,1)的均匀分布的样本值，括号里面生成多少数组，一个数字是一维数组，大小代表个数，两个数字代表二维数组，这里面是两行100列
y_data = np.dot([0.100, 0.200], x_data) + 0.300 
#dot函数返回两个数组的点积，一行两列乘两行100列，结果是一行100列，再加上0.300

# 构造一个线性模型
#
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random.uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b


# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)


# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(W), sess.run(b))

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]
