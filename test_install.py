import tensorflow as tf

tf.compat.v1.disable_eager_execution()
hello = tf.constant('Hello, TensorFlow!')
sess = tf.compat.v1.Session()
print(sess.run(hello))

#在路径C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin 下，把cudart64_102.dll文件改名为cudart64_101.dll才能够加载成功，版本不对应
