#!/usr/bin/python3
#--config: utf8--
import tensorflow as tf
import numpy as np

def main():
    input_num = 6
    output_num = 6
    x_data = np.linspace(-1,1,60).reshape((-1,input_num))
    noise = np.random.normal(0,0.05,x_data.shape)
    y_data = np.square(x_data) + 0.5 + noise
    
    xs = tf.placeholder(tf.float32,[None,input_num])
    ys = tf.placeholder(tf.float32,[None,output_num])

    #第一层神经元个数:3
    #输入矩阵 m*6,即每一行为一个样本
    #输出矩阵 6*3,即每一列表示一个神经元的权值向量
    neuro_layer_1 = 3
    w1 = tf.Variable(tf.random_normal([input_num,neuro_layer_1]))

    b1 = tf.Variable(tf.zeros([1,neuro_layer_1]) + 0.1)
    l1 = tf.nn.relu(tf.matmul(xs,w1) + b1)

    neuro_layer_2 = 6
    w2 = tf.Variable(tf.random_normal([neuro_layer_1,neuro_layer_2]))
    b2 = tf.Variable(tf.zeros([1,neuro_layer_2]) + 0.1)
    l2 = tf.matmul(l1,w2) + b2

    #目标函数公式
    #reduction_indices = 0 表示将行数据累加在一起
    #reduction_indices = 1 表示将列数据累加在一起
    loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys - l2)),reduction_indices=[1]))

    #梯度下降法进行优化
    train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(100000):
        sess.run(train, feed_dict={xs: x_data, ys:y_data})
        if i % 1000 == 0:
            print(sess.run(loss,feed_dict={xs: x_data, ys: y_data}))

    return

if __name__ == "__main__":
    main()
