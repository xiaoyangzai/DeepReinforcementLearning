#!/usr/bin/python
import threading
import tensorflow as tf
import numpy as np
import time
import random

class global_var:
    def __init__(self,var = 10):
        self.var = var
    def get_var(self):
        print("var = %d"%self.var)
        return self.var 

class agent():
    def __init__(self,gv):
        self.sess = tf.InteractiveSession()
        self.weights = tf.Variable([1,2])
        self.b = tf.Variable([4,5])
        self.sess.run(tf.global_variables_initializer())
        self.gv = gv
        print("agent initial finsh!!")
    def function(self):
        op = [self.weights.assign(self.b)]
        self.sess.run(op)
        print("=============")

def main():
    #1. create data
    X = np.random.uniform(-5,5,100) 
    Y_label = []

    with tf.variable_scope('network'):
        x_holder = tf.placeholder(tf.float32,shape=[None,1],name='x')
        y_holder = tf.placeholder(tf.float32,shape=[None,1],name='y')
        hidden_layer_0 = tf.layers.batch_normalization(tf.layers.dense(inputs=x_holder,activation=tf.nn.relu,units=3),name='hidden_0')
        #hidden_layer_1 = tf.layers.batch_normalization(tf.layers.dense(inputs=hidden_layer_0,activation = tf.nn.relu,units=5),name='hidden_1')
        hidden_layer_2 = tf.layers.dense(inputs=hidden_layer_0,activation = tf.nn.relu,units=3,name='hidden_2')
        y_predict = tf.layers.dense(inputs=hidden_layer_2,units=1,name="output")

    #with tf.name_scope('weights'):
    #    w1_initial = tf.truncated_normal([1,5])
    #    #initial = tf.constant([[1.],[1.],[1.],[2.]],shape = [4,1])
    #    #w1_initial = tf.zeros([1,5],tf.float32)
    #    w_1 = tf.Variable(w1_initial,name='target_w')

    #    w2_initial = tf.truncated_normal([5,1])
    #    #w2_initial = tf.zeros([2,1],tf.float32)
    #    w_2 = tf.Variable(w2_initial,name='target_w')

    #with tf.name_scope('output'):
    #    hidden = tf.matmul(x,w_1,name='hidden')
    #    y = tf.matmul(hidden,w_2,name='y')

    #with tf.name_scope('real_output'):
    #    y_ = tf.placeholder(tf.float32,shape=[None,1],name='y')

    #with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(0.0001)

    #with tf.name_scope('gradient'):
        loss = 0.5 * tf.reduce_mean(tf.square(y_holder - y_predict))
        gradient_all = optimizer.compute_gradients(loss)
        grads_vars = [v for (g,v) in gradient_all if g is not None]
        gradient = optimizer.compute_gradients(loss,grads_vars)
        grads_holder = [(tf.placeholder(tf.float32,shape = g.get_shape()),v) for (g,v) in gradient]
        train_op = optimizer.apply_gradients(grads_holder)

    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='network')
    current_step = 0
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    while current_step < 20000:
        y_i = []
        for x in X:
            y_i.append(2*x*x + 2)
        x_i = X.reshape(len(X),1)
        y_i = np.array(y_i)
        y_i = y_i.reshape(len(y_i),1)

        loss1 = sess.run(loss,feed_dict={x_holder:x_i,y_holder:y_i})
        grads = sess.run(gradient_all,feed_dict={x_holder:x_i,y_holder:y_i})
        print("=================================")
        for index in range(len(grads)):
            print("layer %d [weights]:"%index)
            print grads[index][0]
            print("layer %d [grads]:"%index)
            print grads[index][1]
            break
        #time.sleep(1)
        print "[%d]loss :%f"%(current_step,loss1)
        print("=================================")
        grads_dict = {}
        for i in range(len(grads_holder)):
            k = grads_holder[i][0]
            grads_dict[k] =  grads[i][0]
        sess.run(train_op,feed_dict = grads_dict)
        current_step += 1
        #print("network weights:",sess.run(weights))

    return

if __name__ == "__main__":
    main()


