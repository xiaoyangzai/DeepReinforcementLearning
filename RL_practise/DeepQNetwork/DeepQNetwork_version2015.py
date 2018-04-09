#!/usr/bin/python3
import gym
#from gridworld import *
import tensorflow as tf
import numpy as np
import random


class QNetworkTarget():
    def __init__(self,dim_input,dim_output,dim_hidden,gamma = 0.9,learning_rate = 0.01):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.create_Q_network()
        #init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        #design the structure of the network
        self.W1 = self.weight_variable([self.dim_input,self.dim_hidden])
        self.b1 = self.bias_variable([self.dim_hidden])

        self.W2 = self.weight_variable([self.dim_hidden,self.dim_output])
        self.b2 = self.bias_variable([self.dim_output])

        #input layer
        self.state_input = tf.placeholder("float",[None,self.dim_input])

        #hidden layer, active function: ReLU
        h_layer = tf.nn.relu(tf.matmul(self.state_input,self.W1) + self.b1)

        #Q Value layer, active function: Linear
        self.Q_value = tf.matmul(h_layer,self.W2) + self.b2

    def calculate_target_Q_value(self,next_state_batch,reward_batch,done_batch):
        #step 2: calculate y
        target_Q_value_batch = []
        #Q_value_batch = [[1.2,3,0],[0,1,5],[0,13,6],[1,4,2]]
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        #target: y_batch = [10+0.9*3,5 + 0.9*5, -5 + 0.9*13, 0 + 0.9*4] = [12.7,9.5,6.7,3.6]
        for i in range(0,len(next_state_batch)):
            done = done_batch[i] 
            if done:
                target_Q_value_batch.append(reward_batch[i])
            else:
                target_Q_value_batch.append(reward_batch[i] + self.gamma * np.max(Q_value_batch[i]))

        return target_Q_value_batch 

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01,shape = shape)
        return tf.Variable(initial)


class DeepQNetwork():
    def __init__(self,dim_input,dim_output,dim_hidden,gamma = 0.9,learning_rate = 0.01):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.target_dqn = QNetworkTarget(dim_input,dim_output,dim_hidden,gamma,learning_rate) 
        self.create_Q_network()
        self.create_training_method()
        self.time_step = 0

        #init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        self.update_target_Q_network()
        print("DeepQNetwork version-2 2015 has initialized!!")
        
    def update_target_Q_network(self):
        self.cp_ops = [self.target_dqn.W1.assign(self.W1),self.target_dqn.b1.assign(self.b1),self.target_dqn.W2.assign(self.W2),self.target_dqn.b2.assign(self.b2)]
        self.session.run(self.cp_ops)


    def create_Q_network(self):
        #design the structure of the network
        self.W1 = self.weight_variable([self.dim_input,self.dim_hidden])
        self.b1 = self.bias_variable([self.dim_hidden])

        self.W2 = self.weight_variable([self.dim_hidden,self.dim_output])
        self.b2 = self.bias_variable([self.dim_output])

        #input layer
        self.state_input = tf.placeholder("float",[None,self.dim_input])

        #hidden layer, active function: ReLU
        h_layer = tf.nn.relu(tf.matmul(self.state_input,self.W1) + self.b1)

        #Q Value layer, active function: Linear
        self.Q_value = tf.matmul(h_layer,self.W2) + self.b2

    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.dim_output])
        
        #target Q value
        self.y_input = tf.placeholder("float",[None])

        #predict Q value
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)

        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)


    def train_Q_network(self,minibatch):
        self.time_step += 1
        #state_batch = [[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        state_batch = [data[0] for data in minibatch]

        #action_batch = [[1,0,0],[0,1,0],[1,0,0],[0,0,1]]
        action_batch = [data[1] for data in minibatch]

        #reward_batch = [10,5,-5,0]
        reward_batch = [data[2] for data in minibatch]

        #next_state_batch = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
        next_state_batch = [data[3] for data in minibatch]

        done_batch = [data[4] for data in minibatch]

        #step 2: calculate target Q value in the target Q network 
        y_batch = self.target_dqn.calculate_target_Q_value(next_state_batch,reward_batch,done_batch) 
        #print("target Q value: ",y_batch)

        self.optimizer.run(feed_dict={self.y_input: y_batch,self.action_input:action_batch,self.state_input:state_batch})
        #update the target Q network every 100 iteration
        if self.time_step % 30 == 0:
            self.update_target_Q_network()

    def calculate_Q_value(self,state):
        Q_value = self.Q_value.eval(feed_dict={self.state_input:[state]})[0]
        return Q_value

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01,shape = shape)
        return tf.Variable(initial)

def main():
    dqn = DeepQNetwork(2,2,10)
    return

if __name__ == "__main__":
    main()

