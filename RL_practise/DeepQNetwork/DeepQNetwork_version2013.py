#!/usr/bin/python3
import gym
from gridworld import *
import tensorflow as tf
import numpy as np
import random

class DeepQNetwork():
    def __init__(self,dim_input,dim_output,dim_hidden,gamma = 0.9,learning_rate = 0.01,checkpoint_path = None):

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.learning_rate = learning_rate
        self.checkpoint_path = checkpoint_path
        self.gamma = gamma
        self.time_step = 0

        self.create_Q_network()
        self.create_training_method()

        #init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

        #loading network
        if self.checkpoint_path is not None:
            self.saver = tf.train.Saver()
            checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session,checkpoint.model_checkpoint_path)
                print("Successfully loaded: %s"%checkpoint.model_checkpoint_path)
            else:
                print("Could not find old network weights from %s"%self.checkpoint_path)

        global summary_writer
        summary_writer = tf.summary.FileWriter('./logs',graph=self.session.graph)
        print("DeepQNetwork has initilaized.\n\tinput size: %d\n\toutput size: %d\n\tlearning rate: %.2f\n\tgamma: %.3f\n---------------------"%(self.dim_input,self.dim_output,self.learning_rate,self.gamma))


    def create_Q_network(self):
        #design the structure of the network
        W1 = self.weight_variable([self.dim_input,self.dim_hidden])
        b1 = self.bias_variable([self.dim_hidden])

        W2 = self.weight_variable([self.dim_hidden,self.dim_output])
        b2 = self.bias_variable([self.dim_output])

        #input layer
        self.state_input = tf.placeholder("float",[None,self.dim_input])

        #hidden layer, active function: ReLU
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)

        #Q Value layer, active function: Linear
        self.Q_value = tf.matmul(h_layer,W2) + b2

    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.dim_output])
        
        #target Q value
        self.y_input = tf.placeholder("float",[None])

        #predict Q value
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)

        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        tf.summary.scalar("loss",self.cost)

        global merged_summary_op
        merged_summary_op = tf.summary.merge_all()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)


    def train_Q_network(self,minibatch):
        self.time_step += 1
        #step 1: obtain random minibatch from replay memory
        #minibatch = random.sample(self.replay_buffer,self.batch_size)

        #state_batch = [[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        state_batch = [data[0] for data in minibatch]

        #action_batch = [[1,0,0],[0,1,0],[1,0,0],[0,0,1]]
        action_batch = [data[1] for data in minibatch]

        #reward_batch = [10,5,-5,0]
        reward_batch = [data[2] for data in minibatch]

        #next_state_batch = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
        next_state_batch = [data[3] for data in minibatch]

        #step 2: calculate y
        y_batch = []

        #Q_value_batch = [[1.2,3,0],[0,1,5],[0,13,6],[1,4,2]]
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})

        #target: y_batch = [10+0.9*3,5 + 0.9*5, -5 + 0.9*13, 0 + 0.9*4] = [12.7,9.5,6.7,3.6]
        for i in range(0,len(minibatch)):
            done = minibatch[i][-1]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.gamma * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={self.y_input: y_batch,self.action_input:action_batch,self.state_input:state_batch})
        summary_str = self.session.run(merged_summary_op,feed_dict={self.y_input:y_batch,self.action_input:action_batch,self.state_input:state_batch})

        summary_writer.add_summary(summary_str,self.time_step)

        #save network every 1000 iteration
        if self.time_step % 1000 == 0 and self.checkpoint_path:
            self.saver.save(self.session, self.checkpoint_path + '/' + 'network' + '-dqn',global_step = self.time_step)

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

