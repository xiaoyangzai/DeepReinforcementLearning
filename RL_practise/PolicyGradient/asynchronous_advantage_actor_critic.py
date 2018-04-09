#!/usr/bin/python

import gym
import numpy as np
import tensorflow as tf

class A2Cagent:
    def __init__(self, agent_id,state_size,action_size,hidden_dim = 29, discount_factor=0.99,actor_lr = 0.0001,critic_lr = 0.01,T_max = 2000, t_max = 100,load_model = False,model_path = None):
        print("start to create A2Cagent[%s]..."%str(agent_id))
        self.render = False
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.value_size = 1
        self.discount_factor = discount_factor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.load_model = load_model 
        self.model_path = model_path

        self.build_actor()
        self.build_critic()

        #init session
        self.session = tf.InteractiveSession()
        if self.load_model:
            self.saver = tf.train.Saver()
            self.saver.restore(self.session,model_path)
            print("model has restored from file: %s"%self.model_path)
        else:
            self.session.run(tf.global_variables_initializer())

