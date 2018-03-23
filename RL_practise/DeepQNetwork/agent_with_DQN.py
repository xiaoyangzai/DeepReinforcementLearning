#!/usr/bin/python3

import tensorflow as tf
from gym import spaces,Env 
import numpy as np
import random
from DeepQNetwork_version2015 import DeepQNetwork 
import time
from collections import deque
import time

class Agent():
    def __init__(self,env: Env=None,max_episodes = 10000,hidden_dim = 20,replay_size = 5000,learning_rate = 0.01,gamma = 0.9,batch_size = 64,min_epsilon = 0.01,max_step_each_episode = 50,epsilon_decay = True):
        self.env = env
        self.max_step_each_episode = max_step_each_episode
        if isinstance(env.observation_space, spaces.Discrete):
            self.state_dim = env.observation_space.n
        elif isinstance(env.observation_space, spaces.Box):
            self.state_dim = env.observation_space.shape[0]

        self.action_dim = env.action_space.n
        self.max_episodes = max_episodes

        self.batch_size = batch_size
        self.min_epsilon = min_epsilon
        self.cur_epsilon = 1 
        self.replay_buffer = deque()
        self.replay_size = replay_size
        self.epsilon_decay = epsilon_decay
        self.dqn = DeepQNetwork(dim_input = self.state_dim,dim_output = self.action_dim,dim_hidden = hidden_dim,learning_rate = learning_rate,gamma = gamma) 

    @property
    def epsilon(self):
        return self.cur_epsilon

    def performPolicy(self,state,egreedy_flag = True):
        Q_value = self.dqn.calculate_Q_value(state) 
        if egreedy_flag and random.random() <= self.cur_epsilon:
            action = random.randint(0,self.action_dim - 1)
        else:
            action = np.argmax(Q_value)

        if egreedy_flag == False:
            return action

        if self.epsilon_decay:
            if self.cur_epsilon > self.min_epsilon:
                self.cur_epsilon -= (1 - self.min_epsilon) /20000
            else:
                self.cur_epsilon = self.min_epsilon

        return action
    
    def _learn_from_memory(self):
        minibatch = random.sample(self.replay_buffer,self.batch_size)
        self.dqn.train_Q_network(minibatch)

    def memory_experience(self,state,action,reward,next_state,done):
        #convert the action to one-hot code
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1

        #memory experience
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))

        if len(self.replay_buffer) > self.replay_size:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > self.batch_size:
            self._learn_from_memory()
        return len(self.replay_buffer)

    
    def learning(self):
        for episode in range(1,self.max_episodes,1):
            episode_reward = 0.0
            state = self.env.reset()
            self.env.render()
            #for step in range(self.max_step_each_episode):
            done = False
            while not done:
                action = self.performPolicy(state) 
                next_state,reward,done,info = self.env.step(action)
                self.memory_experience(state,action,reward,next_state,done)
                episode_reward += reward
                state = next_state
                self.env.render()
            if episode % 5 == 0:
                print("Episode[%d] start to training....Reward[%.2f] Epsilon[%.2f]"%(episode,episode_reward,self.cur_epsilon))

            if episode % 50 == 0:
                print("start to test....")
                total_reward = 0.0
                successful_num = 0
                for i in range(10):
                    state = self.env.reset()
                    for j in range(self.max_step_each_episode):
                        self.env.render()
                        action = self.performPolicy(state,egreedy_flag = False) 
                        next_state,reward,done,info = self.env.step(action)
                        total_reward += reward
                        state = next_state
                        if done:
                            successful_num += 1
                            break
                
                state = self.env.reset()
                print('episode: %d Evaluation Avg Reward:%.2f\t Epsilon: %.2f Done: %d/10'%(episode,total_reward/10,self.cur_epsilon,successful_num))
        print("DQN has been trained successfully!")
        done = False
        for i in range(50):
            total_reward = 0.0
            successful_num = 0
            state = self.env.reset()
            for j in range(self.max_step_each_episode):
                self.env.render()
                action = self.performPolicy(state,egreedy_flag = False) 
                next_state,reward,done,info = self.env.step(action)
                total_reward += reward
                state = next_state
                if done:
                    successful_num += 1
                    break
            print("test[%d]: Reward[%d]  Done[%d/%d]"%(i,total_reward,successful_num,self.max_step_each_episode))
        return
