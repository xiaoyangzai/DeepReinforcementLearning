#!/usr/bin/python3
#--coding: utf8--

from random import random
from gym import Env
import gym
from gridworld import *

class sarsaLambdaAgent():
    def __init__(self,env:Env):
        self.env = env
        self.Q = {}
        self.E = {}
        self.state = None
        self._init_agent()
        return
    
    def _init_agent(self):
        self.state = self.env.reset()
        s_name = self._name_state(self.state)
        self._assert_state_in_QE(s_name,randomized = False)

    def _curPolicy(self,s,num_episode,use_epsilon):
        epsilon = 1.00 / (num_episode + 1)
        Q_s = self.Q[s]
        rand_value = random()
        if use_epsilon and rand_value < epsilon:
            return self.env.action_space.sample()
        else:
            return int(max(Q_s,key=Q_s.get))

    def performPolicy(self,s,num_episode,use_epsilon=True):
        return self._curPolicy(s,num_episode,use_epsilon)

    def act(self,a):
        return self.env.step(a)

    def _is_state_in_Q(self,s):
        return self.Q.get(s) is not None

    def _init_state_value(self,s_name,randomized = True):
        if not self._is_state_in_Q(s_name):
            self.Q[s_name],self.E[s_name] = {},{}
            for action in range(self.env.action_space.n):
                default_v = random()/10 if randomized is True else 0.0
                self.Q[s_name][action] = default_v
                self.E[s_name][action] = 0.0

    def _assert_state_in_QE(self,s,randomized = True):
        if not self._is_state_in_Q(s):
            self._init_state_value(s,randomized)

    def _name_state(self,state):
        return str(state)

    def _get_(self,QorE,s,a):
        self._assert_state_in_QE(s,randomized=True)
        return QorE[s][a]

    def _set_(self,QorE,s,a,value):
        self._assert_state_in_QE(s,randomized = True)
        QorE[s][a] = value
    
    def _resetEValue(self):
        for value_dic in self.E.values():
            for action in range(self.env.action_space.n):
                value_dic[action] = 0.00

    def learning(self,lambda_,gamma,alpha,max_episode_num):
        total_time = 0
        time_in_episode = 0
        num_episode = 1
        while num_episode <= max_episode_num:
            self._resetEValue()
            s0 = self._name_state(self.env.reset())
            a0 = self.performPolicy(s0,num_episode)
            self.env.render()

            time_in_episode = 0
            is_done = False
            while not is_done:
                s1,r1,is_done,info = self.act(a0)
                self.env.render()
                s1 = self._name_state(s1)
                self._assert_state_in_QE(s1,randomized = True)
                a1 = self.performPolicy(s1,num_episode,use_epsilon = False)

                delta = r1 + gamma * self._get_(self.Q,s1,a1) - self._get_(self.Q,s0,a0)

                self._set_(self.E,s0,a0,self._get_(self.E,s0,a0)+1)

                state_action_list = list(zip(self.E.keys(),self.E.values()))
                for s, a_es in state_action_list:
                    for a in range(self.env.action_space.n):
                        e_value = a_es[a]
                        old_q = self._get_(self.Q,s,a)
                        new_q = old_q + alpha * delta * e_value
                        self._set_(self.E,s,a,alpha * delta * e_value
)
                if num_episode == max_episode_num:
                    print("t:{0:>2}: s:{2}, a:{2:10}, s1:{3}".format(time_in_episode,s0,a0,s1))
                s0 = s1
                a0 = a1
                time_in_episode += 1

            print("Episode {0} takes {1} steps.".format(num_episode, time_in_episode))
            total_time += time_in_episode
            num_episode += 1
        return





    def showOptimalPolicy(self):
        print("The Optimal policy: ")
        print(self.Q)



def main():
    env = GridWorldEnv(n_width = 10,n_height=5,u_size=60,default_reward = -1,windy=False)
    env.start = (0,0)
    env.ends = [(9,0)]
    env.rewards = [(9,0,10),(1,0,-10),(2,0,-10),(3,0,-10),(4,0,-10),(5,0,-10),(6,0,-10),(7,0,-10),(8,0,-10)]
    env.refresh_setting()
    agent = sarsaLambdaAgent(env)
    print("learning.....")
    agent.learning(lambda_= 0.01,gamma=0.9,alpha =0.2,max_episode_num = 500)
    agent.showOptimalPolicy()
    return

if __name__ == "__main__":
    main()
