#!/usr/bin/python3
#--coding: utf8--

from random import random
from gym import Env
import gym
from gridworld import *

class sarsaAgent():
    def __init__(self,env: Env):
        self.env = env
        self.Q = {}
        self._initAgent()
        self.state = None

    def _get_state_name(self,state):
        return str(state)

    def _is_state_in_Q(self,s):
        return self.Q.get(s) is not None

    def _init_state_value(self,s_name,randomized = True):
        if not self._is_state_in_Q(s_name):
            self.Q[s_name] = {}
            for action in range(self.env.action_space.n):
                default_v = random() / 10 if randomized is True else 0.0
                self.Q[s_name][action] = default_v

    def _assert_state_in_Q(self,s,randomized=True):
        if not self._is_state_in_Q(s):
            self._init_state_value(s,randomized)

    def _get_Q(self,s,a):
        self._assert_state_in_Q(s,randomized=True)
        return self.Q[s][a]

    def _set_Q(self,s,a,value):
        self._assert_state_in_Q(s,randomized=True)
        self.Q[s][a] = value

    def _initAgent(self):
        self.state = self.env.reset()
        s_name = self._get_state_name(self.state)
        self._assert_state_in_Q(s_name,randomized = False)

    def _curPolicy(self,s,episode_num,use_epsilon):
        #epsilon = 1.0 / (episode_num + 1)
        if episode_num <= 150: 
            epsilon = 0.2
        else:
            use_epsilon = False
        Q_s = self.Q[s]
        str_act = "unknown"
        rand_value = random()
        action = None
        if use_epsilon and rand_value < epsilon:
            action = self.env.action_space.sample()
        else:
            str_act = max(Q_s,key=Q_s.get)
            action = int(str_act)
        return action
    
    def performPolicy(self,s,episode_num,use_epsilon=True):
        return self._curPolicy(s,episode_num,use_epsilon)

    def act(self,a):
        return self.env.step(a)

    #sarsa learning
    #Q(s,a) = Q(s,a) + %alpha(R + %gamma*Q(s^',a^') - Q(s,a))
    def learning(self,gamma,alpha,max_episode_num):
        total_time, time_in_episode, num_episode = 0,0,0
        while num_episode < max_episode_num:
            self.state = self.env.reset()
            s0 = self._get_state_name(self.state)
            self.env.render()
            a0 = self.performPolicy(s0,num_episode)
            time_in_episode = 0
            is_done = False
            while not is_done:
                s1, r1, is_done, info = self.act(a0)
                self.env.render()
                s1 = self._get_state_name(s1)
                self._assert_state_in_Q(s1,randomized = False)
                a1 = self.performPolicy(s1,num_episode)
                old_q = self._get_Q(s0,a0)
                new_q = self._get_Q(s1,a1)
                target = r1 + gamma*new_q
                new_q = old_q + alpha * (target - old_q)
                self._set_Q(s0,a0,new_q)

                if num_episode == max_episode_num:
                    print("t:{0:>2} : s:{1},a:{2:2}, s1:{3}".format(time_in_episode, s0, a0, s1))
                s0 ,a0 = s1, a1
                time_in_episode += 1

            print("Episode {0} takes {1} steps.".format(num_episode,time_in_episode))
            total_time += time_in_episode
            num_episode += 1

        return

    def showOptimalPolicy(self):
        print("The Optimal policy: ")
        print(self.Q)



def main():
    env = GridWorldEnv(n_width = 12,n_height=4,u_size=60,default_reward = -1,windy=False)
    env.start = (0,0)
    env.ends = [(11,0)]
    env.rewards = [(11,0,100),(1,0,-100),(2,0,-100),(3,0,-100),(4,0,-100),(5,0,-100),(6,0,-100),(7,0,-100),(8,0,-100),(9,0,-100),(10,0,-100)]
    env.refresh_setting()
    agent = sarsaAgent(env)
    print("learning.....")
    agent.learning(gamma=0.9,alpha =0.1,max_episode_num = 200)
    agent.showOptimalPolicy()
    return

if __name__ == "__main__":
    main()
