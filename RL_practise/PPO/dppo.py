#!/usr/bin/python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import threading,queue 

EP_MAX = 500
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
MIN_BATCH_SIZE = 32 
UPDATE_STEPS = 10
EPSION = 0.2
GAME = 'Pendulum-v0'
S_DIM,A_DIM = 3,1
N_WORKER = 2 

class PPO(object):

    def __init__(self,coord,data_queue,rolling_event,update_event):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32,[None,S_DIM],'state')
        self.coord = coord
        self.data_queue = data_queue
        self.rolling_event = rolling_event
        self.update_event = update_event

        #critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs,100,tf.nn.relu)
            self.v = tf.layers.dense(l1,1)
            self.tfdc_r = tf.placeholder(tf.float32,[None,1],'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        #Actor
        pi,pi_params = self._build_anet('pi',trainable=True)
        oldpi,oldpi_params = self._build_anet('oldpi',trainable=False)

        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1),axis=0)
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p,oldp in zip(pi_params,oldpi_params)]


        self.tfa = tf.placeholder(tf.float32,[None,A_DIM],'action')
        self.tfadv = tf.placeholder(tf.float32,[None,1],'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv

                #clipped surrogate objection
                self.aloss = -tf.reduce_mean(tf.minimum(surr,tf.clip_by_value(ratio,1. - EPSION,1 + EPSION)*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        global GLOBAL_UPDATE_COUNTER

        while not self.coord.should_stop():
            if GLOBAL_EP < EP_MAX:
                self.update_event.wait()
                self.sess.run(self.update_oldpi_op)

                data = [self.data_queue.get() for _ in range(self.data_queue.qsize())]

                s = np.vstack([item[0] for item in data])
                a = np.vstack([item[1] for item in data])
                r = np.vstack([item[2] for item in data])
                #data = np.vstack(data)
                #s,a,r = data[:,:S_DIM],data[:, S_DIM:S_DIM + A_DIM], data[:, -1:]
                adv = self.sess.run(self.advantage,{self.tfs:s,self.tfdc_r:r})
                #update actor
                [self.sess.run(self.atrain_op,{self.tfs:s ,self.tfa: a,self.tfadv: adv}) for _ in range(UPDATE_STEPS)]
                #update critic
                [self.sess.run(self.ctrain_op,{self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEPS)]

                self.update_event.clear()
                GLOBAL_UPDATE_COUNTER = 0
                self.rolling_event.set()

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu,trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1,A_DIM,tf.nn.softplus,trainable=trainable)

            norm_dist = tf.distributions.Normal(loc=mu,scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=name)
        return norm_dist, params

    def choose_action(self,s):
        s = s[np.newaxis,:]
        a = self.sess.run(self.sample_op,{self.tfs:s})[0]
        return np.clip(a,-2,2)

    def get_v(self,s):
        if s.ndim < 2:
            s = s[np.newaxis,:]
        return self.sess.run(self.v,{self.tfs:s})[0,0]

class Worker(object):
    def __init__(self,wid,global_ppo,coord,rolling_event,update_event,data_queue,max_ep = EP_MAX,ep_len = EP_LEN,mini_batch_size = MIN_BATCH_SIZE,gamma=GAMMA):
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = global_ppo 
        self.coord = coord
        self.rolling_event = rolling_event
        self.update_event = update_event
        self.data_queue = data_queue
        self.max_ep = max_ep
        self.ep_len = ep_len
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma

    def work(self):
        global GLOBAL_EP, GLOBAL_UPDATE_COUNTER
        while not self.coord.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s,buffer_a,buffer_r = [],[],[]

            for t in range(self.ep_len):
                if not self.rolling_event.is_set():
                    self.rolling_event.wait()
                    buffer_s,buffer_a,buffer_r = [],[],[]
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r+8)/8)
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1
                if t == self.ep_len -1 or GLOBAL_UPDATE_COUNTER >= self.mini_batch_size:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.gamma * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    #bs, ba, br = np.vstack(buffer_s),np.vstack(buffer_a),np.array(discounted_r)[:,np.newaxis]
                    print("Worker[%i] collects %d samples!"%(self.wid,len(buffer_s)))
                    for index in range(len(buffer_s)):
                        self.data_queue.put((buffer_s[index],buffer_a[index],discounted_r[index]))
                    buffer_s,buffer_a,buffer_r = [],[],[]

                    if GLOBAL_UPDATE_COUNTER >= self.mini_batch_size:
                        self.update_event.set()
                        self.rolling_event.clear()

                    if GLOBAL_EP >= self.max_ep:
                        self.coord.request_stop()
                        break
            GLOBAL_EP += 1 
            print("%d-%d  Worker[%i] reward: %.2f"%(GLOBAL_EP,self.max_ep,self.wid,ep_r))

GLOBAL_UPDATE_COUNTER,GLOBAL_EP = 0 , 0
def main():
    env = gym.make('Pendulum-v0').unwrapped
    COORD = tf.train.Coordinator()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(),threading.Event()
    QUEUE = queue.Queue()
    UPDATE_EVENT.clear()
    ROLLING_EVENT.set()
    GLOBAL_PPO = PPO(COORD,data_queue=QUEUE,rolling_event = ROLLING_EVENT,update_event = UPDATE_EVENT)

    workers = [Worker(wid=i,global_ppo = GLOBAL_PPO,coord=COORD,rolling_event = ROLLING_EVENT,update_event = UPDATE_EVENT, data_queue = QUEUE) for i in range(N_WORKER)]
    

    threads = []
    for worker in workers:
        t = threading.Thread(target=worker.work,args=())
        t.start()
        threads.append(t)

    threads.append(threading.Thread(target=GLOBAL_PPO.update,args=()))
    threads[-1].start()
    COORD.join(threads)

    env = gym.make('Pendulum-v0').unwrapped
    while True:
        s = env.reset()
        for t in range(300):
            env.render()
            s = env.step(GLOBAL_PPO.choose_action(s))[0]

if __name__ == "__main__":
    main()
