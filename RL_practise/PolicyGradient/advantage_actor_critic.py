#!/usr/bin/python
import gym
import numpy as np
import tensorflow as tf

class A2Cagent:
    def __init__(self,state_size,action_size,hidden_dim = 25,discount_factor=0.99,actor_lr = 0.001,critic_lr=0.005,hidden_layers = 1):
        self.render = True
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim 
        self.value_size = 1
        self.load_model = False

        self.discount_factor = discount_factor 
        self.actor_lr = actor_lr 
        self.critic_lr = critic_lr 

        self.build_actor()
        self.build_critic()

        #init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

        print "Actor-Critic network has initilized!!"

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01,shape = shape)
        return tf.Variable(initial)

    def build_actor(self):
        self.actor_W1 = self.weight_variable([self.state_size,self.hidden_dim])    
        self.actor_b1 = self.bias_variable([self.hidden_dim])

        self.actor_W2 = self.weight_variable([self.hidden_dim,self.action_size])    
        self.actor_b2 = self.bias_variable([self.action_size])

        self.actor_ylabel= tf.placeholder(tf.float32,shape=[1,self.action_size])

        #hidden layer
        self.actor_state_input = tf.placeholder(tf.float32,shape=[None,self.state_size])
        self.actor_action_input = tf.placeholder(tf.int32)
        actor_h_layer = tf.nn.relu(tf.matmul(self.actor_state_input,self.actor_W1) + self.actor_b1)
        #actor output: the probilities of all actions
        self.actor_yperdict = tf.nn.softmax(tf.matmul(actor_h_layer,self.actor_W2) + self.actor_b2)

        self.actor_loss = tf.reduce_mean(-tf.reduce_sum(self.actor_ylabel*(tf.one_hot(self.actor_action_input,self.action_size) * tf.log(self.actor_yperdict + 1e-13)),reduction_indices=[1]))
        self.actor_optimizer = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss)

    def build_critic(self):
        self.critic_W1 = self.weight_variable([self.state_size,self.hidden_dim]) 
        self.critic_b1 = self.bias_variable([self.hidden_dim])

        self.critic_W2 = self.weight_variable([self.hidden_dim,self.value_size])
        self.critic_b2 = self.bias_variable([self.value_size])

        self.critic_state_input = tf.placeholder(tf.float32,shape = [None,self.state_size])

        self.critic_ylabel = tf.placeholder(tf.float32,shape = [None,self.value_size])

        #hidden layer 1
        critic_h_layer = tf.nn.relu(tf.matmul(self.critic_state_input,self.critic_W1) + self.critic_b1)

        #output layer
        self.critic_yperdict = tf.matmul(critic_h_layer,self.critic_W2) + self.critic_b2 

        self.advantages = self.critic_ylabel - self.critic_yperdict
        
        #loss function
        self.critic_loss = tf.reduce_mean(tf.square(self.advantages))
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)

    def train_model(self,state,action,reward,next_state,done):
        #state value
        critic_ypredict = self.session.run(self.critic_yperdict,feed_dict={self.critic_state_input: state})
        #next state value
        next_critic_ypredict = self.session.run(self.critic_yperdict,feed_dict={self.critic_state_input: next_state})
        advantages = np.zeros((1,self.action_size))
        critic_ylabel = np.zeros((1,self.value_size))

        if done:
            advantages[0][action] = reward - critic_ypredict 
            critic_ylabel[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * next_critic_ypredict - critic_ypredict 
            critic_ylabel[0][0] = reward + self.discount_factor * next_critic_ypredict

        self.actor_optimizer.run(feed_dict = {self.actor_ylabel: advantages, self.actor_state_input: state,self.actor_action_input:action})
        self.critic_optimizer.run(feed_dict = {self.critic_state_input: state,self.critic_ylabel: critic_ylabel})

    def choose_action(self,state):
        policy = self.session.run(self.actor_yperdict,feed_dict={self.actor_state_input: state}).flatten() 
        return np.random.choice(self.action_size,1,p=policy)[0]


def main():
    env = gym.make('CartPole-v1')
    #env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2Cagent(state_size,action_size)

    scores,episodes = [],[]
    
    max_episodes = 1000
    for e in range(1,max_episodes):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state,[1,state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.choose_action(state)
            next_state,reward,done,info = env.step(action)
            next_state = np.reshape(next_state,[1,state_size])
            #reward = reward if not done or score == 499 else -100

            agent.train_model(state,action,reward,next_state,done)
            score += reward
            state = next_state

            if done:
                #score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                print("Episode: ",e," Score: ",score)

    return

if __name__ == "__main__":
    main()
