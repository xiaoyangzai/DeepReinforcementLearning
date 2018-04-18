#!/usr/bin/python
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class A2Cagent:
    def __init__(self,env,hidden_dim = 29,discount_factor=0.99,actor_lr = 0.0001,critic_lr=0.01,hidden_layers = 1,load_model = False,model_path = None,max_episodes = 1000):
        self.render = True

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        print("state size:",self.state_size)
        print("action size:",self.action_size)
        self.hidden_dim = hidden_dim 
        self.value_size = 1
        self.load_model = load_model 
        self.model_path = model_path
        self.max_episodes = max_episodes
        self.env = env

        self.discount_factor = discount_factor 
        self.actor_lr = actor_lr 
        self.critic_lr = critic_lr 

        self.build_actor()
        self.build_critic()
        self.saver = tf.train.Saver()

        #init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        if self.load_model and self.model_path is not None:
            try:
                self.saver.restore(self.session,model_path)
                print("model has restored from file: %s"%self.model_path)
            except:
                print("load model from %s failed!"%self.model_path)
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

    def learning(self):
        print("A2Cagent begin to learn....")
        max_score_count = 0
        for e in range(1,self.max_episodes):
            done = False
            score = 0
            state = self.env.reset()
            state = np.reshape(state,[1,self.state_size])
            while not done:
                if self.render:
                    self.env.render()

                action = self.choose_action(state)
                next_state,reward,done,info = self.env.step(action)
                next_state = np.reshape(next_state,[1,self.state_size])
                #reward = reward if not done or score == 499 else -100

                self.train_model(state,action,reward,next_state,done)
                score += reward
                state = next_state

                if done:
                    if score == 500:
                        max_score_count += 1
                    else:
                        max_score_count = 0
            print("Episode[%d] Score: %d"%(e,score))
            #save model each 100 episodes
            if e % 20 == 0 and self.model_path is not None:
                self.saver.save(self.session,self.model_path)
            if max_score_count >= 25:
                break



def draw_reward(score_list):
    fig, ax = plt.subplots()
    x = [i for i in range(len(score_list))]
    ax.plot(x,score_list)
    ax.set_xlabel("Episode Index",fontsize=20)
    ax.set_ylabel("Reward",fontsize=20)
    plt.yticks([i for i in range(0,1000,50)])
    plt.title("Advantage Actor-Critic(A2C)")
    plt.show()


def main():
    env = gym.make('CartPole-v1')
    agent = A2Cagent(env,load_model = True,model_path = "./Model/a2c_model.ckpt")
    agent.learning()
    return

if __name__ == "__main__":
    main()
