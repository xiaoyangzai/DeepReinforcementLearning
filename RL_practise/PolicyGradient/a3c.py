#!/usr/bin/python
import gym
import numpy as np
import tensorflow as tf

from semaphore import semaphore
import threading

class global_A3C_parameters:
    def __init__(self,T=0,sess = None):
        self.T = 0
        self.initialize_weights_flag = False
        self.actor_weights_list = []  
        self.critic_weights_list = [] 

        self.actor_weights_gradients = tf.Variable(0.0) 
        self.critic_weights_gradients = tf.Variable(0.0) 
        self.actor_weights_gradients_list = []  
        self.critic_weights_gradients_list = [] 
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        #create semaphore
        self.sems = semaphore(n_sems = 5,n_values = [1,1,0,1,0])

    def accumulate_T(self):
        self.sems.P(index = 0)
        self.T += 1
        self.sems.V(index = 0)

    def get_T(self):
        self.sems.P(index = 0)
        T = self.T 
        self.sems.V(index = 0)
        return T
    def initialize_weights(self,actor_weights,critic_weights):
        self.initialize_weights_flag= True
        self.actor_weights_list = []
        self.critic_weights_list = []
        for index in range(len(actor_weights)):
            self.actor_weights_list.append(actor_weights[index])
            self.sess.run(self.actor_weights_list[index].initializer)
        for index in range(len(critic_weights)):
            self.critic_weights_list.append(critic_weights[index])
            self.sess.run(self.critic_weights_list[index].initializer)
        print("global actor and critic weiths has initialized!!")


    def set_A3C_weights(self, actor_weights,critic_weights):
        self.sems.P(index = 1)
        #self.actor_weights_list = actor_weights 
        #self.critic_weights_list = critic_weights 
        self.initialize_weights(actor_weights,critic_weights)
        #if self.initialize_weights_flag == False:
        #    self.initialize_weights(actor_weights,critic_weights)
        #else:
        #    #self.critic_weights_list = critic_weights
        #    #self.actor_weights_list = actor_weights
        #    for index in range(len(critic_weights)):
        #        cp_ops = [tf.assign(self.critic_weights_list[index],critic_weights[index])]
        #        self.sess.run(cp_ops)

        #    for index in range(len(actor_weights)):
        #        cp_ops = [tf.assign(self.actor_weights_list[index],actor_weights[index])]
        #        self.sess.run(cp_ops)
        #print("afger set A3C parameters in Critic: ",self.sess.run(self.critic_weights_list))
        #print("afger set A3C parameters in Actor: ",self.sess.run(self.actor_weights_list))
        self.sems.V(index = 1)

    def set_A3C_weights_gradients(self,actor_weights_gradients,critic_weights_gradients):
        self.sems.P(index = 3)
        self.actor_weights_gradients_list = actor_weights_gradients  
        self.critic_weights_gradients_list = critic_weights_gradients 
        self.sems.V(index = 4)

    def get_A3C_weights(self):
        self.sems.P(index = 1)
        actor_weights,critic_weights = self.actor_weights_list, self.critic_weights_list
        self.sems.V(index = 1)
        return actor_weights,critic_weights

    def get_A3C_weights_gradients(self):
        self.sems.P(index = 4)
        actor_weights_gradients,critic_weights_gradients = self.actor_weights_gradients_list, self.critic_weights_gradients_list
        self.sems.V(index = 3)
        return actor_weights_gradients, critic_weights_gradients


class A2Cagent:
    def __init__(self,agent_id,sess,hidden_dim = 25,discount_factor=0.99,actor_lr = 0.0005,critic_lr = 0.01,model_path = None,max_steps_each_episode = 200,max_episodes = 10000,is_master_process = False,global_parameters = None):
        self.render = False
        if agent_id == 1 and is_master_process == False:
            self.render = True 
        self.env = gym.make('CartPole-v1')
        self.is_master_process = is_master_process
        self.id = str(agent_id)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.hidden_dim = hidden_dim 
        self.value_size = 1
        self.model_path = model_path
        self.session =  sess 
        self.max_episodes = max_episodes
        self.max_steps_each_episode = max_steps_each_episode

        self.discount_factor = discount_factor 
        self.actor_lr = actor_lr 
        self.critic_lr = critic_lr 


        self.build_actor()
        self.build_critic()
        self.saver = tf.train.Saver()
        self.global_parameters =  global_parameters 

        self.session.run(tf.global_variables_initializer())
        #init session
        if self.is_master_process and self.model_path is not None:
            try:
                self.saver.restore(self.session,model_path)
                print("Master agent model has restored from file: %s"%self.model_path)
            except:
                print("Master load model from file[%s] failed"%self.model_path )

        if self.is_master_process:
            print("Master agent is initializing the weights....")
            self.global_parameters.set_A3C_weights(self.actor_weights,self.critic_weights)
            print("Master agent has initialized the weights!!")
        if is_master_process:
            print "Mater Actor-Critic network[%s] has initilized!!"%self.id
        else:
            print "Slave Actor-Critic network[%s] has initilized!!"%self.id


    def build_actor(self):
        
        scope = 'actor_network_%s'%self.id
        with tf.variable_scope(scope):
            self.actor_state_input = tf.placeholder(tf.float32,shape=[None,self.state_size])
            self.actor_action_input = tf.placeholder(tf.int32)
            self.actor_ylabel= tf.placeholder(tf.float32,shape=[1,self.action_size])

            hidden_layer_0 = tf.layers.dense(inputs=self.actor_state_input,activation=tf.nn.relu,units=self.hidden_dim,name='actor_hidden_%s'%self.id)

            self.actor_ypredict = tf.layers.dense(inputs=hidden_layer_0,activation=tf.nn.softmax,units=self.action_size,name='actor_ypredict_%s'%self.id)

            self.actor_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)

            self.actor_loss = tf.reduce_mean(-tf.reduce_sum(self.actor_ylabel*(tf.one_hot(self.actor_action_input,self.action_size) * tf.log(self.actor_ypredict + 1e-13)),reduction_indices=[1]))
            self.actor_optimizer = tf.train.GradientDescentOptimizer(self.actor_lr)

            # compute the gradients of actor network
            gradient_all = self.actor_optimizer.compute_gradients(self.actor_loss)
            self.actor_grads_vars = [v for (g,v) in gradient_all if g is not None]
            self.actor_gradients = self.actor_optimizer.compute_gradients(self.actor_loss,self.actor_grads_vars)

            self.actor_grads_holder = [(tf.placeholder(tf.float32,shape = g.get_shape()),v) for (g,v) in self.actor_gradients]

            #update parameters by using gradients
            self.actor_train_op = self.actor_optimizer.apply_gradients(self.actor_grads_holder)

    def build_critic(self):
        scope = 'critic_network_%s'%self.id
        with tf.variable_scope(scope):
            self.critic_state_input = tf.placeholder(tf.float32,shape = [None,self.state_size])

            self.critic_ylabel = tf.placeholder(tf.float32,shape = [None,self.value_size])

            #hidden layer 1
            hidden_layer_0 = tf.layers.dense(inputs=self.critic_state_input,activation=tf.nn.relu,units=self.hidden_dim,name='critic_hidden_%s'%self.id)
            self.critic_ypredict = tf.layers.dense(inputs=hidden_layer_0,units=self.value_size,name='critic_ypredict_%s'%self.id)

            self.advantages = self.critic_ylabel - self.critic_ypredict
        
            #loss function
            self.critic_loss = tf.reduce_mean(tf.square(self.advantages))
            self.critic_optimizer = tf.train.GradientDescentOptimizer(self.critic_lr)
            self.critic_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)

            gradients_all = self.critic_optimizer.compute_gradients(self.critic_loss)

            self.critic_gradient_vars = [v for (g,v) in gradients_all if g is not None]
            self.critic_gradients = self.critic_optimizer.compute_gradients(self.critic_loss,self.critic_gradient_vars)

            self.critic_grads_holder = [(tf.placeholder(tf.float32,shape=g.get_shape()),v) for (g,v) in self.critic_gradients]
            self.critic_train_op = self.critic_optimizer.apply_gradients(self.critic_grads_holder)
    

    def train_model(self):
        print("Master Agents begins to train....")
        while True:
            index = self.global_parameters.get_T() 
            print("Master current episode %d"%index)
            if  index >= self.max_episodes:
                break
            #1. get gradients from global_parameters
            #PIC 
            #P()
            print("Master is waiting for updating parameters by using gradients....")
            actor_gradients,critic_gradients = self.global_parameters.get_A3C_weights_gradients()
            actor_gradients_dict = {}
            critic_gradients_dict = {}
            for i in range(len(self.actor_grads_holder)):
                k = self.actor_grads_holder[i][0]
                actor_gradients_dict[k] = sum([g[i][0] for g in actor_gradients]) 

            for i in range(len(self.critic_grads_holder)):
                k = self.critic_grads_holder[i][0]
                critic_gradients_dict[k] = sum([g[i][0] for g in critic_gradients]) 
            #V()
            #2. update the master agent weights
            #print("Master agent is updating the parameters by using gradients....")
            #print("before decent gradient parameters in Critic: ",self.session.run(self.critic_weights))
            #print("before decent gradient parameters in Actor: ",self.session.run(self.actor_weights))
            self.session.run(self.actor_train_op,feed_dict = actor_gradients_dict)
            self.session.run(self.critic_train_op,feed_dict = critic_gradients_dict)
            #actor_scope = 'actor_network_%s'%self.id
            #critic_scope = 'critic_network_%s'%self.id
            #self.actor_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=actor_scope)
            #self.critic_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=critic_scope)
            #3. update the weights of the global_parameters
            #PIC
            #P()
            #print("After decent gradient before set A3C parameters in Critic: ",self.session.run(self.critic_weights))
            #print("After decent gradient before set A3C parameters in Actor: ",self.session.run(self.actor_weights))
            self.global_parameters.set_A3C_weights(self.actor_weights,self.critic_weights)
            #V()

            print("Master has updated parameters!!")
        self.thread._Thread__stop()
        if self.model_path is not None:
            self.saver.save(self.session,self.model_path)
            print("Master agent has stored the model at %s"%self.model_path)
        print("Master Agent has exited!")
        return


    def choose_action(self,state):
        policy = self.session.run(self.actor_ypredict,feed_dict={self.actor_state_input: state}).flatten() 
        return np.random.choice(self.action_size,1,p=policy)[0]
    
    def update_parameters(self,actor_weights,critic_weights):
        for index in range(len(self.critic_weights)):
            cp_ops = [self.critic_weights[index].assign(critic_weights[index])]
            self.session.run(cp_ops)
        for index in range(len(self.actor_weights)):
            cp_ops = [self.actor_weights[index].assign(actor_weights[index])]
            self.session.run(cp_ops)


    
    def accumulate_gridents(self,experience_list,last_item):
        experience_list.reverse()
        actor_weights_gradients = []
        critic_weights_gradients = [] 
        R = last_item 
        for i in range(1,len(experience_list)):
            R = experience_list[i][2] + self.discount_factor * R
            temp_actor_grads,temp_critic_grads = self.compute_gradients(experience_list[:i],R)
            #print("****************")
            #print("actor grads: ")
            #print temp_actor_grads
            #print("critic grads: ")
            #print temp_critic_grads
            #print("****************")
            actor_weights_gradients.append(temp_actor_grads)
            critic_weights_gradients.append(temp_critic_grads)
            #if len(actor_weights_gradients) == 0 and len(critic_weights_gradients) == 0:
            #    actor_weights_gradients = temp_actor_grads
            #    critic_weights_gradients = temp_critic_grads
            #else:
            #    actor_weights_gradients += temp_actor_grads
            #    critic_weights_gradients += temp_critic_grads


        return actor_weights_gradients,critic_weights_gradients

    def compute_gradients(self,experience_list,R):
        state,action,reward,next_state,done = experience_list[0][0],experience_list[0][1],experience_list[0][2],experience_list[0][3],experience_list[0][4]

        #state value
        critic_ypredict = self.session.run(self.critic_ypredict,feed_dict={self.critic_state_input: state})

        advantages = np.zeros((1,self.action_size))
        critic_ylabel = np.zeros((1,self.value_size))

        advantages[0][action] = R - critic_ypredict
        critic_ylabel[0][0] = R

        temp_critic_grads = self.session.run(self.critic_gradients,feed_dict = {self.critic_state_input: state,self.critic_ylabel: critic_ylabel})

        temp_actor_grads = self.session.run(self.actor_gradients,feed_dict = {self.actor_ylabel: advantages, self.actor_state_input: state,self.actor_action_input:action})

        return temp_actor_grads,temp_critic_grads 

    def create_thread(self):
        self.thread = None
        if self.is_master_process:
            #create slave agents
            self.thread = threading.Thread(target=self.train_model)
        else:
            #create master agents
            self.thread = threading.Thread(target=self.learning)
        self.thread.setDaemon(True)

    def learning(self):
        print("Slave[%s] agent begins to learn...."%self.id)
        max_score_count = 0
        episode_index = 0 
        while True: 
            actor_grads = []
            critic_grads = []
            experience_list = []
            current_step = 0
            #PIC
            #P()
            print"Slave[%s] agent is loading the global parameters...."%self.id
            global_actor_weights,global_critic_weights = self.global_parameters.get_A3C_weights()
            #V()
            #print("Slave[%s] agent has got the global weights... "%self.id)
            #print("global actor weigths:")
            #print global_actor_weights
            #print("global critic weigths:")
            #print global_critic_weights
            if len(global_actor_weights) > 0 and len(global_critic_weights) > 0:
                self.update_parameters(global_actor_weights,global_critic_weights)
                print("Slave agent get the weights!!")
            else:
                print("Slave[%s] agent gets global parameters failed!!"%self.id)

            #get the global episode index
            episode_index = self.global_parameters.get_T() 
            #print("******after update********")
            #print("Salve critic weights: \n",self.session.run(self.critic_weights))
            #print("Salve actor weights: \n",self.session.run(self.actor_weights))
            #print("**************************")

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

                experience_list.append([state,action,reward,next_state,done])
                #reward = reward if not done or score == 499 else -100
                score += reward
                state = next_state
                #if done:
                #    if score == 500:
                #        max_score_count += 1
                #    else:
                #        max_score_count = 0
                current_step += 1
                if current_step >= self.max_steps_each_episode:
                    break

            #compute and accumulate the gradients of weights   
            last_experience = experience_list[:-1]
            last_item = 0
            if last_experience[:-1]:
                #done
                last_item = 0
            else:
                critic_ypredict = self.session.run(self.critic_ypredict,feed_dict={self.critic_state_input: last_experience[0]})
                last_item = critic_ypredict 
            print("Slave[%s] is accumulating the gradients...."%self.id)
            actor_weights_gradients,critic_weights_gradients = self.accumulate_gridents(experience_list[:-1],last_item)

            #PIC
            #P()
            print("Slave[%s] agent is updating the gradients....."%self.id)
            self.global_parameters.set_A3C_weights_gradients(actor_weights_gradients,critic_weights_gradients)
            #V()
            if self.id == "1":
                print("=====================\n\n\nSlaver[%s] agent: Episode[%d] Score = %d\n\n\n================"%(self.id,episode_index,score))
            #save model each 100 episodes
            if max_score_count >= 25 or episode_index >= self.max_episodes:
                break

            #update T = T + 1
            self.global_parameters.accumulate_T()
        
        #stop the thread of agent
        self.thread._Thread__stop()
        print("Slaver[%s] agent has exited!!"%self.id)
        return


class A3C:
    def __init__(self,n_A2Cagents = 5,T_max = 10000,master_model_path = None):
        self.n_A2Cagents = n_A2Cagents
        self.T_max = T_max
        self.session = tf.InteractiveSession()
        self.master_model_path = master_model_path
        self.global_parameters = global_A3C_parameters(sess = self.session) 
        self.slave_agents = self.create_agents(is_master = False)
        self.master_agents = self.create_agents()
        print("[%d] slave agents have created in A3C!!"%n_A2Cagents)
        self.start_agents_threads(self.slave_agents)
        self.start_agents_threads(self.master_agents)

    def start_agents_threads(self,agents):
        for agent in agents:
            agent.thread.start()
        return

    def join_all_agents(self):
        agents = self.slave_agents
        for agent in agents:
            agent.thread.join()

        agents = self.master_agents
        for agent in agents:
            agent.thread.join()
        return

    def create_agents(self,is_master = True):
        agents = []
        if is_master:
            agent = A2Cagent(agent_id = 0,sess = self.session,is_master_process = is_master,global_parameters = self.global_parameters,max_episodes = self.T_max,model_path = self.master_model_path)  
            agent.create_thread()
            agents.append(agent)
            return agents
        n_agents = self.n_A2Cagents + 1 
        for i in range(1,n_agents):
            #create agent
            agent = A2Cagent(agent_id = i,sess = self.session,is_master_process = is_master,global_parameters = self.global_parameters,max_episodes = self.T_max,model_path = self.master_model_path)  
            #create thread
            agent.create_thread()
            agents.append(agent)
        return agents

def main():
    a3c = A3C(n_A2Cagents = 10,T_max = 10000,master_model_path = "./Model/a3c_model.ckpt")
    a3c.join_all_agents()
    print("All agents has exited!!")
    print("main thread is exiting...")
    return

if __name__ == "__main__":
    main()
