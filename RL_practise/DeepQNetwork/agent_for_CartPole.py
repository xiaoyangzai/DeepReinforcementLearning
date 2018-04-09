#!/usr/bin/python

from agent_with_DQN import *
import gym

def main():
    env = gym.make("CartPole-v1")

    cartpoleagent = Agent(env = env,max_episodes = 600,learning_rate = 0.001,hidden_dim = 20,replay_size = 10000,batch_size = 50,max_step_each_episode = 5000)
    cartpoleagent.learning()
    return

if __name__ == "__main__":
    main()
