#!/usr/bin/python3

from agent_with_DQN import *
import gym

def main():
    env = gym.make("Acrobot-v1")

    cartpoleagent = Agent(env = env,max_episodes = 10000,learning_rate = 0.005,hidden_dim = 20,replay_size = 1000,batch_size = 64,max_step_each_episode = 1000)
    cartpoleagent.learning()
    return

if __name__ == "__main__":
    main()
