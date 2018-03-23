#!/usr/bin/python3

from agent_with_DQN import *
import gym

def main():
    env = gym.make("CartPole-v0")

    cartpoleagent = Agent(env = env,max_episodes = 1000,learning_rate = 0.0005,hidden_dim = 20,replay_size = 10000,batch_size = 50,max_step_each_episode = 200)
    cartpoleagent.learning()
    return

if __name__ == "__main__":
    main()
