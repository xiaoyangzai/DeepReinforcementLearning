#!/usr/bin/python3

from gridworld import *
from agent_with_DQN import *

def main():
    env = GridWorldEnv(n_width = 10,n_height=5,u_size=60,default_reward = -1,windy=False)
    env.start = (0,0)
    env.ends= [(9,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0)]
    env.rewards = [(9,0,10),(1,0,-10),(2,0,-10),(3,0,-10),(4,0,-10),(5,0,-10),(6,0,-10),(7,0,-10),(8,0,-10)]
    env.refresh_setting()
    gridagent = Agent(env = env,max_episodes = 3000,hidden_dim = 20,replay_size = 10000,learning_rate = 0.005,batch_size = 60)
    gridagent.learning()
    return

if __name__ == "__main__":
    main()
