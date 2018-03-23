#!/usr/bin/python
#--coding: utf8--
'''
    使用动态规划法通过迭代评估给定策略。
    本程序中给定一个随机策略，评估4*4方格世界中各个状态的价值函数
    状态空间S：S1-S14为非终止状态；S0，S15为终止状态
    行为空间A：{n,e,s,w}对于任何非终止状态都有四个方向的移动行为
    转移概率P：任何能够离开方格世界的动作，其位置将不会发生变化；其他条件下将100%的状态到动作指向的位置
    即时奖励R：任何非终止状态之间的转移得到的即时奖励均为-1，进入终止状态的即时奖励为0
    衰减系数：1
    当前策略：随机策略，任何一个非终止状态下所有行为的概率相同
    状态更新公式为使用贝尔曼期望公式
'''
states = [i for i in range(16)]
values = [0 for _ in range(16)]
actions = ["n","e","s","w"]
da_actions = {"n":-4,"e":1,"s":4,"w":-1}

gamma = 1.00


def nextState(s,a):
    '''
        根据当前状态和采取的行为计算下一个状态id以及得到的即时奖励
    '''
    next_state = s
    #判断当前状态和行为是否会走出方格世界
    if(s%4 == 0 and a == "w") or (s < 4 and a == "n") or ((s+1)%4 == 0 and a == "e") or (s > 11 and a == "s"):
        pass
    else:
        next_state = s + da_actions[a]

    return next_state

def rewardFunction(s):
    '''
        reward function that retures the reward  immediately  when the state s is changed.
    '''
    return 0 if s in [0,15] else -1 

    #当任何状态的立即回报都是-1时，则所有状态的价值都相同
    #return -1

def isTerminateState(s):
    return s in [0,15]

def getSuccessorsState(s):
    successors = []
    if isTerminateState(s):
        successors = [0,0,0,0]
        return successors
    for a in actions:
        next_state = nextState(s,a)
        successors.append(next_state)

    return successors

def updateValue(s):
    successors = getSuccessorsState(s)
    newValue = 0
    reward = rewardFunction(s)
    p_state_action = 0.25
    for next_state in successors:
        newValue += p_state_action * (reward + gamma * values[next_state])

    return newValue

def performOneIteration():
    newValues = [0 for _ in range(16)]
    for s in states:
        newValues[s] = updateValue(s)
    global values
    values = newValues

def printValue(v):
    for i in range(16):
        print "value[%d]: %.5f"%(i,v[i]),
        if (i + 1)% 4 == 0:
            print



def main():
    max_iterate_times = 160
    cur_iterate_times = 0
    printValue(values)
    while cur_iterate_times < max_iterate_times:
        performOneIteration()
        cur_iterate_times += 1
        if cur_iterate_times == 1:
            print "Iteration: %d" % cur_iterate_times
            printValue(values)

    printValue(values)
    return

if __name__ == "__main__":
    main()
