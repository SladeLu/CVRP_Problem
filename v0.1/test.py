import gym 
import numpy as np
import random

from cvrp_env import CVRPEnv
from cvrp_env_v2 import CVRPEnv_v2

def run_CVRP(env):
    step = 0    # 用来控制什么时候学习
    for episode in range(300):
        # 初始化环境
        observation = env.reset()

        while True:
            # 刷新环境
            # env.render()

            # DQN 根据观测值选择行为
            action = RL.choose_action(observation)

            # 环境根据行为给出下一个 state, reward, 是否终止
            observation_, reward, done = env.step(action)

            # DQN 存储记忆
            RL.store_transition(observation, action, reward, observation_)

            # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # 将下一个 state_ 变为 下次循环的 state
            observation = observation_

            # 如果终止, 就跳出循环
            if done:
                break
            step += 1   # 总步数

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = CVRPEnv_v2()

    ob = env.reset()
    print(ob)
    while True:
    #     env.render()
        # nextaction = ob.nextaction
        r = random.randint(0,env.n_action-1)
        print(r)
        ob, reward, done = env.step(ob[r])
        print(ob)
        if done:
            break
    # print(ob)
    # for x in range(1,5):
    # ob,done = env.step(ob[1])
    # print(ob)
    # ob,done = env.step(ob[2])
    # print(ob)
    # ob,done = env.step(ob[1])
    # print(ob)
    # ob = env.step(ob[0])
    # print(ob)
    # print(env.route)

    print("route is ",ob)
    # print("Game over,reward is ",reward)