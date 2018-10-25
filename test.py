import gym 
import numpy as np
import my_env
import random


env = gym.make('CVRPEnv-v0')
env.reset()
ob= env.getState()
print(ob)
while True:
    r = random.randint(0,len(ob)-1)
    ob, reward, done,info= env.step(ob[r])
    # print(ob)
    if done:
        break
print("Game over,reward is ",reward)
