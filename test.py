import gym 
import numpy as np
import random

from cvrp_env import CVRPEnv

env = CVRPEnv()
ob = env.reset()
# env.EnableUI()

while True:
    nextaction = ob.nextaction
    r = random.randint(0,len(nextaction)-1)

    ob, reward, done,info= env.step(nextaction[r])

    if done:
        break

print("route is ",ob.route)
print("Game over,reward is ",reward)