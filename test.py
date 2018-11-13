#!/usr/bin/env python
# coding: utf-8

# In[4]:


from test_env import TestEnv
import random
import numpy as np


# In[5]:


if __name__ == "__main__":
    env = TestEnv()


# In[9]:


env.reset()
ob = np.empty((100, 7), dtype=np.float)
i = 0
total_reward = 0.0
while True:
    i += 1
    r = random.randint(0,99)
    ob, reward, done, _ = env.step(r)
    total_reward += reward
    if done:
        print(i,total_reward,done,_)
        break;

