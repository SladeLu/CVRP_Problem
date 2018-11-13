#!/usr/bin/env python
# coding: utf-8

# In[155]:


import math
import random
import numpy as np
import copy


# In[45]:


class WPoint():
    def __init__(self,xParam,yParam,weight):
        self.x = xParam
        self.y = yParam
        self.weight = weight

    def __str__(self):
        return "\n(%.2f, %.2f) weight=%.2f"% (self.x ,self.y, self.weight)
    
    def distance (self,pt):
        xDiff = self.x - pt.x
        yDiff = self.y - pt.y
        return math.sqrt(xDiff ** 2 + yDiff ** 2)

    def __repr__(self):
        return str(self)

class WState(WPoint):
    def __init__(self, xParam, yParam, weight, used=False, is_current=False):
        super(WState,self).__init__(xParam = xParam, yParam = yParam, weight = weight)
        self.used = used
        self.is_current = is_current

    def state(self):
        state = (self.x, self.y, self.weight, self.used, self.is_current)
        return np.array(state)
    
    def __setitem__(self, used, v):
            self.used = v
            
    def __str__(self):
        return "\n(%.2f, %.2f) weight=%.2f used:%d current:%d"% (self.x ,self.y, self.weight, self.used, self.is_current)

    def __repr__(self):
        return str(self)


# In[337]:


class TestEnv():


    def __init__(self):
        self.origin_weight = 8.0
        self.full_power = 100.0
        self.speed = 8.0
        self.point_num = 100            #include base
        self.site_num = self.point_num - 1
        self.max_weight = 20.0
        self.max_cargo_weight = 5.0

        self.pointlist = []
        self.point_state = []

        self._creat_map()


        self.state = np.empty((self.point_num,7),dtype=np.float)
        self.reward = 0

    def step(self, action):
        distance = self.pointlist[action].distance(self.pointlist[self.site])
        power_prediction = self.power - self._power_cost(distance, self.weight)
        weight_prediction = self.weight + self.pointlist[action].weight

        if self.point_state[action].used or (action == 0 and self.site == 0):
            self._stay()
        elif power_prediction < 0 or weight_prediction > self.max_weight:
            self._reset_round()
        else:
            self._go_next_point(action)

        return self.state, self.reward, self.done, [self.finished, self.rounds, self.total_distance]

    def reset(self):
        self.done = False
        self.site = 0
        self.finished = 0
        self.rounds = 0
        self.total_distance = 0
        self.round_distance = 0
        self.round_finished = 0
        self.round_reward = 0
        self.round_resets = 0
        self.power = self.full_power

        self._reset_site()
        self.point_state[0].is_current = True
        self.round_state = copy.deepcopy(self.point_state)
        self.load = 0.0
        self._update_weight()
        self._go_base()
        self._get_state()

    def render(self, mode='human', close=False):
        pass

    def _reset_round(self):
        self.point_state = copy.deepcopy(self.round_state)
        self.reward = -1 * self.round_reward
        self.round_reward = 0
        self.finished -= self.round_finished
        self.round_finished = 0
        self.total_distance -= self.round_distance
        self.round_distance = 0
        self.round_resets += 1
        self._go_base()
        self._get_state()

    def _stay(self):
        self.reward = 0

    def _go_next_point(self, point_index):
        self.point_state[self.site].is_current = False
        self.point_state[point_index].is_current = True
        distance = self.pointlist[point_index].distance(self.pointlist[self.site])
        if distance == 0:
            self.reward = 5         # this is a magic number, but its hard to avoid
        else:
            self.reward = 1.0/distance
        self.total_distance += distance
        self.site = point_index

        if point_index == 0:
            self._go_base()
            self.point_state[point_index].used = False
            self.round_state = copy.deepcopy(self.point_state)
            self.round_reward = 0
            self.round_finished = 0
            self.round_distance = 0
            self.rounds += 1

            if self.finished == self.point_num - 1:
                self.done = True
        else:
            self.point_state[point_index].used = True
            self.finished += 1
            self.round_finished += 1
            self.power -= self._power_cost(distance,self.weight)
            self.load += self.pointlist[point_index].weight
            self.round_reward += self.reward
            self.round_distance += distance
            self._update_weight()

        self._get_state()

    def _power_cost(self,d,w):
        return 0.1*d*(w**(2)+w**(1.5))

    def _go_base(self):
        self.site = 0
        self.load = 0
        self._update_weight()
        self.power = self.full_power

    def _update_weight(self):
        self.weight = self.origin_weight + self.load

    def _get_state(self):
        for i in range(self.point_num):
            for j in range(5):
                self.state[i][j] = self.point_state[i].state()[j]

            distance = self.pointlist[i].distance(self.pointlist[self.site])
            self.state[i][5] = self.weight + self.pointlist[i].weight
            self.state[i][6] = self.power - self._power_cost(distance,self.weight)


    def _reset_site(self):
        for i in range(self.point_num):
            self.point_state[i][3] = False
            self.point_state[i][4] = False

    def _creat_map(self):

        zeropoint = WPoint(0,0,0)
        zerostate = WState(0,0,0,False,True)

        # % of the weight of categories  
        weight = np.around(np.random.rand(self.site_num)*self.max_cargo_weight, 2)

        # distance from 0-1
        position = np.around(np.random.rand(self.site_num,2), 2)

        #initial points
        self.pointlist.append(zeropoint)
        self.point_state.append(zerostate)
        for x in range(self.site_num):
            temp = WPoint(position[x][0],position[x][1],weight[x])
            self.pointlist.append(temp)
            temp = WState(position[x][0],position[x][1],weight[x])
            self.point_state.append(temp)


# In[338]:


env = TestEnv()


# In[344]:


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

